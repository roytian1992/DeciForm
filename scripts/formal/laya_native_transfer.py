"""Task contrasts and geometry in Laya's native trainable paths.

The shared full-rank W learns shared adaptation. Only centered task contrasts
B R_t A are added. Geometry transforms the existing scorer's normalized input;
there is no second bilinear scorer. Zero task maps preserve initial predictions
and receive gradients immediately through nonzero shared bases.
"""
import json
import math
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from laya.common import DecisionModel, QTYPES, parallel_option_masks, _autocast_enabled
from laya_mechanisms import bounded_centered, TASK_WEIGHTS


class TaskContrastLinear(nn.Linear):
    def forward(self, x):
        router = self._transfer_router_ref()
        base = F.linear(x, self.weight, self.bias)
        if router.audit_disabled:
            return base
        if router.maps is None or x.ndim != 3 or x.shape[0] != router.batch_size:
            raise RuntimeError('Missing task contrast context')
        low = self.transfer_A(x)
        contrast = torch.bmm(low, router.maps[self.transfer_index].to(low.dtype).transpose(-2,-1))
        return base + self.transfer_B(contrast)*self.transfer_scale


class TaskContrastRouter(nn.Module):
    def __init__(self, encoder, rank=16, alpha=32, radius=.5):
        super().__init__()
        targets=[]
        for i,layer in enumerate(encoder.layers):
            for name,mod in [('attn.Wqkv',layer.attn.Wqkv),('attn.Wo',layer.attn.Wo),
                             ('mlp.Wi',layer.mlp.Wi),('mlp.Wo',layer.mlp.Wo)]:
                if type(mod) is not nn.Linear:
                    raise TypeError('Expected unmodified native linear layer')
                targets.append((f'layers.{i}.{name}',mod))
        self.raw=nn.Parameter(torch.zeros(len(targets),3,rank,rank))
        self.register_buffer('task_weights',torch.tensor(TASK_WEIGHTS))
        self.radius=radius
        self.module_names=[n for n,_ in targets]
        self.maps,self.batch_size=None,None
        self.audit_disabled=False
        for i,(_,mod) in enumerate(targets):
            mod.__class__=TaskContrastLinear
            mod.transfer_A=nn.Linear(mod.in_features,rank,bias=False)
            mod.transfer_B=nn.Linear(rank,mod.out_features,bias=False)
            nn.init.orthogonal_(mod.transfer_A.weight)
            nn.init.orthogonal_(mod.transfer_B.weight)
            mod.transfer_index,mod.transfer_scale=i,alpha/rank
            object.__setattr__(mod,'_transfer_router_ref',weakref.ref(self))

    def prepare(self,tasks,dtype):
        if self.maps is not None:
            raise RuntimeError('Uncleared task context')
        self.maps=bounded_centered(self.raw,self.task_weights,self.radius)[:,tasks].to(dtype)
        self.batch_size=len(tasks)

    def clear(self):
        self.maps,self.batch_size=None,None

    def penalty(self):
        c=self.raw-(self.raw*self.task_weights[None,:,None,None]).sum(1,keepdim=True)
        return c.square().sum((-2,-1)).mean()


class NativeGeometryReadout(nn.Module):
    def __init__(self,hidden_size,projection=256,quadratic_rank=64):
        super().__init__()
        if projection>hidden_size:
            raise ValueError('Geometry basis must fit hidden size')
        # A fixed orthonormal basis makes ||P^T H_t P||_2 <= ||H_t||_F <= .5.
        # The fully trainable encoder/scorer can learn coordinates in this basis.
        basis=torch.empty(projection,hidden_size)
        nn.init.orthogonal_(basis)
        self.register_buffer('geometry_basis',basis)
        self.geometry_raw=nn.Parameter(torch.zeros(3,projection,projection))
        self.register_buffer('task_weights',torch.tensor(TASK_WEIGHTS))
        self.project=nn.Linear(hidden_size,quadratic_rank,bias=False)
        self.query_condition=nn.Linear(hidden_size,quadratic_rank,bias=False)
        self.qf_out=nn.Linear(quadratic_rank,1,bias=False)
        nn.init.zeros_(self.qf_out.weight)
        self.rank,self.cap=quadratic_rank,4.

    def geometry(self,normalized,tasks):
        with torch.autocast(normalized.device.type,enabled=False):
            h=normalized.float()
            maps=bounded_centered(self.geometry_raw,self.task_weights)[tasks]
            low=F.linear(h,self.geometry_basis)
            delta=torch.bmm(low,maps.transpose(-2,-1)) @ self.geometry_basis
            return h+delta

    def quadratic(self,query,normalized,tasks,valid):
        result=normalized.new_zeros(valid.shape,dtype=torch.float32)
        choices=(tasks==QTYPES['choice']).nonzero().flatten()
        if choices.numel():
            with torch.autocast(normalized.device.type,enabled=False):
                c=self.project(normalized[choices].float())
                q=self.query_condition(F.layer_norm(query[choices].float(),(query.shape[-1],)))
                z=F.layer_norm(c-q[:,None],(self.rank,))
                # Centering removes the common quadratic intercept; the sqrt(2)
                # normalizer is the Gaussian squared-feature variance convention.
                phi=(z.square()-1)/math.sqrt(2)
                raw=self.qf_out(phi).squeeze(-1)/math.sqrt(self.rank)
                result=result.index_add(0,choices,self.cap*torch.tanh(raw/self.cap))
        return result.masked_fill(~valid,0.)

    def penalty(self):
        c=self.geometry_raw-(self.geometry_raw*self.task_weights[:,None,None]).sum(0,keepdim=True)
        return c.square().sum((-2,-1)).mean()


class NativeTransferDecisionModel(DecisionModel):
    def clear_transfer_context(self):
        self.transfer_router.clear()

    def regularization(self):
        return 1e-3*(self.transfer_router.penalty()+self.transfer_readout.penalty())

    def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype,
                detach_encoder=False,position_ids=None,option_ids=None):
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.transfer_router.raw.dtype
        self.transfer_router.prepare(qtype,dtype)
        masks=attention_mask if option_ids is None else parallel_option_masks(
            attention_mask,position_ids,option_ids,getattr(self.encoder.config,'sliding_window',None))
        extra={} if position_ids is None else {'position_ids':position_ids}
        h=self.encoder(input_ids=input_ids,attention_mask=masks,**extra).last_hidden_state
        if detach_encoder:
            h=h.detach()
        h=h+self.type_emb(qtype)[:,None]
        if self.head is not None:
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h=checkpoint(layer,h,src_key_padding_mask=~attention_mask.bool(),use_reentrant=False)
                else:
                    h=layer(h,src_key_padding_mask=~attention_mask.bool())
        markers=h.gather(1,marker_pos.clamp(min=0)[:,:,None].expand(-1,-1,h.shape[-1]))
        normalized=self.scorer[0](markers)
        transformed=self.transfer_readout.geometry(normalized,qtype)
        logits=self.scorer[3](self.scorer[2](self.scorer[1](transformed))).squeeze(-1).float()
        logits=logits+self.transfer_readout.quadratic(h[:,0],transformed,qtype,marker_mask)
        logits=logits.masked_fill(~marker_mask,-1e4)
        p=logits.detach().softmax(-1)
        k=marker_mask.sum(-1).clamp(min=2).float()
        entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)/k.log()
        top=p.topk(min(2,p.shape[-1]),-1).values
        if top.shape[-1]==1:
            top=F.pad(top,(0,1))
        features=torch.stack([top[:,0],top[:,0]-top[:,1],entropy,k/255.],-1)
        act=torch.cat([h[:,0].float(),features],-1)
        if not _autocast_enabled(h.device.type):
            act=act.to(self.act_head[0].weight.dtype)
        act=self.act_head(act)
        if not torch.is_grad_enabled():
            self.clear_transfer_context()
        return logits,act


def is_transfer_parameter(name):
    return 'transfer_A.' in name or 'transfer_B.' in name or name.startswith(('transfer_router.','transfer_readout.'))


def original_hashes(model):
    from laya_mechanisms import original_hashes as original
    return original(model)


def install_mechanisms(model,rank=16,alpha=32,projection=256,quadratic_rank=64):
    if type(model) is not DecisionModel or next(model.parameters()).is_cuda:
        raise ValueError('Install on fresh native CPU model')
    before=original_hashes(model)
    with torch.random.fork_rng(devices=[]):
        model.transfer_router=TaskContrastRouter(model.encoder,rank,alpha)
        model.transfer_readout=NativeGeometryReadout(model.encoder.config.hidden_size,projection,quadratic_rank)
    model.__class__=NativeTransferDecisionModel
    assert original_hashes(model)==before
    return model


def optimizer_groups(model,cfg):
    groups={'encoder':[],'head':[],'task_maps':[],'quadratic':[]}
    for n,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('encoder.'):
            group='encoder'
        elif n in ['transfer_router.raw','transfer_readout.geometry_raw']:
            group='task_maps'
        elif n.startswith('transfer_readout.'):
            group='quadratic'
        else:
            group='head'
        groups[group].append(p)
    rates={'encoder':cfg['encoder_lr'],'head':cfg['head_lr'],
           'task_maps':cfg['module_lr'],'quadratic':cfg['module_lr']}
    return [{'name':n,'params':ps,'lr':rates[n],'weight_decay':0. if n=='task_maps' else .01} for n,ps in groups.items()]


def inventory(model):
    added=sum(p.numel() for n,p in model.named_parameters() if is_transfer_parameter(n))
    return {'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'added_parameters':added,'task_order':['choice','score','noul'],
        'full_encoder_finetuning':True,'original_scorer_retained':True,
        'routing':'centered task-only B R_t A; no redundant shared BA branch',
        'geometry':'native scorer LN input + orthonormal subspace task transform',
        'initial_residual':'zero task maps and QF output; nonzero routing bases',
        'structural_regularization':1e-3,'data_loss':'unchanged upstream RLCD',
        'candidate_independence':False,'layout':'original parallel Laya'}


@torch.no_grad()
def verify_initial_equivalence(model,args,kwargs):
    assert torch.count_nonzero(model.transfer_router.raw)==0
    assert torch.count_nonzero(model.transfer_readout.geometry_raw)==0
    assert torch.count_nonzero(model.transfer_readout.qf_out.weight)==0
    actual=model(*args,**kwargs)
    model.transfer_router.audit_disabled=True
    try:
        reference=DecisionModel.forward(model,*args,**kwargs)
    finally:
        model.transfer_router.audit_disabled=False
    deltas=[(a-b).abs().max().item() for a,b in zip(actual,reference)]
    if max(deltas)>1e-5:
        raise AssertionError(f'Initial native outputs differ: {deltas}')
    return {'max_logit_difference':deltas[0],'max_act_difference':deltas[1]}


def load_transfer_checkpoint(path,device='cpu'):
    from laya.common import build_model
    from safetensors.torch import load_file
    path=Path(path)
    cfg=json.loads((path/'transfer_config.json').read_text())
    model=build_model(cfg['model'],encoder_dir=str(path/'encoder'),pretrained=False)
    install_mechanisms(model,**cfg['mechanisms'])
    model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    return model.to(device)
