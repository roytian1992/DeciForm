"""D2 mechanisms transferred onto a pinned, fully fine-tuned Laya baseline.

No candidate resampling, masking, loss, or backbone replacement. The existing
Laya scorer remains live. Added branches initially contribute exactly zero.
Routing context persists until backward completes for checkpoint recomputation.
"""
import hashlib
import math
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from laya.common import DecisionModel, QTYPES, parallel_option_masks, _autocast_enabled

TASK_WEIGHTS = (.25, .375, .375)  # Laya order: choice, score, noul.


def bounded_centered(raw, weights, radius=.5):
    centered = raw - (raw * weights[..., None, None]).sum(-3, keepdim=True)
    denominator = centered.square().sum((-2, -1)).amax(-1, keepdim=True).clamp_min(1.).sqrt()
    return radius * centered / denominator[..., None, None]


class RoutedProjection(nn.Linear):
    def forward(self, x):
        router = self._transfer_router_ref()
        if router.audit_disabled:
            return F.linear(x,self.weight,self.bias)
        if router.maps is None or x.ndim != 3 or x.shape[0] != router.batch_size:
            raise RuntimeError('Missing batch-aligned transfer routing context')
        original = F.linear(x, self.weight, self.bias)
        low = self.transfer_A(x)
        matrix = router.maps[self.transfer_index].to(low.dtype)
        low = low + torch.bmm(low, matrix.transpose(-2, -1))
        return original + self.transfer_B(low) * self.transfer_scale


class TaskRouter(nn.Module):
    def __init__(self, encoder, rank=16, alpha=32, radius=.5):
        super().__init__()
        projections = []
        for layer_index, layer in enumerate(encoder.layers):
            for family, module in [('attn.Wqkv', layer.attn.Wqkv), ('attn.Wo', layer.attn.Wo),
                                   ('mlp.Wi', layer.mlp.Wi), ('mlp.Wo', layer.mlp.Wo)]:
                if type(module) is not nn.Linear:
                    raise TypeError('Transfer requires original plain linear projections')
                projections.append((f'layers.{layer_index}.{family}', module))
        self.rank, self.radius = rank, radius
        self.module_names = [name for name, _ in projections]
        self.raw = nn.Parameter(torch.zeros(len(projections), 3, rank, rank))
        self.register_buffer('task_weights', torch.tensor(TASK_WEIGHTS))
        self.maps, self.batch_size = None, None
        self.audit_disabled = False
        for index, (_, module) in enumerate(projections):
            module.__class__ = RoutedProjection
            module.transfer_A = nn.Linear(module.in_features, rank, bias=False)
            module.transfer_B = nn.Linear(rank, module.out_features, bias=False)
            nn.init.kaiming_uniform_(module.transfer_A.weight, a=math.sqrt(5))
            nn.init.zeros_(module.transfer_B.weight)
            module.transfer_index, module.transfer_scale = index, alpha/rank
            object.__setattr__(module, '_transfer_router_ref', weakref.ref(self))

    def prepare(self, tasks, dtype):
        if self.maps is not None:
            raise RuntimeError('Clear transfer context after forward/backward before reuse')
        self.maps = bounded_centered(self.raw, self.task_weights, self.radius)[:, tasks].to(dtype)
        self.batch_size = len(tasks)

    def clear(self):
        self.maps, self.batch_size = None, None


class GeometryQuadraticResidual(nn.Module):
    def __init__(self, hidden_size, projection=256, quadratic_rank=64):
        super().__init__()
        self.q = nn.Linear(hidden_size, projection, bias=False)
        self.k = nn.Linear(hidden_size, projection, bias=False)
        # Shared bilinear branch receives a nonzero gradient immediately;
        # geometry learns after k leaves zero. Never initialize both factors zero.
        nn.init.zeros_(self.k.weight)
        self.geometry_raw = nn.Parameter(torch.zeros(3, projection, projection))
        self.register_buffer('task_weights', torch.tensor(TASK_WEIGHTS))
        self.project = nn.Linear(hidden_size, quadratic_rank, bias=False)
        self.query_condition = nn.Linear(projection, quadratic_rank, bias=False)
        self.qf_out = nn.Linear(quadratic_rank, 1, bias=False)
        nn.init.zeros_(self.qf_out.weight)
        self.projection, self.quadratic_rank, self.cap = projection, quadratic_rank, 4.

    def forward(self, query, candidates, tasks, valid):
        with torch.autocast(device_type=query.device.type, enabled=False):
            query, candidates = query.float(), candidates.float()
            q, k = self.q(query), self.k(candidates)
            geometry = bounded_centered(self.geometry_raw, self.task_weights)[tasks]
            q = q + torch.bmm(geometry.transpose(-2,-1), q.unsqueeze(-1)).squeeze(-1)
            result = (k * q[:,None]).sum(-1)/math.sqrt(self.projection)
            choices = (tasks == QTYPES['choice']).nonzero().flatten()
            if len(choices):
                projected = self.project(F.layer_norm(candidates[choices], (candidates.shape[-1],)))
                reference = self.query_condition(F.layer_norm(q[choices], (self.projection,)))
                z = F.layer_norm(projected-reference[:,None], (self.quadratic_rank,))
                delta = self.qf_out(z.square()).squeeze(-1)/math.sqrt(self.quadratic_rank)
                delta = self.cap * (delta/self.cap).tanh()
                result = result.index_add(0, choices, delta)
            return result.masked_fill(~valid, 0.)


class MechanismDecisionModel(DecisionModel):
    def clear_transfer_context(self):
        self.transfer_router.clear()

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                detach_encoder=False, position_ids=None, option_ids=None):
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.transfer_router.raw.dtype
        self.transfer_router.prepare(qtype, dtype)
        if option_ids is None:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            masks = parallel_option_masks(attention_mask, position_ids, option_ids,
                                          getattr(self.encoder.config, 'sliding_window', None))
            h = self.encoder(input_ids=input_ids, attention_mask=masks, position_ids=position_ids).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:,None,:]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:,:,None].expand(-1,-1,h.size(-1))
        markers = torch.gather(h, 1, idx)
        logits = self.scorer(markers).squeeze(-1).float()
        logits = logits + self.transfer_readout(h[:,0], markers, qtype, marker_mask)
        logits = logits.masked_fill(~marker_mask, -1e4)
        # Keep original auxiliary act output exact; no act supervision is added.
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)/torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2,-1).values
        else:
            top1 = p.topk(1,-1).values
            top2 = torch.cat([top1,torch.zeros_like(top1)],-1)
        feats = torch.stack([top2[:,0],top2[:,0]-top2[:,1],ent,k/255.],-1)
        act_input = torch.cat([h[:,0].float(),feats],-1)
        if not _autocast_enabled(h.device.type):
            act_input = act_input.to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        # During training keep graph/context for encoder checkpoint replay.
        if not torch.is_grad_enabled():
            self.clear_transfer_context()
        return logits, act_logits


def is_transfer_parameter(name):
    return ('transfer_A.' in name or 'transfer_B.' in name or
            name.startswith(('transfer_router.','transfer_readout.')))


def original_hashes(model):
    def digest(items):
        h=hashlib.sha256()
        for name,p in items:
            h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    return {'encoder_sha256':digest((n,p) for n,p in model.encoder.named_parameters()
                                    if 'transfer_A.' not in n and 'transfer_B.' not in n),
            'head_sha256':digest((n,p) for n,p in model.named_parameters()
                                 if not n.startswith('encoder.') and not is_transfer_parameter(n))}


def install_mechanisms(model, *, rank=16, alpha=32, projection=256, quadratic_rank=64):
    if type(model) is not DecisionModel:
        raise TypeError('Install only on a fresh upstream Laya DecisionModel')
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('Install on CPU before moving the model, preserving baseline RNG')
    before = original_hashes(model)
    with torch.random.fork_rng(devices=[]):
        model.transfer_router = TaskRouter(model.encoder,rank,alpha)
        model.transfer_readout = GeometryQuadraticResidual(model.encoder.config.hidden_size,projection,quadratic_rank)
    model.__class__ = MechanismDecisionModel
    assert original_hashes(model) == before, 'Original backbone/head parameters changed during installation'
    return model


def inventory(model):
    total = sum(p.numel() for p in model.parameters())
    added = {name:p.numel() for name,p in model.named_parameters() if is_transfer_parameter(name)}
    groups = {'routed_low_rank':sum(n for k,n in added.items() if 'transfer_A.' in k or 'transfer_B.' in k),
              'task_routing':model.transfer_router.raw.numel(),
              'head_geometry_and_shared_bilinear':sum(p.numel() for n,p in model.transfer_readout.named_parameters()
                                                    if n.startswith(('q.','k.','geometry_raw'))),
              'choice_qf':sum(p.numel() for n,p in model.transfer_readout.named_parameters()
                              if n.startswith(('project.','query_condition.','qf_out.')))}
    assert sum(groups.values()) == sum(added.values())
    return {'total_parameters':total,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'added_parameters':sum(added.values()),'added_groups':groups,
            'targets':model.transfer_router.module_names,'task_order':['choice','score','noul'],
            'full_encoder_finetuning':True,'original_scorer_retained':True,'initial_residual':'zero',
            'routing_radius':.5,'geometry_radius':.5,'no_extra_task_loss':True}


def load_transfer_checkpoint(path, device='cpu'):
    import json
    from laya.common import build_model
    from safetensors.torch import load_file
    path = Path(path)
    cfg = json.loads((path/'transfer_config.json').read_text())
    model = build_model(cfg['model'],encoder_dir=str(path/'encoder'),pretrained=False)
    install_mechanisms(model,**cfg['mechanisms'])
    model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    return model.to(device)


@torch.no_grad()
def verify_initial_equivalence(model, args, kwargs):
    if model.training:
        raise ValueError('Initial equivalence audit requires eval mode')
    zeros=[p for n,p in model.named_parameters() if 'transfer_B.weight' in n]
    zeros += [model.transfer_readout.k.weight,model.transfer_readout.qf_out.weight]
    assert all(torch.count_nonzero(p).item()==0 for p in zeros)
    actual=model(*args,**kwargs)
    model.transfer_router.audit_disabled=True
    try:
        reference=DecisionModel.forward(model,*args,**kwargs)
    finally:
        model.transfer_router.audit_disabled=False
    deltas=[float((a-b).abs().max()) for a,b in zip(actual,reference)]
    if max(deltas)>1e-5:
        raise AssertionError(f'Initial logits differ from original Laya: {deltas}')
    return {'max_logit_difference':deltas[0],'max_act_difference':deltas[1],
            'reference':'unmodified upstream DecisionModel.forward and original linear projections',
            'all_added_output_projections_zero':True}
