import copy
import itertools
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/formal'),str(ROOT/'runs/formal_laya_20261106_v1/vendor/laya_release')]
from laya.common import DecisionModel, collate_items, QTYPES
from laya_adapter import make_parts, encode_parts, canonical_probabilities
from laya_mechanisms import install_mechanisms, bounded_centered, inventory, original_hashes
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

@pytest.fixture(scope='module')
def tok():
    torch.set_num_threads(2)
    return AutoTokenizer.from_pretrained(ROOT/'runs/formal_laya_20261106_v1/base',local_files_only=True)

def baseline():
    torch.manual_seed(18)
    cfg=ModernBertConfig(vocab_size=50368,hidden_size=64,intermediate_size=128,num_hidden_layers=3,
                        num_attention_heads=4,local_attention=8,global_attn_every_n_layers=2,
                        pad_token_id=50283,attn_implementation='sdpa')
    return DecisionModel(ModernBertModel(cfg),head_layers=2)

def row(task='choice',k=3):
    r={'id':'test/1','group_id':'test/1','source_family':'test','task':task,
       'state':{'text':'A detailed billing complaint and request for a refund.'},'label':'billing',
       'json_keys':['c'+str(i) for i in range(k)],
       'question':{'instructions':'Select the category.','options':['billing']+['type '+str(i) for i in range(k-1)]}}
    if task=='noul':r.update(question={'instructions':'Is this billing?'},label=True)
    if task=='score':r.update(question={'instructions':'Rate quality.','scale':{'min':1,'max':5}},label=4)
    return r

def batch(tok,layout='parallel',rows=None,orders=None):
    rows=rows or [row(t) for t in ('choice','score','noul')]
    orders=orders or [None]*len(rows)
    return collate_items([[encode_parts(make_parts(r,tok),layout,o) for r,o in zip(rows,orders)]],tok.pad_token_id)

def forward(model,b):
    return model(*(b[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')),
                 **{k:b[k] for k in ('position_ids','option_ids') if k in b})

def excite(model):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'transfer_B.weight' in name or name.startswith('transfer_readout.k.') or name=='transfer_readout.qf_out.weight':
                p.normal_(std=.02)
        model.transfer_router.raw.normal_(std=.05)
        model.transfer_readout.geometry_raw.normal_(std=.01)

@pytest.mark.parametrize('layout',['sequential','parallel'])
def test_original_weights_rng_logits_and_variable_support_preserved(tok,layout):
    original=baseline().eval();model=copy.deepcopy(original)
    before=original_hashes(original);rng=torch.get_rng_state().clone()
    install_mechanisms(model,projection=16,quadratic_rank=8)
    assert torch.equal(rng,torch.get_rng_state()) and original_hashes(model)==before
    for k in (2,7,77):
        b=batch(tok,layout,[row('choice',k),row('score'),row('noul')])
        with torch.no_grad():
            old=forward(original,b);new=forward(model,b)
        for a,z in zip(old,new):torch.testing.assert_close(a,z,atol=0,rtol=0)
        assert model.transfer_router.maps is None
    info=inventory(model);assert info['added_parameters']>0 and info['full_encoder_finetuning']


def test_centering_radius_and_gradients():
    raw=torch.randn(12,3,16,16,requires_grad=True);weights=torch.tensor([.25,.375,.375])
    mapped=bounded_centered(raw,weights)
    assert (mapped*weights[None,:,None,None]).sum(1).abs().max()<1e-6
    assert mapped.square().sum((-2,-1)).sqrt().max()<=.500001
    mapped[0,0].square().sum().backward();assert torch.isfinite(raw.grad).all()

@pytest.mark.parametrize('checkpointing',[False,True])
def test_all_mechanisms_receive_gradients_with_encoder_checkpointing(tok,checkpointing):
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).train();excite(model)
    if checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.head_checkpointing=True
    b=batch(tok)
    logits,_=forward(model,b)
    F.cross_entropy(logits,b['target'].argmax(-1)).backward()
    model.clear_transfer_context()
    names=['encoder.layers.0.attn.Wqkv.weight','encoder.layers.0.attn.Wqkv.transfer_A.weight',
           'encoder.layers.0.attn.Wqkv.transfer_B.weight','transfer_router.raw',
           'transfer_readout.geometry_raw','transfer_readout.qf_out.weight','scorer.3.weight']
    for name in names:
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    assert model.transfer_router.maps is None


def test_zero_initialized_branches_can_start_learning(tok):
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).train()
    logits,_=forward(model,batch(tok))
    logits[:,0].sum().backward();model.clear_transfer_context()
    assert model.transfer_readout.k.weight.grad.abs().sum()>0
    assert model.transfer_readout.qf_out.weight.grad.abs().sum()>0
    assert model.encoder.layers[0].attn.Wqkv.transfer_B.weight.grad.abs().sum()>0


def test_parallel_equivariance_after_nonzero_mechanism_updates(tok):
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).eval();excite(model)
    def evaluate(order):
        with torch.no_grad():z,_=forward(model,batch(tok,rows=[row()],orders=[list(order)]))
        return torch.tensor(canonical_probabilities(z[0],list(order)))
    reference=evaluate([0,1,2])
    for order in itertools.permutations(range(3)):
        torch.testing.assert_close(reference,evaluate(order),atol=2e-6,rtol=2e-5)


def test_state_dict_reconstructs_all_added_parameters(tok):
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).eval();excite(model)
    restored=install_mechanisms(baseline(),projection=16,quadratic_rank=8).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        for a,b in zip(forward(model,batch(tok)),forward(restored,batch(tok))):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_qf_only_changes_choice():
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8)
    head=model.transfer_readout;h=torch.randn(3,64);c=torch.randn(3,5,64);tasks=torch.tensor([0,1,2]);valid=torch.ones(3,5,dtype=torch.bool)
    original=head(h,c,tasks,valid)
    nn.init.normal_(head.qf_out.weight)
    altered=head(h,c,tasks,valid)
    assert (original[0]-altered[0]).abs().sum()>0
    torch.testing.assert_close(original[1:],altered[1:],atol=0,rtol=0)


def test_baseline_audit_uses_unmodified_upstream_forward(tok):
    from laya_mechanisms import verify_initial_equivalence
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).eval()
    b=batch(tok)
    args=tuple(b[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype'))
    extra={k:b[k] for k in ('position_ids','option_ids') if k in b}
    proof=verify_initial_equivalence(model,args,extra)
    assert proof['max_logit_difference']==proof['max_act_difference']==0
    assert not model.transfer_router.audit_disabled and model.transfer_router.maps is None


def test_export_loader_recovers_real_logits(tok,tmp_path):
    import json
    from safetensors.torch import save_file
    from laya_mechanisms import load_transfer_checkpoint
    model=install_mechanisms(baseline(),projection=16,quadratic_rank=8).eval();excite(model)
    model.encoder.config.save_pretrained(tmp_path/'encoder')
    cfg={'model':{'encoder':str(tmp_path/'encoder'),'head_layers':2,'act_costs':{'escalate':1}},
         'mechanisms':{'rank':16,'alpha':32,'projection':16,'quadratic_rank':8}}
    (tmp_path/'transfer_config.json').write_text(json.dumps(cfg))
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(tmp_path/'model.safetensors'))
    restored=load_transfer_checkpoint(tmp_path).eval()
    with torch.no_grad():
        for a,b in zip(forward(model,batch(tok)),forward(restored,batch(tok))):torch.testing.assert_close(a,b,atol=0,rtol=0)
