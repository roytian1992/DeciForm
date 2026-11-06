import copy
import itertools
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/formal'),str(ROOT/'tests'),str(ROOT/'runs/formal_laya_20261106_v1/vendor/laya_release')]
from test_laya_mechanisms import baseline,row,batch,forward
from laya_adapter import canonical_probabilities
from laya_native_transfer import install_mechanisms,verify_initial_equivalence,original_hashes,optimizer_groups,load_transfer_checkpoint
from transformers import AutoTokenizer


@pytest.fixture(scope='module')
def tok():
    torch.set_num_threads(2)
    return AutoTokenizer.from_pretrained(ROOT/'runs/formal_laya_20261106_v1/base',local_files_only=True)


@pytest.mark.parametrize('layout',['parallel','sequential'])
def test_initial_outputs_rng_and_original_parameters(tok,layout):
    original=baseline().eval()
    model=copy.deepcopy(original)
    before=torch.get_rng_state().clone()
    install_mechanisms(model,rank=8,projection=16,quadratic_rank=8)
    assert torch.equal(before,torch.get_rng_state())
    assert original_hashes(model)==original_hashes(original)
    for k in [2,7,77]:
        b=batch(tok,layout,[row('choice',k),row('noul'),row('score')])
        with torch.no_grad():
            x=forward(model,b);y=forward(original,b)
        for a,z in zip(x,y):torch.testing.assert_close(a,z,atol=0,rtol=0)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_task_maps_and_qf_receive_initial_gradients(tok,checkpointing):
    model=install_mechanisms(baseline(),rank=8,projection=16,quadratic_rank=8).train()
    if checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.head_checkpointing=True
    b=batch(tok)
    z,_=forward(model,b)
    (nn.functional.cross_entropy(z,b['target'].argmax(-1))+model.regularization()).backward()
    for n in ['transfer_router.raw','transfer_readout.geometry_raw','transfer_readout.qf_out.weight','scorer.1.weight']:
        g=dict(model.named_parameters())[n].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,n
    model.clear_transfer_context()


def test_geometry_operator_and_task_contrast_constraints(tok):
    from laya_mechanisms import bounded_centered
    model=install_mechanisms(baseline(),rank=8,projection=16,quadratic_rank=8)
    with torch.no_grad():
        model.transfer_router.raw.normal_()
        model.transfer_readout.geometry_raw.normal_()
    head=model.transfer_readout
    basis=head.geometry_basis
    torch.testing.assert_close(basis@basis.T,torch.eye(16),atol=1e-6,rtol=1e-6)
    maps=bounded_centered(head.geometry_raw,head.task_weights)
    full=basis.T@maps@basis
    assert torch.linalg.matrix_norm(full,ord=2).max()<=.500001
    assert (full*head.task_weights[:,None,None]).sum(0).abs().max()<1e-6
    maps=bounded_centered(model.transfer_router.raw,head.task_weights)
    assert (maps*head.task_weights[None,:,None,None]).sum(1).abs().max()<1e-6


def test_nonzero_parallel_equivariance_and_serialization(tok,tmp_path):
    from safetensors.torch import save_file
    import json
    model=install_mechanisms(baseline(),rank=8,projection=16,quadratic_rank=8).eval()
    with torch.no_grad():
        model.transfer_router.raw.normal_(std=.01)
        model.transfer_readout.geometry_raw.normal_(std=.01)
        model.transfer_readout.qf_out.weight.normal_(std=.01)
    reference=None
    for order in itertools.permutations(range(3)):
        b=batch(tok,rows=[row()],orders=[list(order)])
        with torch.no_grad():z,_=forward(model,b)
        p=torch.tensor(canonical_probabilities(z[0],list(order)))
        if reference is None:reference=p
        else:torch.testing.assert_close(p,reference,atol=3e-6,rtol=2e-5)
    model.encoder.config.save_pretrained(tmp_path/'encoder')
    (tmp_path/'transfer_config.json').write_text(json.dumps({'model':{'encoder':str(tmp_path/'encoder'),'head_layers':2,'act_costs':{'escalate':1}},
        'mechanisms':{'rank':8,'alpha':32,'projection':16,'quadratic_rank':8}}))
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(tmp_path/'model.safetensors'))
    restored=load_transfer_checkpoint(tmp_path).eval()
    with torch.no_grad():
        for x,y in zip(forward(model,batch(tok)),forward(restored,batch(tok))):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_optimizer_unique_parameters_and_qf_task_scope(tok):
    model=install_mechanisms(baseline(),rank=8,projection=16,quadratic_rank=8)
    groups=optimizer_groups(model,{'encoder_lr':2.5e-5,'head_lr':1e-4,'module_lr':1e-3})
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    h=torch.randn(3,64);c=torch.randn(3,5,64);t=torch.tensor([0,1,2]);v=torch.ones(3,5,dtype=torch.bool)
    nn.init.normal_(model.transfer_readout.qf_out.weight)
    out=model.transfer_readout.quadratic(h,c,t,v)
    assert out[0].abs().sum()>0
    assert torch.count_nonzero(out[1:])==0
