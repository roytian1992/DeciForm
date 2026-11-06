import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from classification_rlcd import rlcd_rows, classification_losses, FormalModel, QTYPES
from score_distribution import DistributionObjective, task_data_loss
from laya.train import rlcd_loss
from prepare_classification_rlcd import stratified_indices

OBJ=DistributionObjective(kind='ordinal_main',ordinal_scale=3.0383850790577127)


@pytest.mark.parametrize('lengths,tasks', [([2],['noul']),([2,3,4],['choice']*3),([2,4,77],['noul','choice','choice'])])
def test_native_value_gradient_and_padding(lengths,tasks):
    z=torch.linspace(-1,1,len(lengths)*max(lengths)).reshape(len(lengths),-1).requires_grad_()
    mask=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
    targets=torch.zeros_like(z)
    for i,k in enumerate(lengths):targets[i,k//2]=1
    qtype=torch.tensor([QTYPES[t] for t in tasks])
    torch.manual_seed(42);ref=rlcd_loss(z,targets,mask,qtype,.4,4,.75,1.)
    rg=torch.autograd.grad(ref,z)[0];rng=torch.get_rng_state()
    torch.manual_seed(42);actual=rlcd_rows(z,targets,mask,qtype).mean()
    ag=torch.autograd.grad(actual,z)[0]
    torch.testing.assert_close(actual,ref,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(ag,rg,rtol=1e-6,atol=1e-6)
    assert torch.equal(rng,torch.get_rng_state())
    assert not torch.count_nonzero(ag[~mask])


def test_ce_is_exact_baseline_and_rng_preserved():
    zs=[torch.tensor([.3,-.2,1.],requires_grad=True),torch.tensor([.4,-.1],requires_grad=True)]
    ys=[1,0];tasks=['choice','noul']
    torch.manual_seed(34);rng=torch.get_rng_state()
    baseline=classification_losses(zs,ys,tasks,[],OBJ)
    expected=torch.stack([task_data_loss(z,y,t,OBJ) for z,y,t in zip(zs,ys,tasks)])
    assert torch.equal(baseline,expected) and torch.equal(rng,torch.get_rng_state())
    changed=classification_losses(zs,ys,tasks,['choice'],OBJ)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(changed[1],baseline[1])
    bg=torch.autograd.grad(baseline.sum(),zs,retain_graph=True)
    cg=torch.autograd.grad(changed.sum(),zs)
    assert not torch.allclose(bg[0],cg[0])
    assert torch.equal(bg[1],cg[1])


class Dummy(FormalModel):
    def __init__(self,diagnostics):
        torch.nn.Module.__init__(self)
        self.device='cpu';self.collect_loss_diagnostics=diagnostics;self._loss_diagnostics=None
        self.rlcd_tasks=('noul',);self.distribution_objective=OBJ;self.calls=0
        self.zs=[torch.tensor([.3,-.2,1.],requires_grad=True),torch.tensor([.4,-.1],requires_grad=True)]
        self.net=type('Net',(),{'regularization':lambda s:torch.tensor(.03)})()
    def scores(self,*args):
        self.calls+=1
        return self.zs


def test_diagnostics_do_not_change_loss_gradient_or_rng():
    rs=[{'task':'choice','question':{'options':['a','b','c']},'label':'b'},
        {'task':'noul','question':{},'label':False}]
    plain,diag=Dummy(False),Dummy(True)
    torch.manual_seed(11);a=plain.losses(rs,[None,None]);rng=torch.get_rng_state()
    torch.manual_seed(11);b=diag.losses(rs,[None,None])
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    a.sum().backward();b.sum().backward()
    for x,y in zip(plain.zs,diag.zs):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    stats=diag.take_loss_diagnostics()
    assert stats.shape==(2,7) and stats[:,0].sum()==2 and torch.isfinite(stats).all()
    assert diag.take_loss_diagnostics().sum()==0 and plain.calls==diag.calls==1


def test_score_rejected_and_subsets_reproducible():
    with pytest.raises(ValueError):classification_losses([torch.ones(2)],[0],['score'],[],OBJ)
    rows=[{'task':'choice','source':str(i%2),'label':str(i%3),'id':str(i)} for i in range(90)]
    a=stratified_indices(rows,'choice',30,'train');b=stratified_indices(rows,'choice',30,'train')
    assert a==b and len(set(a))==30
    from collections import Counter
    assert set(Counter((rows[i]['source'],rows[i]['label']) for i in a).values())=={5}
