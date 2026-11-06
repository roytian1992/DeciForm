import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from score_rlcd import RLCDObjective, RLCDLossMixin, rlcd_rows, QTYPES
from score_specialist import interpolation
from laya.train import rlcd_loss


@pytest.mark.parametrize('lengths',[(4,), (4,11,61), (51,101)])
def test_upstream_value_gradient_and_rng_parity(lengths):
    z=torch.linspace(-2,2,len(lengths)*max(lengths)).reshape(len(lengths),-1).requires_grad_()
    mask=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
    target=torch.zeros_like(z)
    for j,k in enumerate(lengths):
        target[j,:k]=torch.tensor(interpolation(list(range(k)),(k-1)*.37))
    task=torch.full((len(lengths),),QTYPES['score'])
    torch.manual_seed(86)
    reference=rlcd_loss(z,target,mask,task,.4,4,.75,1.)
    rg=torch.autograd.grad(reference,z)[0];rng=torch.get_rng_state()
    torch.manual_seed(86)
    actual=rlcd_rows(z,target,mask,RLCDObjective()).mean()
    ag=torch.autograd.grad(actual,z)[0]
    torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(ag,rg,rtol=1e-6,atol=1e-6)
    assert torch.equal(torch.get_rng_state(),rng)
    assert not torch.count_nonzero(ag[~mask])
    assert torch.isfinite(ag).all() and ag.abs().sum()>0


def test_rlcd_gradient_includes_reward_term():
    z=torch.tensor([[.3,-.2,1.]],requires_grad=True)
    mask=torch.ones_like(z,dtype=torch.bool);target=torch.tensor([[0.,1.,0.]])
    torch.manual_seed(72)
    loss=rlcd_rows(z,target,mask,RLCDObjective()).mean()
    g=torch.autograd.grad(loss,z)[0]
    ce=torch.autograd.grad(-(target*z.log_softmax(-1)).sum(),z)[0]
    assert not torch.allclose(g,ce)


class Dummy(RLCDLossMixin):
    def __init__(self,diagnostics):
        self.calls=0;self.device='cpu';self.collect_loss_diagnostics=diagnostics
        self._loss_diagnostics=None;self.distribution_objective=RLCDObjective()
        self.zs=[torch.linspace(-1,1,k,requires_grad=True) for k in (4,11,51)]
        self.net=type('Net',(),{'regularization':lambda s:torch.tensor(.03)})()
    def scores(self,rows,encodings):
        self.calls+=1
        return self.zs


def test_mixin_keeps_single_forward_fractional_targets_and_gradient():
    rows=[{'task':'score','score_support':list(range(k)),'label':(k-1)*.37} for k in (4,11,51)]
    plain,diagnostic=Dummy(False),Dummy(True)
    torch.manual_seed(28);a=plain.losses(rows,[None]*3);state=torch.get_rng_state()
    torch.manual_seed(28);b=diagnostic.losses(rows,[None]*3)
    torch.testing.assert_close(a,b)
    assert torch.equal(state,torch.get_rng_state())
    a.sum().backward();b.sum().backward()
    for x,y in zip(plain.zs,diagnostic.zs):torch.testing.assert_close(x.grad,y.grad)
    stats=diagnostic.take_loss_diagnostics()
    assert stats.shape==(1,7) and stats[0,0]==3 and torch.isfinite(stats).all()
    assert plain.calls==diagnostic.calls==1
    assert diagnostic.take_loss_diagnostics()[0,0]==0


def test_invalid_support_never_silently_truncates_or_rounds():
    with pytest.raises(ValueError):interpolation([0.,1.],1.01)
    q=interpolation([0.,.1,.2],.175)
    assert q==pytest.approx([0.,.25,.75])
    with pytest.raises(ValueError):RLCDObjective(samples=1)
