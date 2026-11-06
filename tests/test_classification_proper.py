import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from classification_proper import ClassificationObjective, proper_loss, objective_losses, FormalModel
from classification_rlcd import classification_losses, TASKS
from score_distribution import DistributionObjective, task_data_loss
OBJ=DistributionObjective(kind='ordinal_main',ordinal_scale=3.0383850790577127)

@pytest.mark.parametrize('k',[2,3,4,21,77])
def test_uniform_ce_scale_and_analytic_gradient(k):
    z=torch.zeros(k,dtype=torch.float64,requires_grad=True);y=k//2;a=.75
    g=torch.autograd.grad(proper_loss(z,y,a),z)[0]
    target=F.one_hot(torch.tensor(y),k).double()
    torch.testing.assert_close(g,z.softmax(0)-target,rtol=1e-12,atol=1e-12)
    z=torch.linspace(-3,2,k,dtype=torch.float64,requires_grad=True)
    p=z.softmax(0);v=(2*z).softmax(0)
    expected=((p-target)+a*k**.5*p[y]/p.norm()*(v-target))/(1+a)
    g=torch.autograd.grad(proper_loss(z,y,a),z)[0]
    torch.testing.assert_close(g,expected,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(lambda x:proper_loss(x,y,a),(z,),fast_mode=True)

@pytest.mark.parametrize('k',[2,4,77])
def test_proper_regret_and_permutation(k):
    g=torch.Generator().manual_seed(41)
    z=torch.randn(k,dtype=torch.float64,generator=g,requires_grad=True)
    q=torch.randn(k,dtype=torch.float64,generator=g).softmax(0);p=z.softmax(0);a=.75
    risk=(q*torch.stack([proper_loss(z,i,a) for i in range(k)])).sum()
    optimal=(q*torch.stack([proper_loss(q.log(),i,a) for i in range(k)])).sum()
    regret=((q*(q.log()-p.log())).sum()+a*k**.5*(q.norm()-q@p/p.norm()))/(1+a)
    torch.testing.assert_close(risk-optimal,regret,rtol=1e-12,atol=1e-12)
    assert regret>=0
    perm=torch.randperm(k,generator=g);y=k//2;newy=(perm==y).nonzero().item()
    original=proper_loss(z,y,a);changed=proper_loss(z[perm],newy,a)
    torch.testing.assert_close(original,changed,rtol=1e-12,atol=1e-12)
    go=torch.autograd.grad(original,z,retain_graph=True)[0];gc=torch.autograd.grad(changed,z)[0]
    torch.testing.assert_close(go,gc,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
def test_extreme_logits_finite(dtype):
    z=torch.tensor([-10000.,10000.,0.],dtype=dtype,requires_grad=True)
    for y in range(3):
        loss=proper_loss(z,y);grad=torch.autograd.grad(loss,z)[0]
        assert torch.isfinite(loss) and torch.isfinite(grad).all()

@pytest.mark.parametrize('beta',[0.,.1,1.])
def test_scaled_increment_preserves_anchor_and_rng(beta):
    zs=[torch.tensor([.3,-.2,1.],requires_grad=True),torch.tensor([.4,-.1],requires_grad=True)]
    ys=[1,0];tasks=['choice','noul']
    torch.manual_seed(44);rng=torch.get_rng_state()
    ce=torch.stack([task_data_loss(z,y,t,OBJ) for z,y,t in zip(zs,ys,tasks)])
    native=classification_losses(zs,ys,tasks,TASKS,OBJ)
    actual=objective_losses(zs,ys,tasks,OBJ,ClassificationObjective(kind='scaled_rlcd',increment_weight=beta))
    expected=ce+beta*(native-ce)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    ga=torch.autograd.grad(actual.sum(),zs,retain_graph=True);ge=torch.autograd.grad(expected.sum(),zs)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

class Dummy(FormalModel):
    def __init__(self,diagnostics):
        torch.nn.Module.__init__(self)
        self.device='cpu';self.collect_loss_diagnostics=diagnostics;self._loss_diagnostics=None
        self.rlcd_tasks=TASKS;self.distribution_objective=OBJ;self.calls=0
        self.classification_objective=ClassificationObjective()
        self.zs=[torch.tensor([.3,-.2,1.],requires_grad=True),torch.tensor([.4,-.1],requires_grad=True)]
        self.net=type('Net',(),{'regularization':lambda s:torch.tensor(.03)})()
    def scores(self,*args):self.calls+=1;return self.zs

def test_one_forward_diagnostics_preserve_gradients():
    rows=[{'task':'choice','question':{'options':['a','b','c']},'label':'b'},
          {'task':'noul','question':{},'label':False}]
    plain,diag=Dummy(False),Dummy(True);rng=torch.get_rng_state()
    a=plain.losses(rows,[None,None]);b=diag.losses(rows,[None,None])
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    a.sum().backward();b.sum().backward()
    for x,y in zip(plain.zs,diag.zs):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    stats=diag.take_loss_diagnostics()
    assert stats[:,0].sum()==2 and torch.isfinite(stats).all()
    assert plain.calls==diag.calls==1

def test_invalid_inputs():
    for kwargs in [{'alpha':-1},{'alpha':float('nan')},{'increment_weight':1.1},{'kind':'focal'}]:
        with pytest.raises(ValueError):ClassificationObjective(**kwargs)
    with pytest.raises(ValueError):objective_losses([torch.zeros(2)],[0],['score'],OBJ,ClassificationObjective())
