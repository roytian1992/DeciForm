import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from score_proper import ScoreProperObjective, ScoreProperLossMixin, proper_score_loss
from score_specialist import interpolation
from classification_proper import proper_loss

@pytest.mark.parametrize('k',[2,5,51,61,101])
def test_onehot_matches_classification_and_soft_uniform_gradient(k):
    y=k//2;q=torch.zeros(k,dtype=torch.float64);q[y]=1
    z=torch.linspace(-3,2,k,dtype=torch.float64,requires_grad=True)
    a=proper_score_loss(z,q);b=proper_loss(z,y)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
    q[y]=.3;q[y-1]=.7;zero=torch.zeros(k,dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(proper_score_loss(zero,q),zero)[0]
    torch.testing.assert_close(gradient,zero.softmax(0)-q,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('k',[5,51,101])
def test_soft_gradient_risk_difference_and_additivity(k):
    z=torch.linspace(-2,1,k,dtype=torch.float64,requires_grad=True)
    q=torch.linspace(1,2,k,dtype=torch.float64);q/=q.sum();p=z.softmax(0);v=(2*z).softmax(0);a=.75
    loss=proper_score_loss(z,q,a)
    expected=((p-q)+a*k**.5/p.norm()*((q@p)*v-q*p))/(1+a)
    torch.testing.assert_close(torch.autograd.grad(loss,z,retain_graph=True)[0],expected,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(lambda x:proper_score_loss(x,q,a),(z,),fast_mode=True)
    regret=((q*(q.log()-p.log())).sum()+a*k**.5*(q.norm()-q@p/p.norm()))/(1+a)
    torch.testing.assert_close(loss-proper_score_loss(q.log(),q,a),regret,rtol=1e-12,atol=1e-12)
    assert regret>=0
    mixture=sum(q[i]*proper_loss(z,i,a) for i in range(k))
    torch.testing.assert_close(loss,mixture,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('values,label', [([0,1,2,3,4],2.35),([0.,.1,.2,.3],.17),([0,10,100],31.5),([2,3,4],2),([2,3,4],4)])
def test_fractional_targets_preserve_raw_value_and_support(values,label):
    q=torch.tensor(interpolation(values,label),dtype=torch.float64)
    torch.testing.assert_close(q.sum(),torch.tensor(1.,dtype=torch.float64),rtol=0,atol=1e-12)
    torch.testing.assert_close(q@torch.tensor(values,dtype=torch.float64),torch.tensor(float(label),dtype=torch.float64),rtol=0,atol=1e-12)
    assert (q>=0).all() and (q>0).sum()<=2 and len(q)==len(values)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
def test_extreme_logits_and_no_random_draws(dtype):
    z=torch.tensor([-10000.,10000.,0.],dtype=dtype,requires_grad=True);q=torch.tensor([.7,.3,0.])
    state=torch.get_rng_state();value=proper_score_loss(z,q);g=torch.autograd.grad(value,z)[0]
    assert torch.isfinite(value) and torch.isfinite(g).all() and torch.equal(state,torch.get_rng_state())
    z=z.float();torch.testing.assert_close(proper_score_loss(z,q,0),-(q*z.log_softmax(0)).sum())

class Dummy(ScoreProperLossMixin,torch.nn.Module):
    def __init__(self,diagnostics):
        super().__init__();self.device='cpu';self.collect_loss_diagnostics=diagnostics;self._loss_diagnostics=None
        self.distribution_objective=ScoreProperObjective();self.calls=0
        self.zs=[torch.tensor([-.2,.3,.8],requires_grad=True),torch.tensor([.4,-.1],requires_grad=True)]
        self.net=type('Net',(),{'regularization':lambda s:torch.tensor(.03)})()
    def scores(self,*args):self.calls+=1;return self.zs

def test_single_forward_and_nonperturbing_diagnostics():
    rows=[{'task':'score','score_support':[0.,.5,1.],'label':.23},{'task':'score','score_support':[1.,5.],'label':5.}]
    plain,diag=Dummy(False),Dummy(True);a=plain.losses(rows,[None,None]);b=diag.losses(rows,[None,None])
    assert torch.equal(a,b) and plain.calls==diag.calls==1
    a.sum().backward();b.sum().backward()
    for x,y in zip(plain.zs,diag.zs):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    stats=diag.take_loss_diagnostics();assert stats.shape==(1,7) and stats[0,0]==2 and torch.isfinite(stats).all()
    assert diag.take_loss_diagnostics().sum()==0

def test_invalid_objective_and_missing_candidates():
    with pytest.raises(ValueError):ScoreProperObjective(alpha=-1)
    with pytest.raises(ValueError):ScoreProperObjective(kind='rlcd')
    model=Dummy(False)
    with pytest.raises(ValueError):model.losses([{'task':'score','score_support':[0.,1.],'label':.23},{'task':'score','score_support':[1.,5.],'label':5.}],[None,None])

@pytest.mark.parametrize('values,y',[([0,1,2,3],1),([2,3,4,5,6,7,8,9,10,11,12],7),(list(range(61)),30)])
def test_cdf_reference_matches_original_d2_value_and_gradient(values,y):
    from score_distribution import DistributionObjective, task_data_loss
    from score_specialist import data_loss, SpecialistObjective
    z=torch.linspace(-2,1,len(values),dtype=torch.float64,requires_grad=True)
    original=task_data_loss(z,values.index(y),'score',DistributionObjective(kind='ordinal_main',ordinal_scale=3.0383850790577127),values)
    actual=data_loss(z,values,y,SpecialistObjective(kind='soft_cdf'))
    torch.testing.assert_close(actual,original,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(actual,z,retain_graph=True)[0];gb=torch.autograd.grad(original,z)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)

def test_small_selection_reproducible_and_keeps_sources():
    from prepare_score_proper_pilot import stratified
    rows=[{'id':str(i),'source_family':str(i%2),'label':(i%100)/10,'score_support':list(range(11))} for i in range(1000)]
    a=stratified(rows,128,'train','continuous');b=stratified(rows,128,'train','continuous')
    assert a==b and len(set(a))==128
    assert sum(rows[i]['source_family']=='0' for i in a)==64
    rows=[{'id':str(i),'source_family':'asap_'+str(i%8),'label':i%5,'score_support':list(range(6))} for i in range(1000)]
    a=stratified(rows,64,'dev','asap');assert len(set(a))==64 and len({rows[i]['source_family'] for i in a})==8
