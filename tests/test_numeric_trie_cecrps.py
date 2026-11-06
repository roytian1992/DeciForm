import copy
import pytest
import torch
from test_numeric_trie_consistency import fixture
from models_numeric_trie_pilot import PilotObjective
from models_numeric_trie_cecrps import CECRPSObjective, FormalModel, normalized_crps, loss_components
from score_specialist import interpolation, data_loss, SpecialistObjective

@pytest.mark.parametrize('values,label', [([0.,1.],.3),([i/10 for i in range(51)],3.6666666666666665),(list(range(101)),73.2),([2.,3.,7.,12.],6.25),([0.,1.,2.],1.)])
def test_crps_energy_identity_fractional_constant_and_gradient(values,label):
    z=torch.linspace(-1,1,len(values),dtype=torch.float64,requires_grad=True)
    t=z.new_tensor(values);span=t[-1]-t[0];t=(t-t[0])/span;u=(label-values[0])/float(span)
    p=z.softmax(0);q=z.new_tensor(interpolation(values,label));g=q.cumsum(0)[:-1]
    energy=(p*(t-u).abs()).sum()-.5*(p[:,None]*p[None,:]*(t[:,None]-t[None,:]).abs()).sum()
    constant=(t.diff()*g*(1-g)).sum();got=normalized_crps(z,values,label)
    torch.testing.assert_close(got+constant,energy)
    torch.testing.assert_close(torch.autograd.grad(got,z,retain_graph=True)[0],torch.autograd.grad(energy,z)[0])
    torch.testing.assert_close(got,normalized_crps(z,[10*v+31 for v in values],10*label+31))
    assert torch.autograd.gradcheck(lambda a:normalized_crps(a,values,label),(z,))


def test_expected_crps_minimized_at_true_distribution():
    values=[0.,1.,3.];q=torch.tensor([.2,.5,.3],dtype=torch.float64)
    def expected(p):return sum(q[j]*normalized_crps(p.log(),values,y) for j,y in enumerate(values))
    optimum=expected(q)
    for p in [[.3,.4,.3],[.05,.9,.05],[.8,.1,.1]]:assert expected(q.new_tensor(p))>optimum


@pytest.mark.parametrize('w',[0.,.5,1.])
def test_mixture_and_extreme_gradients(w):
    z=torch.tensor([-1e4,0.,1e4],requires_grad=True);obj=CECRPSObjective(crps_weight=w)
    total,ce,crps,scaled=loss_components(z,[0.,50.,100.],.123,obj)
    torch.testing.assert_close(total,(1-w)*ce+w*2*obj.ordinal_scale*crps)
    total.backward();assert torch.isfinite(total) and torch.isfinite(z.grad).all()


def test_predeclared_threshold_gradient_scale():
    alpha=CECRPSObjective().ordinal_scale
    for target in [0.,.3,1.]:
        f=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
        a=2*alpha*(f-target)**2;b=alpha*(-target*f.log()-(1-target)*(1-f).log())
        torch.testing.assert_close(torch.autograd.grad(a,f)[0],torch.autograd.grad(b,f)[0])


def test_same_initial_architecture_and_no_extra_forward():
    old,rows,enc=fixture('trie_token_ce_cdf');old.distribution_objective=PilotObjective(token_weight=0.)
    new=copy.deepcopy(old);new.__class__=FormalModel;new.distribution_objective=CECRPSObjective()
    assert old.state_dict().keys()==new.state_dict().keys()
    for k in old.state_dict():torch.testing.assert_close(old.state_dict()[k],new.state_dict()[k],rtol=0,atol=0)
    new.collect_loss_diagnostics=True;new._loss_diagnostics=None;new.objective_component_sums=None
    out=new.losses(rows,enc);out.sum().backward()
    assert new.lm.calls==1 and torch.isfinite(out).all()
    assert new.take_objective_components()[0].item()==2
    assert new.take_loss_diagnostics()[0,0].item()==2
    for n,p in new.named_parameters():
        if p.grad is not None:assert torch.isfinite(p.grad).all(),n


def test_zero_crps_matches_ce_gradient():
    z=torch.randn(5,requires_grad=True);values=list(range(5));label=2.7
    total,*_=loss_components(z,values,label,CECRPSObjective(crps_weight=0.))
    ce=data_loss(z,values,label,SpecialistObjective('soft_ce'))
    torch.testing.assert_close(total,ce,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(total,z,retain_graph=True)[0],torch.autograd.grad(ce,z)[0],rtol=0,atol=0)


@pytest.mark.parametrize('bad',[-.1,1.1,float('nan'),float('inf')])
def test_reject_invalid_mix(bad):
    with pytest.raises(ValueError):CECRPSObjective(crps_weight=bad)
