import copy
import pytest
import torch
from test_numeric_trie_consistency import fixture
from models_numeric_trie_pilot import PilotObjective
from models_numeric_trie_r1 import FormalModel, R1Objective, mean_regression, r1_components
from numeric_trie import components


def test_lambda_zero_matches_control_value_and_gradients():
    old,rows,enc=fixture('trie_token_ce_cdf')
    old.distribution_objective=PilotObjective(token_weight=0.)
    new=copy.deepcopy(old);new.__class__=FormalModel
    new.distribution_objective=R1Objective(mean_weight=0.)
    a=old.losses(rows,enc);b=new.losses(rows,enc)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward();b.sum().backward()
    for (n,p),(m,q) in zip(old.named_parameters(),new.named_parameters()):
        assert n==m and p.shape==q.shape
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    assert new.lm.calls==1


@pytest.mark.parametrize('values,label', [([0.,1.],.3),([i*.1 for i in range(51)],3.6666666666666665),(list(range(101)),73.2),([2.,3.,7.,12.],6.25)])
def test_fractional_targets_and_affine_range_invariance(values,label):
    z=torch.linspace(-1,1,len(values),dtype=torch.float64,requires_grad=True)
    got=mean_regression(z,values,label)
    mu=(z.softmax(0)*z.new_tensor(values)).sum()
    expected=((mu-label)/(values[-1]-values[0]))**2
    torch.testing.assert_close(got,expected)
    changed=mean_regression(z,[10*v+31 for v in values],10*label+31)
    torch.testing.assert_close(got,changed)
    assert torch.autograd.gradcheck(lambda a:mean_regression(a,values,label),(z,))


def test_mean_mse_does_not_penalize_variance_by_itself():
    # Both distributions have mean1; variance differs. Both regression terms0.
    for p in ([.49,.02,.49],[.01,.98,.01]):
        r=mean_regression(torch.tensor(p,dtype=torch.float64).log(),[0.,1.,2.],1.)
        assert r.item()<1e-20


def test_auxiliary_changes_gradients_but_adds_no_parameters():
    old,rows,enc=fixture('trie_token_ce_cdf');old.distribution_objective=PilotObjective(token_weight=0.)
    new=copy.deepcopy(old);new.__class__=FormalModel;new.distribution_objective=R1Objective()
    old.losses(rows,enc).sum().backward();new.losses(rows,enc).sum().backward()
    assert not torch.allclose(old.lm.embedding.weight.grad,new.lm.embedding.weight.grad)
    assert list(old.state_dict())==list(new.state_dict())
    assert new.lm.calls==1


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf')])
def test_invalid_weight(bad):
    with pytest.raises(ValueError):R1Objective(mean_weight=bad)


def test_extreme_logits_and_diagnostics():
    z=torch.tensor([-1e4,0.,1e4],requires_grad=True)
    total,*_=r1_components(z,[0.,50.,100.],.123,R1Objective())
    assert torch.isfinite(total)
    total.backward();assert torch.isfinite(z.grad).all()
    m,rows,enc=fixture('trie_token_ce_cdf');m.__class__=FormalModel;m.distribution_objective=R1Objective()
    m.collect_loss_diagnostics=True;m._loss_diagnostics=None;m.objective_component_sums=None
    out=m.losses(rows,enc);d=m.take_loss_diagnostics();c=m.take_objective_components()
    assert d[0,0].item()==2 and c[0].item()==2
    assert torch.isfinite(d).all() and c[3].item()>=0
    out.sum().backward()
