import copy
import torch
import pytest
from test_numeric_trie_consistency import fixture
from models_numeric_trie_pilot import PilotObjective
from numeric_trie import components


def test_zero_weight_is_exact_ce_cdf():
    m, rows, encs = fixture('trie_token_ce_cdf')
    m.distribution_objective = PilotObjective(token_weight=0.)
    other = copy.deepcopy(m)
    actual = m.losses(rows, encs)
    z, _ = other.joint_scores_and_token_loss(rows, encs)
    expected = torch.stack([components(a,r['score_support'],r['label'],other.distribution_objective.distribution())[0] for a,r in zip(z,rows)])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.sum().backward(); expected.sum().backward()
    for (a,p),(b,q) in zip(m.named_parameters(),other.named_parameters()):
        assert a==b
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


def test_auxiliary_has_distinct_backbone_gradients():
    a, rows, encs = fixture('trie_token_ce_cdf'); b = copy.deepcopy(a)
    a.distribution_objective = PilotObjective(token_weight=0.)
    b.distribution_objective = PilotObjective(token_weight=.5)
    za,ta=a.joint_scores_and_token_loss(rows,encs);a.clear()
    zb,tb=b.joint_scores_and_token_loss(rows,encs);b.clear()
    for x,y in zip(za,zb):torch.testing.assert_close(x,y,rtol=0,atol=0)
    la=a.losses(rows,encs);lb=b.losses(rows,encs)
    torch.testing.assert_close(lb-la,.5*tb)
    la.sum().backward();lb.sum().backward()
    assert not torch.allclose(a.lm.embedding.weight.grad,b.lm.embedding.weight.grad)
    assert torch.isfinite(b.lm.embedding.weight.grad).all()


@pytest.mark.parametrize('weight',[-1.,.1,1.,float('nan')])
def test_frozen_weights(weight):
    with pytest.raises(ValueError):PilotObjective(token_weight=weight)
