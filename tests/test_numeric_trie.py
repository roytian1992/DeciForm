import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from numeric_trie import ALPHABET, NumericTrieReadout, TrieObjective, components, numeric_tree
from score_distribution import decisions
from score_specialist import interpolation


def small():
    torch.manual_seed(71)
    return NumericTrieReadout(torch.randn(12,16), torch.randn(12,16), 8)


@pytest.mark.parametrize('values', [list(range(0,4)), list(range(2,13)), list(range(0,61)),
                                   list(range(101)), [i/10 for i in range(51)]])
def test_complete_normalized_support_and_gradients(values):
    m=small();h=torch.randn(2,16,requires_grad=True);q=torch.randn(2,8,requires_grad=True)
    # Nonzero readout exercises the decoder's recurrent gradients as well.
    torch.nn.init.normal_(m.readout.weight,std=.1)
    z=m(h,q,[values,values])
    assert all(len(p)==len(values) and torch.isfinite(p).all() for p in z)
    for p in z: torch.testing.assert_close(p.exp().sum(),p.new_ones(()),atol=2e-6,rtol=2e-6)
    loss=sum(components(p,values,(values[0]+values[-1])*.5,TrieObjective('trie_ce_cdf'))[0] for p in z)
    loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_single_digit_initialization_exactly_preserves_restricted_lm_prior():
    m=small();h=torch.randn(2,16);q=torch.randn(2,8)
    actual=torch.stack(m(h,q,[list(range(2,7))]*2))
    expected=(h@m.output_rows[2:7].T).log_softmax(-1)
    torch.testing.assert_close(actual,expected)


def test_shared_prefixes_normalize_without_length_bias_or_missing_values():
    m=small();m.output_rows.zero_()
    values=[1,10,11,100]
    z=m(torch.zeros(1,16),torch.zeros(1,8),[values])[0]
    # Root '1' deterministic; at prefix '1': 0 / 1 / end equally likely;
    # after '10': 0 / end equally likely. Short values keep a proper leaf mass.
    torch.testing.assert_close(z.exp(),torch.tensor([1/3,1/6,1/3,1/6]))
    strings,branches,needed,paths=numeric_tree(tuple(values))
    assert strings==('1}','10}','11}','100}')


def test_mixed_support_batch_matches_individual_evaluation():
    m=small();h=torch.randn(3,16);q=torch.randn(3,8)
    supports=[[0,1,2],[i/10 for i in range(51)],[0,1,2]]
    actual=m(h,q,supports)
    for i,z in enumerate(actual):
        torch.testing.assert_close(z,m(h[i:i+1],q[i:i+1],[supports[i]])[0])


def test_proper_losses_minimized_at_fractional_target_and_mae_median():
    values=[0.,.1,.2,.3,.4];label=.25
    q=torch.tensor(interpolation(values,label),dtype=torch.float64)
    optimum=(q+1e-12).log().requires_grad_()
    for obj in [TrieObjective('trie_ce',cdf_weight=0.),TrieObjective('trie_ce_cdf')]:
        best=components(optimum,values,label,obj)[0]
        worse=components(torch.zeros_like(optimum),values,label,obj)[0]
        assert best < worse
        grad=torch.autograd.grad(best,optimum,retain_graph=True)[0]
        assert grad.abs().max()<1e-9
    p=[.1,.2,.4,.2,.1]
    d=decisions(values,p)
    risks=[sum(abs(x-y)*v for y,v in zip(values,p)) for x in values]
    assert risks[d['indices']['median']]==min(risks)


@pytest.mark.parametrize('values', [[1,1],[1,0],[-1,0],[0,float('nan')],[0,1e-30]])
def test_invalid_support_rejected(values):
    with pytest.raises(ValueError):numeric_tree(tuple(values))
