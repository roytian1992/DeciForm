"""Numerical distribution, native-action MAE and differentiable projection tests."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
import common
from numeric_score_distribution import projected_log_probabilities, NumericCDFReadout
from score_specialist import interpolation, data_loss, SpecialistObjective
from independent_candidates import shared_hidden
from test_independent_candidates import tiny, cpu_reference_kernels


def test_uniform_density_projection_preserves_native_absolute_risk():
    values = [0., .1, .4, .8, 1.]
    mass = torch.tensor([.001] + [.998/4]*4 + [.001], dtype=torch.float64)
    logp = projected_log_probabilities(torch.zeros(4, dtype=torch.float64), mass.log(), values)
    p = logp.exp()
    for a in values:
        risk = sum(p[i]*abs(a-s) for i, s in enumerate(values))
        expected = .998*(a*a-a+.5) + .001*a + .001*(1-a)
        torch.testing.assert_close(risk, risk.new_tensor(expected), atol=1e-12, rtol=1e-12)


def test_nested_grids_and_affine_ranges_are_consistent():
    w = torch.tensor([-.4, .8, .1], dtype=torch.float64)
    m = torch.tensor([-.8, .5, -.7, .9, -.3], dtype=torch.float64)
    fine, coarse = [0., .1, .2, .5, .8, 1.], [0., .2, .8, 1.]
    pf = projected_log_probabilities(w, m, fine).exp()
    pc = projected_log_probabilities(w, m, coarse).exp()
    transition = torch.tensor([interpolation(coarse, s) for s in fine], dtype=torch.float64)
    torch.testing.assert_close(pf @ transition, pc, rtol=1e-11, atol=1e-12)
    shifted = [s*100-7 for s in fine]
    torch.testing.assert_close(projected_log_probabilities(w, m, shifted).exp(), pf,
                               rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('k', [5, 51, 101])
def test_full_grid_probabilities_and_cdf_loss_gradients(k):
    torch.manual_seed(29)
    head = NumericCDFReadout(8, hidden_size=12, bins=8)
    x = torch.randn(2, 8, requires_grad=True)
    supports = [torch.linspace(0, 5, k).tolist(), torch.linspace(0, 100, k).tolist()]
    logits = head(x, supports)
    loss = 0
    for z, s, y in zip(logits, supports, [2.33, 72.3]):
        assert len(z) == k and torch.isfinite(z).all()
        torch.testing.assert_close(z.exp().sum(), z.new_tensor(1.))
        assert torch.all(z.exp() > 0)
        loss = loss + data_loss(z, s, y, SpecialistObjective('soft_cdf'))
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_projection_autograd_matches_numerical_derivative():
    w = torch.tensor([-.4, .8, .1], dtype=torch.float64, requires_grad=True)
    m = torch.tensor([-.8, .5, -.7, .9, -.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: projected_log_probabilities(a, b, [0., .2, .7, 1.]),
                                    (w, m), eps=1e-6, atol=2e-5, rtol=1e-4)


def test_query_only_branch_matches_full_candidate_query_and_backpropagates():
    lm, router = tiny()
    record = {'prefix': [3, 5, 8], 'query_token': 2,
              'branches': [[21, 22], [23, 24]], 'semantics': [0, 1]}
    full, _ = shared_hidden(lm, [record], 0, torch.tensor([2]), router)
    query, options = shared_hidden(lm, [{**record, 'branches': [], 'semantics': []}],
                                    0, torch.tensor([2]), router)
    assert options[0].shape[0] == 0
    torch.testing.assert_close(query, full, rtol=2e-4, atol=2e-5)
    (query*torch.linspace(-1, 1, query.numel()).reshape_as(query)).sum().backward()
    assert router.matrix_raw.grad.abs().sum() > 0
