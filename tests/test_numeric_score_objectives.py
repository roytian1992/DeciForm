from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from numeric_score_objectives import NumericScoreObjective, data_loss, loss_components, logit_diagnostics


def objective(kind):
    return NumericScoreObjective(kind, mean_weight=1. if kind == 'cdf_mean' else 0.)


def test_regression_is_squared_mean_error_not_variance_penalty():
    p = torch.tensor([.4, .2, .4], dtype=torch.float64)
    parts = loss_components(p.log(), [0., 1., 2.], 1., objective('cdf_mean'))
    assert parts['normalized_mean_mse'].item() == pytest.approx(0., abs=1e-15)
    torch.testing.assert_close(parts['total'], parts['cdf_log'])
    assert (p * torch.tensor([1., 0., 1.])).sum() > 0


@pytest.mark.parametrize('kind', ['cdf_mean', 'crps'])
def test_affine_score_ranges_preserve_loss_and_gradient(kind):
    z = torch.tensor([-.3, 1.1, -.7, .4], dtype=torch.float64, requires_grad=True)
    values = [0., .2, .7, 1.]
    a = data_loss(z, values, .7, objective(kind))
    b = data_loss(z, [v*60+2 for v in values], 44., objective(kind))
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(a, z)[0], torch.autograd.grad(b, z)[0])


def test_crps_matches_energy_form_and_reduces_to_absolute_error_for_point_mass():
    s = torch.tensor([2., 4., 7., 12.], dtype=torch.float64)
    p = torch.tensor([.1, .2, .4, .3], dtype=torch.float64)
    parts = loss_components(p.log(), s.tolist(), 7., objective('crps'))
    energy = (p*(s-7).abs()).sum() - .5*(p[:, None]*p[None, :]*(s[:, None]-s[None, :]).abs()).sum()
    torch.testing.assert_close(parts['normalized_crps'], energy/(s[-1]-s[0]))
    z = torch.tensor([-1000., 0., -1000., -1000.], dtype=torch.float64)
    parts = loss_components(z, s.tolist(), 7., objective('crps'))
    assert parts['normalized_crps'].item() == pytest.approx(3/10)


@pytest.mark.parametrize('kind', ['cdf_mean', 'crps'])
def test_population_optimum_is_true_distribution(kind):
    q = torch.tensor([.15, .5, .35], dtype=torch.float64)
    z = q.log().requires_grad_(True)
    values = [0., .3, 1.]
    expected = sum(q[i]*data_loss(z, values, y, objective(kind)) for i, y in enumerate(values))
    torch.testing.assert_close(torch.autograd.grad(expected, z)[0], torch.zeros_like(z), atol=1e-12, rtol=0)
    perturbed = (q + torch.tensor([.1, -.1, 0.])).log()
    other = sum(q[i]*data_loss(perturbed, values, y, objective(kind)) for i, y in enumerate(values))
    assert other > expected


def test_crps_scale_matches_cdf_gradient_at_binary_half_probability():
    z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    cdf = loss_components(z, [0., 1.], 0., objective('cdf_mean'))['cdf_log']
    crps = data_loss(z, [0., 1.], 0., objective('crps'))
    torch.testing.assert_close(torch.autograd.grad(cdf, z)[0], torch.autograd.grad(crps, z)[0])


@pytest.mark.parametrize('kind', ['cdf_mean', 'crps'])
@pytest.mark.parametrize('k', [4, 6, 11, 31, 61])
def test_native_grids_have_finite_gradients_and_diagnostics(kind, k):
    z = torch.linspace(-10, 10, k, dtype=torch.float64, requires_grad=True)
    values = list(range(k))
    loss = data_loss(z, values, k//2, objective(kind))
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.abs().sum() > 0
    d = logit_diagnostics(z, values, k//2, objective(kind))
    assert d.shape == (7,) and d[0] == 1 and torch.isfinite(d).all()


def test_reject_invalid_support_or_objective():
    with pytest.raises(ValueError):
        NumericScoreObjective('crps', mean_weight=1.)
    with pytest.raises(ValueError):
        data_loss(torch.zeros(3), [0., 1., 1.], 0., objective('cdf_mean'))
