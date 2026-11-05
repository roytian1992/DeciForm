from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from numeric_score_objectives import NumericScoreObjective, loss_components, data_loss
from score_specialist import interpolation


@pytest.mark.parametrize('values,y', [([0., .1, .2, .3], .173), ([0., 20., 50., 100.], 72.315)])
def test_fractional_crps_differs_from_true_continuous_crps_by_label_only_constant(values, y):
    z = torch.tensor([-.3, 1.2, -.6, .2], dtype=torch.float64, requires_grad=True)
    objective = NumericScoreObjective('crps', mean_weight=0.)
    parts = loss_components(z, values, y, objective)
    s = z.new_tensor(values); p = z.softmax(0); q = z.new_tensor(interpolation(values, y))
    energy = ((p*(s-y).abs()).sum()-.5*(p[:, None]*p[None, :]*(s[:, None]-s[None, :]).abs()).sum())/(s[-1]-s[0])
    target_cdf = q.cumsum(0)[:-1]
    constant = (s.diff()/(s[-1]-s[0])*target_cdf*(1-target_cdf)).sum()
    torch.testing.assert_close(parts['normalized_crps']+constant, energy)
    torch.testing.assert_close(torch.autograd.grad(parts['normalized_crps'], z)[0], torch.autograd.grad(energy, z)[0])


@pytest.mark.parametrize('kind', ['cdf_mean', 'crps'])
def test_raw_fractional_targets_change_loss_without_nearest_grid_rounding(kind):
    objective = NumericScoreObjective(kind, mean_weight=1. if kind=='cdf_mean' else 0.)
    z = torch.tensor([-2., 1., 0., -3.], requires_grad=True)
    a = data_loss(z, [0., 1., 2., 3.], 1.11, objective)
    b = data_loss(z, [0., 1., 2., 3.], 1.19, objective)
    assert a != b
    assert torch.isfinite(torch.autograd.grad(a+b, z)[0]).all()
