"""Objective invariants and actual mixed-task training-path isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from choice_losses import ChoiceObjective, choice_loss, logit_diagnostics


@pytest.mark.parametrize('kind', ['ce_pair', 'focal'])
@pytest.mark.parametrize('k', [2, 3, 4, 21, 77])
def test_uniform_output_gradients_match_ce(kind, k):
    z = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    actual = choice_loss(z, k-1, ChoiceObjective(kind))
    expected = F.cross_entropy(z[None], torch.tensor([k-1]))
    ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('weight', [0., .5, 1.])
def test_binary_pair_is_exactly_ce(weight):
    for value in [-100., -2., 0., 2., 100.]:
        z = torch.tensor([value, 0.], requires_grad=True)
        torch.testing.assert_close(choice_loss(z, 0, ChoiceObjective('ce_pair', pair_weight=weight)),
                                   F.cross_entropy(z[None], torch.tensor([0])))


@pytest.mark.parametrize('kind', ['ce_pair', 'focal'])
def test_permutation_and_common_logit_shift_preserve_loss(kind):
    z = torch.tensor([.2, -1.3, 2.1, .8], dtype=torch.float64)
    perm = torch.tensor([3, 1, 0, 2])
    objective = ChoiceObjective(kind)
    loss = choice_loss(z, 2, objective)
    torch.testing.assert_close(choice_loss(z[perm], 3, objective), loss)
    torch.testing.assert_close(choice_loss(z+80, 2, objective), loss)


@pytest.mark.parametrize('objective', [ChoiceObjective('ce_pair'), ChoiceObjective('focal'),
                                       ChoiceObjective('focal', focal_gamma=2)])
@pytest.mark.parametrize('label', [0, 1, 2])
def test_extreme_logits_keep_finite_corrective_gradients(objective, label):
    z = torch.tensor([-10000., 10000., 0.], requires_grad=True)
    loss = choice_loss(z, label, objective)
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    assert grad[label] <= 0
    if label != 1:
        assert grad[label] < 0 and grad[1] > 0


@pytest.mark.parametrize('kind', ['ce', 'ce_pair', 'focal'])
def test_diagnostic_matches_autograd_without_retaining_graph(kind):
    z = torch.tensor([.3, -1.7, 2.4, .1], dtype=torch.float64, requires_grad=True)
    objective = ChoiceObjective(kind)
    loss = choice_loss(z, 0, objective)
    grad = torch.autograd.grad(loss, z)[0]
    d = logit_diagnostics(z, 0, objective, choice=True)
    assert not d.requires_grad
    torch.testing.assert_close(d[1], loss.detach())
    torch.testing.assert_close(d[3], grad.norm(), atol=1e-12, rtol=1e-12)


def test_pair_expected_risk_is_stationary_at_true_distribution():
    eta = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    z = eta.log().requires_grad_()
    expected = sum(eta[y]*choice_loss(z, y, ChoiceObjective('ce_pair')) for y in range(4))
    torch.testing.assert_close(torch.autograd.grad(expected, z)[0], torch.zeros_like(z),
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize('kind', ['ce_pair', 'focal'])
def test_actual_loss_wrapper_changes_only_choice_and_preserves_microbatch_weights(kind):
    from models_choice_loss import FormalModel
    from models_v6_ablation import FormalModel as A4
    rows = [
        {'task': 'choice', 'question': {'options': ['a', 'b', 'c', 'd']}, 'label': 'b'},
        {'task': 'noul', 'question': {}, 'label': True},
        {'task': 'score', 'question': {'scale': {'min': 0, 'max': 4}}, 'label': 3},
        {'task': 'choice', 'question': {'options': ['a', 'b', 'c']}, 'label': 'c'},
    ]
    z = [torch.tensor(v, requires_grad=True) for v in
         ([.2, -.7, 1.4, .6], [-.2, .8], [0., 1., -.3, .7, -.4], [.5, -.8, 1.1])]
    penalty = torch.tensor(.03, requires_grad=True)
    model = SimpleNamespace(scores=lambda rs, es: es,
        net=SimpleNamespace(regularization=lambda: penalty.square(), auxiliary_losses_enabled=False),
        choice_objective=ChoiceObjective(kind), collect_loss_diagnostics=False)
    current = FormalModel.losses(model, rows, z)
    reference = A4.losses(model, rows, z)
    torch.testing.assert_close(current[1:3], reference[1:3], atol=0, rtol=0)
    assert not torch.allclose(current[[0, 3]], reference[[0, 3]])
    split = sum(FormalModel.losses(model, rows[a:b], z[a:b]).sum()/len(rows)
                for a, b in [(0, 1), (1, 4)])
    full = current.mean()
    gs = torch.autograd.grad(split, z+[penalty], retain_graph=True)
    gf = torch.autograd.grad(full, z+[penalty])
    for actual, expected in zip(gs, gf):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('bad', [{'kind': 'unknown'}, {'kind': 'ce_pair', 'pair_weight': 2},
                                {'kind': 'focal', 'focal_gamma': .5},
                                {'kind': 'focal', 'focal_gamma': float('nan')}])
def test_reject_unsupported_objectives(bad):
    with pytest.raises(ValueError):
        ChoiceObjective(**bad)


def test_worker_uses_loss_experiment_model():
    from models_choice_loss import FormalModel
    import worker_choice_loss
    assert worker_choice_loss.main.__globals__['FormalModel'] is FormalModel
