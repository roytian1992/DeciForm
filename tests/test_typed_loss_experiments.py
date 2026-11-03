"""Task isolation, probability/gradient identities and native ordinal semantics."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from typed_losses import TypedObjective, ordinal_log_loss, task_data_loss, logit_diagnostics
from score_decoding import score_decoding_diagnostic


@pytest.mark.parametrize('y', [0, 1])
def test_noul_gradient_matches_derived_boundary_weight(y):
    objective = TypedObjective('noul_brier')
    for margin in [-20., -2., 0., 2., 20.]:
        z = torch.tensor([0., margin], dtype=torch.float64, requires_grad=True)
        loss = task_data_loss(z, y, 'noul', objective)
        grad = torch.autograd.grad(loss, z)[0]
        p = z[1].sigmoid()
        expected = ((1-objective.noul_rho)+4*objective.noul_rho*p*(1-p))*(p-y)
        torch.testing.assert_close(grad, torch.stack((-expected, expected)), atol=1e-12, rtol=1e-12)
        if margin == 0:
            ce = F.cross_entropy(z[None], torch.tensor([y]))
            torch.testing.assert_close(grad, torch.autograd.grad(ce, z)[0])


@pytest.mark.parametrize('eta', [.01, .2, .5, .8, .99])
def test_noul_expected_loss_is_stationary_at_true_probability(eta):
    z = torch.tensor([0., torch.logit(torch.tensor(eta, dtype=torch.float64)).item()],
                     dtype=torch.float64, requires_grad=True)
    objective = TypedObjective('noul_brier')
    risk = (1-eta)*task_data_loss(z, 0, 'noul', objective)+eta*task_data_loss(z, 1, 'noul', objective)
    torch.testing.assert_close(torch.autograd.grad(risk, z)[0], torch.zeros_like(z), atol=1e-12, rtol=0)


@pytest.mark.parametrize('y', [0, 1, 2, 3])
def test_ordinal_matches_independent_threshold_bces_and_affine_support(y):
    z = torch.tensor([.2, -.7, 1.1, .4], dtype=torch.float64, requires_grad=True)
    values = [0, 1, 20, 100]
    cdf = z.softmax(0).cumsum(0)[:-1]
    expected = sum((values[k+1]-values[k])/100*F.binary_cross_entropy(
        cdf[k], z.new_tensor(float(y <= k))) for k in range(3))
    actual = ordinal_log_loss(z, y, values)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, ordinal_log_loss(z, y, [v*7+19 for v in values]))
    assert not torch.allclose(actual, ordinal_log_loss(z, y, [0, 1, 2, 3]))


@pytest.mark.parametrize('y', [0, 1, 2, 3])
def test_ordinal_extreme_logits_remain_finite_and_correct_wrong_predictions(y):
    z = torch.tensor([-10000., 0., 10000., -5000.], requires_grad=True)
    loss = ordinal_log_loss(z, y, [0, 1, 20, 100])
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    if y != 2:
        assert loss > 100 and grad[2] > 0


def test_ordered_expected_loss_is_stationary_at_true_distribution():
    eta = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    z = eta.log().requires_grad_()
    risk = sum(eta[y]*ordinal_log_loss(z, y, [-2, 0, 2, 9]) for y in range(4))
    torch.testing.assert_close(torch.autograd.grad(risk, z)[0], torch.zeros_like(z), atol=1e-12, rtol=0)


@pytest.mark.parametrize('task', ['choice', 'noul', 'score'])
@pytest.mark.parametrize('kind', ['noul_brier', 'score_ordinal'])
def test_diagnostics_match_actual_autograd(task, kind):
    z = torch.tensor([.4, -.7] if task == 'noul' else [.4, -.7, 1.3, .2],
                     dtype=torch.float64, requires_grad=True)
    values = [0, 1, 20, 100] if task == 'score' else None
    objective = TypedObjective(kind)
    loss = task_data_loss(z, 1, task, objective, values)
    grad = torch.autograd.grad(loss, z)[0]
    d = logit_diagnostics(z, 1, task, objective, values)
    assert not d.requires_grad
    torch.testing.assert_close(d[1], loss.detach())
    torch.testing.assert_close(d[3], grad.norm(), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('kind,changed', [('noul_brier', 1), ('score_ordinal', 2)])
def test_actual_model_changes_only_named_task_and_preserves_row_weighting(kind, changed):
    from models_typed_loss import FormalModel
    from models_v6_ablation import FormalModel as A4
    rows = [
        {'task': 'choice', 'question': {'options': ['a', 'b', 'c', 'd']}, 'label': 'b'},
        {'task': 'noul', 'question': {}, 'label': True},
        {'task': 'score', 'question': {'scale': {'min': 0, 'max': 4}}, 'label': 3},
    ]
    z = [torch.tensor(v, requires_grad=True) for v in
         ([.2, -.7, 1.4, .6], [-.2, .8], [0., 1., -.3, .7, -.4])]
    penalty = torch.tensor(.03, requires_grad=True)
    model = SimpleNamespace(scores=lambda rs, es: es,
        net=SimpleNamespace(regularization=lambda: penalty.square(), auxiliary_losses_enabled=False),
        typed_objective=TypedObjective(kind), collect_loss_diagnostics=False)
    current = FormalModel.losses(model, rows, z)
    reference = A4.losses(model, rows, z)
    for i in range(3):
        if i == changed:
            assert not torch.allclose(current[i], reference[i])
        else:
            torch.testing.assert_close(current[i], reference[i], atol=0, rtol=0)
    split = sum(FormalModel.losses(model, rows[a:b], z[a:b]).sum()/3 for a,b in [(0,1),(1,3)])
    gs = torch.autograd.grad(split, z+[penalty], retain_graph=True)
    gf = torch.autograd.grad(current.mean(), z+[penalty])
    for actual, expected in zip(gs, gf):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('values', [[0, 0, 1], [1, 0, 2], [0, float('inf'), 2], [0, 1]])
def test_invalid_support_fails_instead_of_clipping(values):
    with pytest.raises(ValueError):
        ordinal_log_loss(torch.zeros(3), 1, values)


@pytest.mark.parametrize('bad', [{'kind':'unknown'}, {'kind':'noul_brier','noul_rho':1.},
                                {'kind':'score_ordinal','score_weight':-1.}])
def test_invalid_objective_fails(bad):
    with pytest.raises(ValueError):
        TypedObjective(**bad)


def test_native_numeric_median_diagnostic_is_paired_and_nonmutating():
    rows = [{'id':'one','task':'score','valid':True,'k':3,'support':[0,20,100],
             'label_index':1,'label':20,'prediction_index':0,'probabilities':[.4,.35,.25]}]
    before = copy.deepcopy(rows)
    out = score_decoding_diagnostic(rows)
    assert out['rows']==1 and out['mae']=={'argmax':20.,'median':0.}
    assert out['median_changed_predictions']==1 and rows==before
    rows[0]['valid']=False
    with pytest.raises(ValueError):score_decoding_diagnostic(rows)


def test_worker_uses_typed_loss_model():
    from models_typed_loss import FormalModel
    import worker_typed_loss
    assert worker_typed_loss.main.__globals__['FormalModel'] is FormalModel
