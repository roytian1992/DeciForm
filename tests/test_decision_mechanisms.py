"""Gradient flow, task isolation, native score semantics and set-head contracts."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
import common
from decision_mechanisms import (Mechanism, ConditionalQuadraticReadout,
    MedianRegressionHead, install_mechanism, native_prediction, extra_parameters,
    regression_diagnostic)
from models_decision_mechanism import FormalModel
from kev_ours_v5 import TaskGeometryHead, optimizer_groups
from kev_ours_v6 import OrdinalTaskGeometryHead, OrdinalCDFReadout


def head():
    torch.manual_seed(17)
    h = TaskGeometryHead(16, 8, rank=4)
    h.__class__ = OrdinalTaskGeometryHead
    h.ordinal = OrdinalCDFReadout(16, rank=4)
    h.numeric_supports = [None, None, [0, 1, 2, 3, 4]]
    return h


def inputs(k=77):
    return (torch.randn(3, 16), torch.randn(3, k, 16),
            torch.arange(k)[None] < torch.tensor([k, 2, 5])[:, None],
            torch.tensor([0, 1, 2]))


@pytest.mark.parametrize('kind', ['score_regression', 'choice_metric'])
def test_install_preserves_rng_existing_parameters_and_initial_predictions(kind):
    reference = head()
    current = copy.deepcopy(reference)
    state = torch.get_rng_state().clone()
    old = {n: p.clone() for n, p in current.named_parameters()}
    mechanism = Mechanism(kind, regression_rank=4)
    install_mechanism(current, mechanism)
    assert torch.equal(torch.get_rng_state(), state)
    for n, value in old.items():
        torch.testing.assert_close(dict(current.named_parameters())[n], value, rtol=0, atol=0)
    assert sum(p.numel() for p in current.parameters()) - sum(p.numel() for p in reference.parameters()) == extra_parameters(mechanism, 16, 8, 4)
    args = inputs()
    torch.testing.assert_close(current.padded(*args), reference.padded(*args), rtol=0, atol=0)
    if kind == 'score_regression':
        assert set(current.regression_coordinates) == {2}
        assert current.regression_coordinates[2] == .5


@pytest.mark.parametrize('k', [5, 21, 77])
def test_conditional_metric_changes_choice_only_and_is_equivariant(k):
    reference = head()
    current = copy.deepcopy(reference)
    install_mechanism(current, Mechanism('choice_metric'))
    with torch.no_grad():
        current.qf_out.condition.weight.normal_(0, .4)
    args = inputs(k)
    before, after = reference.padded(*args), current.padded(*args)
    assert not torch.allclose(before[0], after[0])
    torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
    perm = torch.randperm(k)
    q, candidates, valid, task = args
    current.numeric_supports = [None]
    permuted = current.padded(q[:1], candidates[:1, perm], valid[:1, perm], task[:1])
    torch.testing.assert_close(permuted[0], after[0, perm], atol=1e-6, rtol=1e-5)
    assert current.qf_out.query is None
    current.numeric_supports = [None, None, [0, 1, 2, 3, 4]]
    current.padded(*args)[:, :2].sum().backward()
    assert current.qf_out.condition.weight.grad.abs().sum() > 0
    assert current.project.weight.grad.abs().sum() > 0
    assert current.query_condition.weight.grad.abs().sum() > 0


def test_metric_precisions_are_positive_normalized_query_dependent_and_finite():
    readout = ConditionalQuadraticReadout(nn.Linear(4, 1, bias=False), 4, .5)
    with torch.no_grad():
        readout.weight.zero_()
        readout.condition.weight.copy_(torch.eye(4))
    q = torch.tensor([[10000., -10000., 0., 1.], [-10000., 10000., 0., 1.]])
    p = readout.precisions(q)
    assert torch.isfinite(p).all() and (p > 0).all()
    torch.testing.assert_close(p.mean(-1), torch.ones(2))
    assert p[0, 0] > p[0, 1] and p[1, 0] < p[1, 1]
    with pytest.raises(RuntimeError):
        readout(torch.ones(2, 5, 4))


def test_regression_is_native_mae_without_training_clipping_or_mean_constraint():
    m = torch.tensor(-.5, requires_grad=True)
    values = [0, 20, 100]
    predicted = native_prediction(m, values)
    assert predicted == -50
    loss = (predicted-20).abs()
    assert loss == 70
    assert torch.autograd.grad(loss, m)[0] == -100
    h = MedianRegressionHead(16, 4)
    with torch.no_grad():
        h.out.weight.normal_(0, .2)
    hidden = torch.randn(2, 16, requires_grad=True)
    (native_prediction(h(hidden), values)-20).abs().mean().backward()
    assert hidden.grad.abs().sum() > 0
    assert h.project.weight.grad.abs().sum() > 0


def test_mixed_loss_keeps_noul_bce_and_choice_ce_and_microbatch_gradients():
    rows = [
        {'task': 'choice', 'question': {'options': ['a', 'b', 'c']}, 'label': 'b'},
        {'task': 'noul', 'question': {}, 'label': True},
        {'task': 'score', 'question': {'scale': {'min': 0, 'max': 4}}, 'label': 3},
    ]
    z = [torch.randn(k, requires_grad=True) for k in [3, 2, 5]]
    coordinate = torch.tensor(.25, requires_grad=True)
    penalty = torch.tensor(.03, requires_grad=True)
    h = SimpleNamespace(regression_coordinates={})
    def scores(rs, es):
        h.regression_coordinates = {i: coordinate for i, r in enumerate(rs) if r['task']=='score'}
        return es
    model = SimpleNamespace(scores=scores, net=SimpleNamespace(head=h,
        regularization=lambda: penalty.square()), mechanism=Mechanism('score_regression'),
        collect_loss_diagnostics=False)
    actual = FormalModel.losses(model, rows, z)
    for i in range(2):
        ref = (F.cross_entropy(z[i][None], torch.tensor([1])) if i == 0 else
               F.binary_cross_entropy_with_logits(z[i][1]-z[i][0], torch.tensor(1.)))
        torch.testing.assert_close(actual[i], ref+penalty.square(), rtol=0, atol=0)
    expected = F.cross_entropy(z[2][None], torch.tensor([3]))+.5*abs(4*coordinate-3)+penalty.square()
    torch.testing.assert_close(actual[2], expected)
    split = sum(FormalModel.losses(model, rows[a:b], z[a:b]).sum()/3 for a,b in [(0,1),(1,3)])
    parameters = z+[coordinate, penalty]
    g1 = torch.autograd.grad(actual.mean(), parameters, retain_graph=True)
    g2 = torch.autograd.grad(split, parameters)
    for a,b in zip(g1,g2):
        torch.testing.assert_close(a,b)


@pytest.mark.parametrize('kind', ['score_regression', 'choice_metric'])
def test_optimizer_includes_new_module_once_at_residual_learning_rate(kind):
    net = nn.Module()
    net.lm = nn.Linear(2, 2)
    net.head = head()
    install_mechanism(net.head, Mechanism(kind, regression_rank=4))
    groups = optimizer_groups(net, 2e-5, 2e-5, 1e-3, 1e-3, .01)
    ps = [p for g in groups for p in g['params']]
    assert len(ps) == len({id(p) for p in ps})
    assert {id(p) for p in ps} == {id(p) for p in net.parameters()}
    branch = net.head.score_regression if kind=='score_regression' else net.head.qf_out.condition
    residual = next(g for g in groups if g['name']=='residual')
    assert residual['lr']==1e-3
    assert {id(p) for p in branch.parameters()} <= {id(p) for p in residual['params']}


def test_regression_diagnostic_uses_native_values_and_leaves_primary_unchanged():
    rows = [{'id':'one', 'task':'score', 'valid':True, 'k':3, 'support':[0,20,100],
             'label_index':1, 'label':20, 'prediction_index':0,
             'probabilities':[.4,.35,.25], 'regression_prediction':60.}]
    before = copy.deepcopy(rows)
    d = regression_diagnostic(rows)
    assert d['mae']=={'raw':40., 'clipped':40., 'nearest_native':0.}
    assert rows==before


def test_prediction_uses_one_forward_and_preserves_argmax_with_regression_output():
    row = {'task':'score', 'question':{'scale':{'min':0, 'max':4}}, 'label':4}
    head = SimpleNamespace(regression_coordinates={0:torch.tensor(.9)})
    calls = []
    def scores(rs, es):
        calls.append(1)
        return [torch.tensor([4., 0., 0., 0., 0.])]
    model = SimpleNamespace(scores=scores, net=SimpleNamespace(head=head),
        mechanism=Mechanism('score_regression'), clear=lambda:None)
    pred = FormalModel.predict(model, [row], [], 0)
    assert len(calls)==1 and pred[0]['prediction_index']==0
    assert pred[0]['regression_prediction']==pytest.approx(3.6)


@pytest.mark.parametrize('values', [[0], [0, 0], [1, 0], [0, float('nan')]])
def test_invalid_native_ranges_fail(values):
    with pytest.raises(ValueError):
        native_prediction(torch.tensor(.5), values)


def test_worker_references_new_model_and_decoding_diagnostics():
    import worker_decision_mechanism as worker
    assert worker.main.__globals__['FormalModel'] is FormalModel
    assert worker.main.__globals__['regression_diagnostic'] is regression_diagnostic
