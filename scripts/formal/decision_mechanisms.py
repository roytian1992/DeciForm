"""Isolated score median supervision and question-conditioned QF geometry."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from kev_ours import TASKS
from kev_ours_v5 import anchored_maps
from kev_ours_v6 import OrdinalTaskGeometryHead


@dataclass(frozen=True)
class Mechanism:
    kind: str
    regression_weight: float = .5
    regression_rank: int = 64
    metric_strength: float = .5

    def __post_init__(self):
        if self.kind not in ('score_regression', 'choice_metric'):
            raise ValueError('Unknown isolated mechanism')
        if not math.isfinite(self.regression_weight) or self.regression_weight <= 0:
            raise ValueError('Positive finite regression weight required')
        if type(self.regression_rank) is not int or self.regression_rank < 1:
            raise ValueError('Positive integer regression rank required')
        if not math.isfinite(self.metric_strength) or self.metric_strength <= 0:
            raise ValueError('Positive finite metric strength required')

    def inventory(self):
        return asdict(self)


class MedianRegressionHead(nn.Module):
    """Unclipped normalized coordinate; native-range scaling stays in the loss."""
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.project = nn.Linear(hidden_size, rank)
        self.out = nn.Linear(rank, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden):
        hidden = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        return .5 + self.out(F.silu(self.project(hidden))).squeeze(-1).float()


def native_prediction(coordinate, values):
    if len(values) < 2 or any(not math.isfinite(v) for v in values) or any(
            a >= b for a, b in zip(values, values[1:])):
        raise ValueError('Native score support must be finite and strictly increasing')
    return values[0] + (values[-1] - values[0]) * coordinate


class ConditionalQuadraticReadout(nn.Module):
    """Negative anisotropic energy relative to the isotropic reference.

    Static and query-generated log precisions are bounded, softplus-positive,
    and normalized to mean one. At zero initialization, the correction is
    exactly zero. The parent head retains its existing final scalar tanh cap.
    Its division by sqrt(rank) is compensated in this readout.
    """
    def __init__(self, original, query_size, strength):
        super().__init__()
        self.weight = original.weight
        self.condition = nn.Linear(query_size, self.weight.shape[1], bias=False)
        nn.init.zeros_(self.condition.weight)
        self.strength = strength
        self.query = None

    def precisions(self, query):
        raw = self.weight.float() + self.condition(query).float()
        positive = F.softplus(2 * (raw / 2).tanh())
        return positive / positive.mean(-1, keepdim=True)

    def forward(self, squared_features):
        if self.query is None or len(self.query) != len(squared_features):
            raise RuntimeError('Missing matching query conditioning')
        precision = self.precisions(self.query)
        raw = self.strength * ((1-precision[:, None]) * squared_features).mean(-1, keepdim=True)
        return raw * math.sqrt(squared_features.shape[-1])


class MechanismHead(OrdinalTaskGeometryHead):
    def padded(self, h_decide, h_opts, valid, task_ids):
        self.regression_coordinates = {}
        conditional = self.mechanism.kind == 'choice_metric'
        if conditional:
            choices = (task_ids == TASKS.index('choice')).nonzero().flatten()
            if choices.numel():
                q = self.q(h_decide[choices])
                geometry = anchored_maps(self.geometry_raw, self.task_weights, .5)[task_ids[choices]].to(q.dtype)
                q = q + torch.bmm(geometry.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)
                self.qf_out.query = F.layer_norm(q.float(), (q.shape[-1],))
        try:
            result = super().padded(h_decide, h_opts, valid, task_ids)
        finally:
            if conditional:
                self.qf_out.query = None
        if self.mechanism.kind == 'score_regression':
            scores = (task_ids == TASKS.index('score')).nonzero().flatten()
            if scores.numel():
                coordinates = self.score_regression(h_decide[scores])
                self.regression_coordinates = dict(zip(scores.tolist(), coordinates.unbind()))
        return result


def install_mechanism(head, mechanism):
    if not isinstance(head, OrdinalTaskGeometryHead):
        raise TypeError('Expected the frozen ordinal QF head')
    head.__class__ = MechanismHead
    head.mechanism = mechanism
    head.regression_coordinates = {}
    # Do not change the base model's RNG trajectory or existing parameter values.
    with torch.random.fork_rng(devices=[]):
        if mechanism.kind == 'score_regression':
            head.score_regression = MedianRegressionHead(
                head.q.in_features, mechanism.regression_rank).to(head.q.weight.device)
        else:
            head.qf_out = ConditionalQuadraticReadout(
                head.qf_out, head.q.out_features, mechanism.metric_strength).to(head.q.weight.device)


def extra_parameters(mechanism, hidden_size=4096, query_size=256, qf_rank=64):
    if mechanism.kind == 'score_regression':
        return hidden_size*mechanism.regression_rank + 2*mechanism.regression_rank + 1
    return query_size*qf_rank


def regression_diagnostic(rows):
    scores = [r for r in rows if r['task'] == 'score']
    if not scores or len({r['id'] for r in scores}) != len(scores):
        raise ValueError('Expected complete unique score predictions')
    totals = dict(raw=0., clipped=0., nearest_native=0.)
    for row in scores:
        values = row['support']
        if not row['valid'] or len(values) != row['k'] or values[row['label_index']] != row['label']:
            raise ValueError('Incomplete score prediction')
        native_prediction(0., values)
        m = row['regression_prediction']
        if not math.isfinite(m):
            raise ValueError('Nonfinite regression output')
        clipped = min(values[-1], max(values[0], m))
        nearest = min(values, key=lambda v: abs(v-m))
        for key, value in [('raw', m), ('clipped', clipped), ('nearest_native', nearest)]:
            totals[key] += abs(value-row['label'])
    return {'split': 'dev', 'rows': len(scores), 'native_full_candidates': True,
            'mae': {k: v/len(scores) for k, v in totals.items()},
            'selection_use': False, 'original_predictions_modified': False,
            'scope': 'auxiliary regression diagnostic only; headline remains categorical argmax',
            'nearest_tie_rule': 'lower native support value'}
