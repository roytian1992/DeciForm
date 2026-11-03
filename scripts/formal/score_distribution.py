"""Proper distribution objectives and explicit native score decision rules."""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from typed_losses import ordinal_log_loss, support_weights

PROTOCOL = 'score_distribution_20261104_v1'


@dataclass(frozen=True)
class DistributionObjective:
    kind: str
    moment_weight: float = .5
    ordinal_scale: float = 1.

    def __post_init__(self):
        if self.kind not in ('mean_moment', 'ordinal_main'):
            raise ValueError('Unknown distribution objective')
        if not math.isfinite(self.moment_weight) or self.moment_weight < 0:
            raise ValueError('Finite nonnegative moment weight required')
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale <= 0:
            raise ValueError('Finite positive global ordinal scale required')

    def inventory(self):
        return asdict(self)


def task_data_loss(logits, label, task, objective, values=None):
    z = logits if logits.dtype == torch.float64 else logits.float()
    if z.ndim != 1 or len(z) < 2 or type(label) is not int or not 0 <= label < len(z):
        raise ValueError('Expected one complete unpadded candidate inventory')
    if task == 'noul':
        if len(z) != 2:
            raise ValueError('Noul has exactly two outcomes')
        return F.binary_cross_entropy_with_logits(z[1]-z[0], z.new_tensor(label))
    if task not in ('choice', 'score'):
        raise ValueError('Unknown task')
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    if task == 'choice':
        return ce
    support_weights(values, len(z), z)  # validates native, finite, ordered support
    if objective.kind == 'ordinal_main':
        return objective.ordinal_scale * ordinal_log_loss(z, label, values)
    s = z.new_tensor(values)
    mean = (z.softmax(0)*s).sum()
    # Division by span gives the auxiliary the same linear scale transformation
    # as native MAE. This is a squared error of the distribution mean, not an
    # absolute-error penalty on a mean and not expected per-class squared error.
    moment = (mean-s[label]).square()/(s[-1]-s[0])
    return ce + objective.moment_weight*moment


@torch.no_grad()
def logit_diagnostics(logits, label, task, objective, values=None):
    z = logits.detach() if logits.dtype == torch.float64 else logits.detach().float()
    p = z.softmax(0)
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    ce_grad = p.clone()
    ce_grad[label] -= 1
    grad = ce_grad.clone()
    if task == 'score':
        weights = support_weights(values, len(z), z)
        if objective.kind == 'mean_moment':
            s = z.new_tensor(values)
            mean = (p*s).sum()
            grad += objective.moment_weight*2*(mean-s[label])/(s[-1]-s[0])*p*(s-mean)
        else:
            grad.zero_()
            for boundary, weight in enumerate(weights):
                selected = slice(0, boundary+1) if label <= boundary else slice(boundary+1, None)
                derivative = p.clone()
                derivative[selected] -= z[selected].softmax(0)
                grad += objective.ordinal_scale*weight*derivative
    value = task_data_loss(z, label, task, objective, values)
    return torch.stack((z.new_ones(()), value, ce, grad.norm(), ce_grad.norm(),
                        p[label], z.new_tensor(len(z)))).double()


def decisions(values, probabilities):
    values = [float(v) for v in values]
    probabilities = [float(p) for p in probabilities]
    if len(values) < 2 or len(values) != len(probabilities) or any(
            not math.isfinite(v) for v in values) or any(a >= b for a,b in zip(values, values[1:])):
        raise ValueError('Complete finite increasing native support required')
    mass = math.fsum(probabilities)
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or abs(mass-1) > 1e-5:
        raise ValueError('Invalid complete predictive distribution')
    p = [v/mass for v in probabilities]
    mode = max(range(len(p)), key=p.__getitem__)
    cumulative = 0.
    median = len(p)-1
    for i, value in enumerate(p):
        cumulative += value
        if cumulative >= .5:
            median = i
            break
    mean = math.fsum(v*prob for v,prob in zip(values,p))
    nearest = min(range(len(values)), key=lambda i: abs(values[i]-mean))
    return {'indices': {'argmax': mode, 'median': median, 'mean_nearest': nearest},
            'values': {'argmax': values[mode], 'median': values[median],
                       'mean': mean, 'mean_nearest': values[nearest]}}


def score_rule_report(rows, split):
    scores = [row for row in rows if row['task'] == 'score']
    if not scores or len(scores) != len({r['id'] for r in scores}):
        raise ValueError('Nonempty unique score records required')
    groups = {'all': {}}
    for row in scores:
        if not row['valid'] or row['k'] != len(row['support']):
            raise ValueError('Invalid/incomplete score prediction; never omit failures')
        if row['support'][row['label_index']] != row['label']:
            raise ValueError('Native label and index differ')
        ds = decisions(row['support'], row['probabilities'])
        rule = row.get('score_decision_rule', 'argmax')
        if rule not in ds['indices'] or ds['indices'][rule] != row['prediction_index']:
            raise ValueError('Stored prediction differs from its declared decision rule')
        for group in ['all', 'source/'+row['source']]:
            totals = groups.setdefault(group, {})
            for key,value in ds['values'].items():
                d = totals.setdefault(key, {'rows': 0, 'mae': 0., 'mse': 0.})
                error = abs(value-row['label'])
                d['rows'] += 1
                d['mae'] += error
                d['mse'] += error*error
    for totals in groups.values():
        for d in totals.values():
            d['mae'] /= d['rows']
            d['mse'] /= d['rows']
    return {'evaluation_protocol': PROTOCOL, 'split': split, 'rows': len(scores),
            'native_full_candidates': True, 'primary_rule': 'median',
            'primary_metric': 'native MAE', 'primary_mae': groups['all']['median']['mae'],
            'rules': groups['all'], 'by_source': {k[7:]:v for k,v in groups.items() if k.startswith('source/')},
            'mean_is_continuous_diagnostic': True, 'original_predictions_modified': False,
            'tie_rules': 'lower native value for median and nearest-mean ties'}
