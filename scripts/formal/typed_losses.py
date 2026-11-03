"""Isolated Bernoulli and ordinal loss treatments on A4 logits.

The ordinal term uses log partition functions, not clipped probabilities.
Input inventories and labels are unchanged; only one task's objective changes.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TypedObjective:
    kind: str
    noul_rho: float = 0.5
    score_weight: float = 0.1

    def __post_init__(self):
        if self.kind not in ('ce', 'noul_brier', 'score_ordinal'):
            raise ValueError('Unknown typed loss objective')
        if not math.isfinite(self.noul_rho) or not 0 <= self.noul_rho < 1:
            raise ValueError('noul_rho must be finite and in [0, 1)')
        if not math.isfinite(self.score_weight) or self.score_weight < 0:
            raise ValueError('score_weight must be finite and nonnegative')

    def inventory(self):
        return asdict(self)


def _inputs(logits, label):
    if logits.ndim != 1 or logits.numel() < 2 or not logits.is_floating_point():
        raise ValueError('Expected at least two unpadded floating-point logits')
    if type(label) is not int or not 0 <= label < logits.numel():
        raise ValueError('Expected an in-range integer label')
    return logits if logits.dtype == torch.float64 else logits.float()


def support_weights(values, k, z):
    # Native supports come from CPU metadata: no per-row GPU synchronization.
    if values is None or len(values) != k:
        raise ValueError('Score support must match the complete current inventory')
    numbers = [float(v) for v in values]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError('Score support must be finite')
    widths = [b-a for a, b in zip(numbers, numbers[1:])]
    if not all(w > 0 for w in widths):
        raise ValueError('Score support must be strictly increasing')
    span = numbers[-1]-numbers[0]
    return z.new_tensor([w/span for w in widths])


def ordinal_log_loss(logits, label, values):
    z = _inputs(logits, label)
    weights = support_weights(values, z.numel(), z)
    left = z.logcumsumexp(0)[:-1]
    right = z.flip(0).logcumsumexp(0).flip(0)[1:]
    true_left = torch.arange(z.numel()-1, device=z.device) >= label
    selected = torch.where(true_left, left, right)
    return (weights*(z.logsumexp(0)-selected)).sum()


def task_data_loss(logits, label, task, objective, values=None):
    z = _inputs(logits, label)
    if task == 'noul':
        if z.numel() != 2:
            raise ValueError('Noul requires exactly two outcomes')
        margin = z[1]-z[0]
        target = z.new_tensor(label)
        bce = F.binary_cross_entropy_with_logits(margin, target)
        if objective.kind == 'noul_brier':
            brier = 2*(margin.sigmoid()-target).square()
            return (1-objective.noul_rho)*bce + objective.noul_rho*brier
        return bce
    if task not in ('choice', 'score'):
        raise ValueError('Unsupported task')
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    if task == 'score' and objective.kind == 'score_ordinal':
        return ce + objective.score_weight*ordinal_log_loss(z, label, values)
    return ce


@torch.no_grad()
def logit_diagnostics(logits, label, task, objective, values=None):
    """Per-row output-gradient proxies, excluding regularization/backbone Jacobians."""
    z = _inputs(logits.detach(), label)
    p = z.softmax(0)
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    ce_grad = p.clone()
    ce_grad[label] -= 1
    grad = ce_grad.clone()
    value = task_data_loss(z, label, task, objective, values)
    if task == 'noul' and objective.kind == 'noul_brier':
        grad *= (1-objective.noul_rho)+4*objective.noul_rho*p[1]*(1-p[1])
    elif task == 'score' and objective.kind == 'score_ordinal':
        weights = support_weights(values, z.numel(), z)
        # Stable conditional softmax gradients even for confidently wrong CDFs.
        # Used only on diagnostic steps, with no extra backbone pass/backward.
        for boundary, weight in enumerate(weights):
            selected = slice(0, boundary+1) if label <= boundary else slice(boundary+1, None)
            boundary_grad = p.clone()
            boundary_grad[selected] -= z[selected].softmax(0)
            grad += objective.score_weight*weight*boundary_grad
    return torch.stack((z.new_ones(()), value, ce, grad.norm(), ce_grad.norm(),
                        p[label], z.new_tensor(z.numel()))).double()
