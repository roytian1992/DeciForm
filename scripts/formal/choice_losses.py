"""Choice-only objectives with a fixed CE gradient reference at uniform logits.

All inputs are one unpadded, already-selected candidate inventory. These losses
never sample candidates and never alter prediction/decoding semantics.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ChoiceObjective:
    kind: str
    pair_weight: float = 0.5
    focal_gamma: float = 1.0
    normalization: str = 'uniform_gradient'

    def __post_init__(self):
        if self.kind not in ('ce', 'ce_pair', 'focal'):
            raise ValueError('Unknown choice objective')
        if not math.isfinite(self.pair_weight) or not 0 <= self.pair_weight <= 1:
            raise ValueError('pair_weight must be in [0, 1]')
        if not math.isfinite(self.focal_gamma) or not (
                self.focal_gamma == 0 or self.focal_gamma >= 1):
            raise ValueError('Use focal gamma=0 or gamma>=1 for finite boundary derivatives')
        if self.normalization != 'uniform_gradient':
            raise ValueError('This comparison fixes uniform-gradient normalization')

    def inventory(self):
        return asdict(self)


def _inputs(logits, label):
    if logits.ndim != 1 or logits.numel() < 2 or not logits.is_floating_point():
        raise ValueError('Expected at least two unpadded floating-point logits')
    if type(label) is not int or not 0 <= label < logits.numel():
        raise ValueError('Expected an in-range integer target')
    return logits if logits.dtype == torch.float64 else logits.float()


@lru_cache(maxsize=256)
def focal_uniform_scale(k, gamma):
    if gamma == 0:
        return 1.0
    p = 1.0 / k
    derivative_scale = (1-p)**gamma - gamma*p*(1-p)**(gamma-1)*math.log(p)
    return 1.0 / derivative_scale


def choice_loss(logits, label, objective):
    z = _inputs(logits, label)
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    if objective.kind == 'ce':
        return ce
    if objective.kind == 'ce_pair':
        negatives = torch.cat((z[:label], z[label+1:]))
        pair = (2.0 / z.numel()) * F.softplus(negatives-z[label]).sum()
        return (1-objective.pair_weight)*ce + objective.pair_weight*pair
    if objective.focal_gamma == 0:
        return ce
    # expm1 retains small 1-p_y values without subtracting nearly equal floats.
    modulation = (-torch.expm1(-ce)).pow(objective.focal_gamma)
    return focal_uniform_scale(z.numel(), objective.focal_gamma)*modulation*ce


@torch.no_grad()
def logit_diagnostics(logits, label, objective, *, choice):
    """Detached output-gradient proxies, not shared-backbone gradient norms.

    Return row count, loss, reference CE, logit-gradient L2, CE-gradient L2,
    true-label probability and candidate count. Structural penalties excluded.
    """
    z = _inputs(logits.detach(), label)
    logp = F.log_softmax(z, dim=0)
    p = logp.exp()
    ce = -logp[label]
    ce_grad = p.clone()
    ce_grad[label] -= 1
    grad = ce_grad
    value = ce
    if choice:
        value = choice_loss(z, label, objective)
        if objective.kind == 'ce_pair':
            pair_grad = (2.0/z.numel()) * torch.sigmoid(z-z[label])
            pair_grad[label] = 0
            pair_grad[label] = -pair_grad.sum()
            grad = (1-objective.pair_weight)*ce_grad + objective.pair_weight*pair_grad
        elif objective.kind == 'focal' and objective.focal_gamma != 0:
            gamma = objective.focal_gamma
            r = -torch.expm1(logp[label])
            factor = r.pow(gamma) - gamma*p[label]*r.pow(gamma-1)*logp[label]
            grad = focal_uniform_scale(z.numel(), gamma)*factor*ce_grad
    return torch.stack((z.new_ones(()), value, ce, grad.norm(), ce_grad.norm(),
                        p[label], z.new_tensor(z.numel()))).double()
