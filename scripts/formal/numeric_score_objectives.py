"""Proper distribution losses and a coherent predictive-mean regression auxiliary."""
from dataclasses import asdict, dataclass
import math

import torch

from score_specialist import SpecialistObjective, data_loss as cdf_loss, interpolation


@dataclass(frozen=True)
class NumericScoreObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127
    mean_weight: float = 1.0
    crps_scale: float = 6.076770158115425

    def __post_init__(self):
        if self.kind not in ('cdf_mean', 'crps'):
            raise ValueError('Expected cdf_mean or crps')
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.ordinal_scale, self.crps_scale)):
            raise ValueError('Loss scales must be positive and finite')
        if not math.isfinite(self.mean_weight) or self.mean_weight < 0:
            raise ValueError('Mean weight must be finite and nonnegative')
        if self.kind == 'crps' and self.mean_weight != 0:
            raise ValueError('Pure CRPS arm has no regression auxiliary')

    def inventory(self):
        return asdict(self)


def loss_components(logits, values, label, objective):
    if logits.ndim != 1 or len(logits) != len(values):
        raise ValueError('One logit for every native score level is required')
    if len(values) < 2 or any(not math.isfinite(v) for v in values) or any(
            a >= b for a, b in zip(values, values[1:])):
        raise ValueError('Complete increasing finite support required')
    q = logits.new_tensor(interpolation(values, label))
    z = logits if logits.dtype == torch.float64 else logits.float()
    q = q.to(z.dtype)
    probabilities = z.softmax(0)
    support = z.new_tensor(values)
    normalized = (support - support[0]) / (support[-1] - support[0])
    target = (float(label) - support[0]) / (support[-1] - support[0])
    mean_error = ((probabilities * normalized).sum() - target).square()
    cdf = cdf_loss(z, values, label, SpecialistObjective('soft_cdf', objective.ordinal_scale))
    # For fractional targets the omitted q-CDF variance is label-only. These
    # runs use native integer labels, so this equals normalized discrete CRPS.
    crps = (normalized.diff() * (probabilities.cumsum(0)[:-1] - q.cumsum(0)[:-1]).square()).sum()
    total = (cdf + objective.mean_weight * mean_error if objective.kind == 'cdf_mean'
             else objective.crps_scale * crps)
    return {'total': total, 'cdf_log': cdf, 'normalized_mean_mse': mean_error,
            'normalized_crps': crps}


def data_loss(logits, values, label, objective):
    return loss_components(logits, values, label, objective)['total']


def logit_diagnostics(logits, values, label, objective):
    with torch.enable_grad():
        z = logits.detach().float().requires_grad_(True)
        value = data_loss(z, values, label, objective)
        ce = cdf_loss(z, values, label, SpecialistObjective('soft_ce'))
        grad = torch.autograd.grad(value, z, retain_graph=True)[0]
        ce_grad = torch.autograd.grad(ce, z)[0]
    p = z.detach().softmax(0)
    q = p.new_tensor(interpolation(values, label))
    return torch.stack((p.new_ones(()), value.detach(), ce.detach(), grad.norm(),
                        ce_grad.norm(), (p*q).sum(), p.new_tensor(len(p)))).double()
