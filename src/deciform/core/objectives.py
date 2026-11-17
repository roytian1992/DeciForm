"""Log+spherical classification and interpolated decimal CE/CDF objectives."""

import math
from bisect import bisect_left
from dataclasses import dataclass, asdict
import torch
from .schemas import support, semantic_key


def interpolation(values, label):
    y = float(label)
    if (
        len(values) < 2
        or any(not math.isfinite(float(v)) for v in values)
        or any(a >= b for a, b in zip(values, values[1:]))
    ):
        raise ValueError("Full strictly increasing finite support required")
    if not math.isfinite(y) or not values[0] <= y <= values[-1]:
        raise ValueError("Raw target outside declared range")
    j = bisect_left(values, y)
    q = [0.0] * len(values)
    if values[j] == y:
        q[j] = 1.0
    else:
        q[j - 1] = (values[j] - y) / (values[j] - values[j - 1])
        q[j] = 1 - q[j - 1]
    return q


@dataclass(frozen=True)
class SpecialistObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127

    def __post_init__(self):
        if (
            self.kind not in ("soft_ce", "soft_cdf")
            or not math.isfinite(self.ordinal_scale)
            or self.ordinal_scale <= 0
        ):
            raise ValueError("Invalid score objective")

    def inventory(self):
        return asdict(self)


def data_loss(logits, values, label, objective):
    if logits.ndim != 1 or len(logits) != len(values):
        raise ValueError("Logits must cover full support")
    z = logits if logits.dtype == torch.float64 else logits.float()
    q = z.new_tensor(interpolation(values, label))
    logz = z.logsumexp(0)
    if objective.kind == "soft_ce":
        return -(q * (z - logz)).sum()
    left = z.logcumsumexp(0)[:-1]
    right = z.flip(0).logcumsumexp(0).flip(0)[1:]
    qcdf = q.cumsum(0)[:-1]
    widths = z.new_tensor([b - a for a, b in zip(values, values[1:])]) / (values[-1] - values[0])
    return (
        objective.ordinal_scale
        * (widths * (qcdf * (logz - left) + (1 - qcdf) * (logz - right))).sum()
    )


def proper_loss(logits, target, alpha=0.75):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    z = logits if logits.dtype == torch.float64 else logits.float()
    if z.ndim != 1 or len(z) < 2:
        raise ValueError("One complete candidate vector required")
    if isinstance(target, int):
        if not 0 <= target < len(z):
            raise ValueError("Target outside support")
        logp = z.log_softmax(0)
        spherical = -torch.expm1(logp[target] - 0.5 * torch.logsumexp(2 * logp, 0))
        return (-logp[target] + alpha * math.sqrt(len(z)) * spherical) / (1 + alpha)
    q = torch.as_tensor(target, device=z.device, dtype=z.dtype)
    if (
        q.shape != z.shape
        or not bool(torch.isfinite(q).all())
        or bool((q < 0).any())
        or not torch.allclose(q.sum(), q.new_tensor(1.0), atol=1e-7)
    ):
        raise ValueError("Invalid target distribution")
    logp = z.log_softmax(0)
    spherical = (q * (-torch.expm1(logp - 0.5 * torch.logsumexp(2 * logp, 0)))).sum()
    return (-(q * logp).sum() + alpha * math.sqrt(len(z)) * spherical) / (1 + alpha)


def request_loss(logits, row, config):
    values = support(row)
    if row["task"] == "score":
        weight = float(config.get("cdf_weight", 0.5))
        if not 0 <= weight <= 1:
            raise ValueError("cdf_weight must be in [0,1]")
        ce = data_loss(logits, values, row["label"], SpecialistObjective("soft_ce"))
        cdf = data_loss(
            logits,
            values,
            row["label"],
            SpecialistObjective("soft_cdf", config.get("ordinal_scale", 3.0383850790577127)),
        )
        return (1 - weight) * ce + weight * cdf
    target = row.get("target_distribution")
    if target is None:
        target = list(map(semantic_key, values)).index(semantic_key(row["label"]))
    return proper_loss(logits, target, config.get("classification_alpha", 0.75))
