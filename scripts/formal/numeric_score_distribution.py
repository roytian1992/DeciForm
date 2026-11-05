"""Conditional monotone CDF and MAE-preserving projection onto full score grids.

The distribution has adaptive-width uniform-density bins and endpoint atoms.
Its integrated CDF is analytic. Averaging that CDF over each adjacent output
interval is the linear (hat-function) categorical projection, not hard rounding.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def projected_log_probabilities(width_logits, mass_logits, support, min_width=1e-4):
    bins = width_logits.numel()
    if width_logits.ndim != 1 or mass_logits.shape != (bins + 2,):
        raise ValueError('Expected adaptive bin widths and bin/endpoint probability masses')
    if not 0 <= min_width < 1 / bins:
        raise ValueError('Invalid minimum bin width')
    values = torch.as_tensor(support, dtype=torch.float64, device=width_logits.device)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all() or not torch.all(values.diff() > 0):
        raise ValueError('Complete increasing finite support required')
    with torch.autocast(device_type=width_logits.device.type, enabled=False):
        widths = min_width + (1 - bins*min_width)*width_logits.double().softmax(-1)
        masses = mass_logits.double().softmax(-1)
        right = widths.cumsum(0)
        left = torch.cat((widths.new_zeros(1), right[:-1]))
        t = (values - values[0]) / (values[-1] - values[0])
        inside = (t[:, None] - left).clamp_min(0).minimum(widths)
        # Integral of clamp((t-left)/width, 0, 1), with the lower endpoint atom.
        integral = masses[0]*t + ((.5*inside.square()/widths +
                                   (t[:, None]-right).clamp_min(0))*masses[1:-1]).sum(-1)
        cumulative = integral.diff()/t.diff()
        probabilities = torch.cat((cumulative[:1], cumulative.diff(), 1-cumulative[-1:]))
        # Only a finite-arithmetic floor; float64 avoids cancellation on fine grids.
        probabilities = probabilities.clamp_min(1e-12)
        return (probabilities.log() - probabilities.sum().log()).to(width_logits.dtype)


class NumericCDFReadout(nn.Module):
    def __init__(self, input_size, hidden_size=64, bins=32):
        super().__init__()
        self.bins = bins
        self.project = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2*bins + 2)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            # Initially nearly uniform density, with small but representable end masses.
            mass = torch.tensor([.001] + [.998/bins]*bins + [.001])
            self.output.bias[bins:].copy_(mass.log())

    def forward(self, features, supports):
        if len(features) != len(supports):
            raise ValueError('Feature/support batch mismatch')
        h = F.silu(self.project(F.layer_norm(features.float(), (features.shape[-1],))))
        parameters = self.output(h).float()
        return [projected_log_probabilities(p[:self.bins], p[self.bins:], s)
                for p, s in zip(parameters, supports)]
