"""Per-row Laya 0.4.26 RLCD, retaining microbatch advantage normalization.

Only the final reduction differs from upstream: the caller averages these rows
over the global batch. No network parameter or extra backbone pass is added.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parents[2] / 'runs/formal_laya_20261106_v1/vendor/laya_release'
sys.path.insert(0, str(VENDOR))
from laya.common import QTYPES, proper_reward
from score_specialist import support, interpolation


@dataclass(frozen=True)
class RLCDObjective:
    kind: str = 'laya_rlcd'
    sigma: float = .4
    samples: int = 4
    w_sph: float = .75
    w_rps: float = 1.

    def __post_init__(self):
        if asdict(self) != dict(kind='laya_rlcd', sigma=.4, samples=4, w_sph=.75, w_rps=1.):
            raise ValueError('This comparison freezes the native Laya RLCD settings')

    def inventory(self):
        return asdict(self)


def rlcd_rows(logits, target, mask, objective):
    """Mean equals upstream rlcd_loss under the same RNG and padded batch."""
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((objective.samples,) + logits.shape, device=logits.device) * objective.sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    noisy = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    tasks = torch.full((logits.shape[0],), QTYPES['score'], device=logits.device, dtype=torch.long)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), tasks, fmask,
                               w_sph=objective.w_sph, w_rps=objective.w_rps)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    log_density = -(((noisy - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * objective.sigma ** 2)
    ce = -(target * F.log_softmax(logits.masked_fill(~mask, -1e4), -1) * mask).sum(-1)
    return -(advantage * log_density).mean(0) + ce


class RLCDLossMixin:
    def losses(self, rows, encodings):
        if any(r['task'] != 'score' for r in rows):
            raise ValueError('This study only replaces the Score objective')
        zs = self.scores(rows, encodings)
        lengths = [len(z) for z in zs]
        maximum = max(lengths)
        z = torch.stack([F.pad(v.float(), (0, maximum-len(v))) for v in zs])
        mask = torch.arange(maximum, device=z.device)[None] < z.new_tensor(lengths, dtype=torch.long)[:, None]
        targets = []
        for row, count in zip(rows, lengths):
            values = support(row)
            if len(values) != count:
                raise ValueError('Every original legal score must be present')
            targets.append(F.pad(z.new_tensor(interpolation(values, row['label'])), (0, maximum-count)))
        target = torch.stack(targets)
        # Explicit FP32, including the reward normalization over the whole microbatch.
        with torch.autocast(z.device.type, enabled=False):
            losses = rlcd_rows(z, target, mask, self.distribution_objective)
        if self.collect_loss_diagnostics:
            gradient = torch.autograd.grad(losses.sum(), z, retain_graph=True)[0].detach()
            with torch.no_grad():
                p = z.masked_fill(~mask, -1e4).softmax(-1)
                ce = -(target*z.masked_fill(~mask, -1e4).log_softmax(-1)).sum(-1)
                fields = torch.stack([torch.ones_like(ce), losses.detach(), ce,
                    gradient.norm(dim=-1), (p-target).norm(dim=-1),
                    (target*p).sum(-1), mask.sum(-1).float()], dim=-1).double()
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((1,7), device=z.device, dtype=torch.float64)
                self._loss_diagnostics += fields.sum(0,keepdim=True)
        return losses + self.net.regularization()

    def take_loss_diagnostics(self):
        values = self._loss_diagnostics
        self._loss_diagnostics = None
        return values if values is not None else torch.zeros((1,7),device=self.device,dtype=torch.float64)
