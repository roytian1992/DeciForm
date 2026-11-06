"""Direct log+spherical Score objective on unchanged complete score grids.

Numeric labels retain the specialist's adjacent-grid interpolation. This is the
expected categorical proper score, with uniform-gradient K normalization; it
does not add an ordinal distance term or change decoding.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from score_specialist import support, interpolation, data_loss, SpecialistObjective

@dataclass(frozen=True)
class ScoreProperObjective:
    kind: str = 'direct_log_spherical'
    alpha: float = .75
    ordinal_scale: float = 3.0383850790577127

    def __post_init__(self):
        if self.kind not in ('direct_log_spherical','cdf_reference') or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('Expected direct log+spherical with finite nonnegative alpha')
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale<=0:
            raise ValueError('Positive finite baseline ordinal scale required')

    def inventory(self):
        return asdict(self)

def proper_score_loss(logits, targets, alpha=.75):
    if logits.ndim != 1 or len(logits) < 2 or targets.shape != logits.shape:
        raise ValueError('Complete unpadded logits and same-shape target required')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Finite nonnegative alpha required')
    z = logits if logits.dtype == torch.float64 else logits.float()
    q = targets.to(dtype=z.dtype, device=z.device)
    logp = F.log_softmax(z, dim=0)
    # A convex average over one-hot scores preserves fractional labels.
    spherical = -(q * torch.expm1(logp - .5*torch.logsumexp(2*logp, dim=0))).sum()
    return (-(q*logp).sum() + alpha*math.sqrt(len(z))*spherical)/(1+alpha)

class ScoreProperLossMixin:
    def losses(self, rows, encodings):
        if not rows or any(r['task']!='score' for r in rows):
            raise ValueError('This study only replaces the Score objective')
        zs=self.scores(rows, encodings)
        if len(zs)!=len(rows):
            raise ValueError('Model row count mismatch')
        targets=[]
        for row,z in zip(rows,zs):
            values=support(row)
            if len(values)!=len(z):
                raise ValueError('Every original legal score must be present')
            targets.append(z.new_tensor(interpolation(values,row['label']),dtype=torch.float32))
        with torch.autocast(zs[0].device.type,enabled=False):
            if self.distribution_objective.kind=='cdf_reference':
                reference=SpecialistObjective('soft_cdf',self.distribution_objective.ordinal_scale)
                losses=torch.stack([data_loss(z,support(r),r['label'],reference) for z,r in zip(zs,rows)])
            else:
                losses=torch.stack([proper_score_loss(z,q,self.distribution_objective.alpha) for z,q in zip(zs,targets)])
        if self.collect_loss_diagnostics:
            gradients=torch.autograd.grad(losses.sum(),zs,retain_graph=True)
            with torch.no_grad():
                if self._loss_diagnostics is None:
                    self._loss_diagnostics=torch.zeros((1,7),device=zs[0].device,dtype=torch.float64)
                for z,q,loss,gradient in zip(zs,targets,losses,gradients):
                    logp=z.float().log_softmax(0);p=logp.exp();ce=-(q*logp).sum()
                    self._loss_diagnostics += torch.stack((
                        z.new_ones(()),loss.detach(),ce,gradient.float().norm(),
                        (p-q).norm(),(q*p).sum(),z.new_tensor(len(z)))).double()[None]
        return losses + self.net.regularization()

    def take_loss_diagnostics(self):
        value=self._loss_diagnostics;self._loss_diagnostics=None
        return value if value is not None else torch.zeros((1,7),device=self.device,dtype=torch.float64)
