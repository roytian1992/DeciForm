"""Normalized log+spherical loss and scaled native RLCD, without model edits."""
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from common import target
from classification_rlcd import FormalModel as RLCDModel, TASKS, PROTOCOL, classification_losses
from score_distribution import task_data_loss

@dataclass(frozen=True)
class ClassificationObjective:
    kind: str = 'proper'
    alpha: float = .75
    increment_weight: float = .1

    def __post_init__(self):
        if self.kind not in ('proper', 'scaled_rlcd'):
            raise ValueError('Unsupported classification objective')
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        if not math.isfinite(self.increment_weight) or not 0 <= self.increment_weight <= 1:
            raise ValueError('increment_weight must be in [0,1]')

    def inventory(self):
        return asdict(self)

def proper_loss(z, y, alpha=.75):
    """Stable loss: train in FP32, retain FP64 for mathematical checks."""
    if z.ndim != 1 or len(z) < 2 or not 0 <= y < len(z):
        raise ValueError('Expected a categorical logit vector and valid target')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    z = z if z.dtype == torch.float64 else z.float()
    logp = F.log_softmax(z, dim=0)
    log_spherical = logp[y] - .5 * torch.logsumexp(2 * logp, dim=0)
    spherical_loss = -torch.expm1(log_spherical)
    return (-logp[y] + alpha * math.sqrt(len(z)) * spherical_loss) / (1 + alpha)

def objective_losses(zs, labels, tasks, distribution_objective, objective):
    if not (len(zs) == len(labels) == len(tasks)) or not zs:
        raise ValueError('Mismatched or empty rows')
    if not set(tasks).issubset(TASKS):
        raise ValueError('Choice/Noul only; Score excluded')
    if objective.kind == 'proper':
        return torch.stack([proper_loss(z, y, objective.alpha) for z,y in zip(zs, labels)])
    base = torch.stack([task_data_loss(z,y,t,distribution_objective) for z,y,t in zip(zs,labels,tasks)])
    if objective.increment_weight == 0:
        return base
    native = classification_losses(zs, labels, tasks, TASKS, distribution_objective)
    if objective.increment_weight == 1:
        return native
    return base + objective.increment_weight * (native - base)

class FormalModel(RLCDModel):
    def __init__(self, *args, classification_objective, **kwargs):
        self.classification_objective = ClassificationObjective(**classification_objective)
        super().__init__(*args, **kwargs)
        if tuple(self.rlcd_tasks) != TASKS:
            raise ValueError('This matched pilot applies objectives to both tasks')

    def losses(self, rows, encodings):
        zs = self.scores(rows, encodings)
        labels, tasks = [target(r) for r in rows], [r['task'] for r in rows]
        with torch.autocast(zs[0].device.type, enabled=False):
            losses = objective_losses(zs, labels, tasks, self.distribution_objective, self.classification_objective)
        if self.collect_loss_diagnostics:
            gradients = torch.autograd.grad(losses.sum(), zs, retain_graph=True)
            with torch.no_grad():
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((2,7), device=self.device, dtype=torch.float64)
                for z,y,task,loss,grad in zip(zs,labels,tasks,losses,gradients):
                    logp=z.float().log_softmax(0); p=logp.exp(); reference=p.clone(); reference[y]-=1
                    self._loss_diagnostics[TASKS.index(task)] += torch.stack((
                        z.new_ones(()),loss.detach(),-logp[y],grad.float().norm(),reference.norm(),p[y],z.new_tensor(len(z)))).double()
        return losses + self.net.regularization()
