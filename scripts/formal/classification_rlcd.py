"""Small classification-only loss study on the frozen independent D2 model.

RLCD uses the pinned Laya reward and microbatch advantage normalization.
Noise draws are RNG-isolated so adding the objective does not advance dropout
RNG in the shared backbone. No additional forward or learned module is added.
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from common import PROJECT, target
from independent_candidates import split_candidates
from models_independent import FormalModel as IndependentModel
from score_distribution import task_data_loss

VENDOR = PROJECT / 'runs/formal_laya_20261106_v1/vendor/laya_release'
sys.path.insert(0, str(VENDOR))
from laya.common import QTYPES, proper_reward

TASKS = ('choice', 'noul')
PROTOCOL = 'classification_rlcd_pilot_20261106_v1'


def rlcd_rows(logits, target_distribution, mask, qtype, sigma=.4, samples=4):
    """Row mean and gradients equal native Laya under the same RNG state."""
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    noisy = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target_distribution.unsqueeze(0), qtype, fmask,
                               w_sph=.75, w_rps=1.)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    log_density = -(((noisy-logits.unsqueeze(0))**2)*fmask).sum(-1)/(2*sigma**2)
    ce = -(target_distribution * F.log_softmax(logits.masked_fill(~mask, -1e4), -1) * mask).sum(-1)
    return ce - (advantage * log_density).mean(0)


def classification_losses(zs, labels, tasks, rlcd_tasks, objective, isolate_rng=True):
    if not set(tasks).issubset(TASKS) or not set(rlcd_tasks).issubset(TASKS):
        raise ValueError('Choice and Noul only; Score is excluded')
    losses = [task_data_loss(z, y, t, objective) for z,y,t in zip(zs,labels,tasks)]
    selected = [i for i,t in enumerate(tasks) if t in rlcd_tasks]
    if selected:
        maximum = max(len(zs[i]) for i in selected)
        z = torch.stack([F.pad(zs[i].float(), (0, maximum-len(zs[i]))) for i in selected])
        mask = torch.arange(maximum, device=z.device)[None] < z.new_tensor([len(zs[i]) for i in selected], dtype=torch.long)[:,None]
        targets = F.one_hot(torch.tensor([labels[i] for i in selected],device=z.device), maximum).float()
        qtype = torch.tensor([QTYPES[tasks[i]] for i in selected],device=z.device)
        devices = [z.device.index] if z.is_cuda else []
        with torch.random.fork_rng(devices=devices, enabled=isolate_rng):
            with torch.autocast(z.device.type, enabled=False):
                replaced = rlcd_rows(z, targets, mask, qtype)
        for j,i in enumerate(selected):
            losses[i] = replaced[j]
    return torch.stack(losses)


class FormalModel(IndependentModel):
    def __init__(self, *args, rlcd_tasks=(), **kwargs):
        if not set(rlcd_tasks).issubset(TASKS):
            raise ValueError('Invalid RLCD task')
        self.rlcd_tasks = tuple(rlcd_tasks)
        super().__init__(*args, **kwargs)

    def scores(self, rows, encodings):
        return super().scores(rows, [split_candidates(e) for e in encodings])

    def losses(self, rows, encodings):
        zs = self.scores(rows, encodings)
        labels, tasks = [target(r) for r in rows], [r['task'] for r in rows]
        losses = classification_losses(zs, labels, tasks, self.rlcd_tasks, self.distribution_objective)
        if self.collect_loss_diagnostics:
            gradients = torch.autograd.grad(losses.sum(), zs, retain_graph=True)
            with torch.no_grad():
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((2,7), device=self.device, dtype=torch.float64)
                for z,y,task,loss,grad in zip(zs,labels,tasks,losses,gradients):
                    p=z.float().softmax(0); ce=-p[y].log(); reference=p.clone();reference[y]-=1
                    self._loss_diagnostics[TASKS.index(task)] += torch.stack((
                        z.new_ones(()),loss.detach(),ce,grad.float().norm(),reference.norm(),p[y],z.new_tensor(len(z)))).double()
        return losses + self.net.regularization()

    def take_loss_diagnostics(self):
        result = self._loss_diagnostics
        self._loss_diagnostics = None
        return result if result is not None else torch.zeros((2,7),device=self.device,dtype=torch.float64)
