"""Frozen v6 architecture and A4 non-choice losses; choice loss is the treatment."""
from models_v6_ablation import *
from models_v6_ablation import FormalModel as A4FormalModel
from choice_losses import ChoiceObjective, choice_loss, logit_diagnostics


A4_FLAGS = {'task_routing': True, 'quadratic': True, 'geometry': True,
            'auxiliary_losses': False}


class FormalModel(A4FormalModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None, choice_objective=None):
        if ablation != A4_FLAGS:
            raise ValueError('Loss experiments retain the full A4 architecture and score CE')
        self.choice_objective = ChoiceObjective(**choice_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation)
        self.collect_loss_diagnostics = False
        self._loss_diagnostics = None

    def losses(self, rs, es):
        logits = self.scores(rs, es)
        losses = []
        for row, z in zip(rs, logits):
            y, task = target(row), row['task']
            if task == 'choice':
                value = choice_loss(z, y, self.choice_objective)
            elif task == 'noul':
                # Exactly the existing A4 Bernoulli NLL path.
                value = task_loss(z, y, task)
            elif task == 'score':
                value = F.cross_entropy(z.float()[None], torch.tensor([y], device=z.device))
            else:
                raise ValueError('Unsupported task')
            losses.append(value)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((len(TASKS), 7), device=z.device,
                                                          dtype=torch.float64)
                self._loss_diagnostics[TASKS.index(task)] += logit_diagnostics(
                    z, y, self.choice_objective, choice=task == 'choice')
        # Preserve A4 per-record aggregation and structural regularization.
        return torch.stack(losses) + self.net.regularization()

    def take_loss_diagnostics(self):
        values = self._loss_diagnostics
        self._loss_diagnostics = None
        if values is None:
            values = torch.zeros((len(TASKS), 7), device=self.device, dtype=torch.float64)
        return values
