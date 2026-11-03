"""Frozen A4 architecture, isolated noul/score objective treatment."""
from models_v6_ablation import *
from models_v6_ablation import FormalModel as A4FormalModel
from typed_losses import TypedObjective, task_data_loss, logit_diagnostics


A4_FLAGS = {'task_routing': True, 'quadratic': True, 'geometry': True,
            'auxiliary_losses': False}


class FormalModel(A4FormalModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None, typed_objective=None):
        if ablation != A4_FLAGS:
            raise ValueError('Typed-loss experiments retain the entire A4 architecture')
        self.typed_objective = TypedObjective(**typed_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation)
        self.collect_loss_diagnostics = False
        self._loss_diagnostics = None

    def losses(self, rs, es):
        logits = self.scores(rs, es)
        losses = []
        for row, z in zip(rs, logits):
            y, task = target(row), row['task']
            values = support(row) if task == 'score' else None
            losses.append(task_data_loss(z, y, task, self.typed_objective, values))
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((len(TASKS), 7), device=z.device,
                                                          dtype=torch.float64)
                self._loss_diagnostics[TASKS.index(task)] += logit_diagnostics(
                    z, y, task, self.typed_objective, values)
        return torch.stack(losses) + self.net.regularization()

    def take_loss_diagnostics(self):
        values = self._loss_diagnostics
        self._loss_diagnostics = None
        if values is None:
            values = torch.zeros((len(TASKS), 7), device=self.device, dtype=torch.float64)
        return values
