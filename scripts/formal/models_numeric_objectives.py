"""Frozen numeric-D2 architecture; only the Score objective changes."""
from models_score_research import NumericD2
from models_score_distribution import A4_FLAGS
from numeric_score_objectives import NumericScoreObjective, loss_components, logit_diagnostics
from score_specialist import support
import torch


class FormalModel(NumericD2):
    def __init__(self, arm, base, checkpointing=True, ablation=None, distribution_objective=None):
        objective = NumericScoreObjective(**distribution_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation,
                         distribution_objective={'kind': 'soft_cdf', 'ordinal_scale': objective.ordinal_scale})
        # The temporary parent objective constructs exactly the frozen model;
        # all actual training/diagnostics use the requested objective below.
        self.distribution_objective = objective
        self.objective_component_sums = None

    def losses(self, rows, encodings):
        logits = self.scores(rows, encodings)
        values = []
        for row, z in zip(rows, logits):
            parts = loss_components(z, support(row), row['label'], self.distribution_objective)
            values.append(parts['total'])
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((1, 7), device=z.device, dtype=torch.float64)
                self._loss_diagnostics[0] += logit_diagnostics(z, support(row), row['label'], self.distribution_objective)
                if self.objective_component_sums is None:
                    self.objective_component_sums = torch.zeros(4, device=z.device, dtype=torch.float64)
                self.objective_component_sums += torch.stack((z.new_ones(()),
                    parts['cdf_log'].detach(), parts['normalized_mean_mse'].detach(),
                    parts['normalized_crps'].detach())).double()
        return torch.stack(values) + self.net.regularization()

    def take_objective_components(self):
        result = self.objective_component_sums
        self.objective_component_sums = None
        return result if result is not None else torch.zeros(4, device=self.device, dtype=torch.float64)
