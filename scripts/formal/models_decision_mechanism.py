"""A4 with exactly one authorized choice/score mechanism; noul stays BCE."""
from models_v6_ablation import *
from models_v6_ablation import FormalModel as A4FormalModel, ablation_inventory
from decision_mechanisms import Mechanism, install_mechanism, native_prediction

A4_FLAGS = {'task_routing': True, 'quadratic': True, 'geometry': True,
            'auxiliary_losses': False}


class FormalModel(A4FormalModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None, mechanism=None):
        if ablation != A4_FLAGS:
            raise ValueError('Keep the entire A4 architecture enabled')
        self.mechanism = Mechanism(**mechanism)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation)
        install_mechanism(self.net.head, self.mechanism)
        self.ablation_inventory = ablation_inventory(self.net)
        self.collect_loss_diagnostics = False
        self._loss_diagnostics = None

    def losses(self, rs, es):
        logits = self.scores(rs, es)
        losses = []
        for i, (row, z) in enumerate(zip(rs, logits)):
            y, task = target(row), row['task']
            z = z.float()
            if task == 'noul':
                if len(z) != 2:
                    raise ValueError('Noul requires two outcomes')
                ce = F.binary_cross_entropy_with_logits(z[1]-z[0], z.new_tensor(float(y)))
            else:
                ce = F.cross_entropy(z[None], torch.tensor([y], device=z.device))
            absolute = z.new_zeros(())
            if task == 'score' and self.mechanism.kind == 'score_regression':
                coordinate = self.net.head.regression_coordinates[i]
                prediction = native_prediction(coordinate, support(row))
                absolute = (prediction - row['label']).abs()
            extra = self.mechanism.regression_weight * absolute
            value = ce + extra
            losses.append(value)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = z.new_zeros((len(TASKS), 7), dtype=torch.float64)
                with torch.no_grad():
                    self._loss_diagnostics[TASKS.index(task)] += torch.stack((
                        z.new_ones(()), value.detach(), ce.detach(), absolute.detach(),
                        extra.detach(), z.softmax(0)[y], z.new_tensor(len(z)))).double()
        return torch.stack(losses) + self.net.regularization()

    def take_loss_diagnostics(self):
        values = self._loss_diagnostics
        self._loss_diagnostics = None
        return values if values is not None else torch.zeros(
            (len(TASKS), 7), device=self.device, dtype=torch.float64)

    def clear(self):
        super().clear()
        self.net.head.regression_coordinates = {}

    @torch.no_grad()
    def predict(self, rs, es, max_new_tokens):
        logits = self.scores(rs, es)
        result = []
        for i, (row, z) in enumerate(zip(rs, logits)):
            p = z.float().softmax(-1).cpu()
            assert torch.isfinite(p).all()
            prediction = {'prediction_index': int(p.argmax()), 'valid': True,
                          'probabilities': p.tolist()}
            if row['task'] == 'score' and self.mechanism.kind == 'score_regression':
                numeric = native_prediction(self.net.head.regression_coordinates[i], support(row))
                if not torch.isfinite(numeric):
                    raise FloatingPointError('Nonfinite score regression')
                prediction['regression_prediction'] = float(numeric)
            result.append(prediction)
        self.clear()
        return result
