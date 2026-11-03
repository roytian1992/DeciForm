"""A4 architecture with a score-distribution objective and matched MAE decision."""
from models_v6_ablation import *
from models_v6_ablation import FormalModel as A4FormalModel
from score_distribution import DistributionObjective, task_data_loss, logit_diagnostics, decisions

A4_FLAGS = {'task_routing': True, 'quadratic': True, 'geometry': True,
            'auxiliary_losses': False}


class FormalModel(A4FormalModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None, distribution_objective=None):
        if ablation != A4_FLAGS:
            raise ValueError('Distribution study keeps the frozen A4 architecture')
        self.distribution_objective = DistributionObjective(**distribution_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation)
        self.collect_loss_diagnostics = False
        self._loss_diagnostics = None

    def losses(self, rs, es):
        logits = self.scores(rs, es)
        losses = []
        for row,z in zip(rs,logits):
            y,task = target(row),row['task']
            values = support(row) if task == 'score' else None
            losses.append(task_data_loss(z,y,task,self.distribution_objective,values))
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((len(TASKS),7),device=z.device,dtype=torch.float64)
                self._loss_diagnostics[TASKS.index(task)] += logit_diagnostics(
                    z,y,task,self.distribution_objective,values)
        return torch.stack(losses) + self.net.regularization()

    def take_loss_diagnostics(self):
        values = self._loss_diagnostics
        self._loss_diagnostics = None
        return values if values is not None else torch.zeros((len(TASKS),7),device=self.device,dtype=torch.float64)

    @torch.no_grad()
    def predict(self, rs, es, max_new_tokens):
        zs = self.scores(rs,es)
        result = []
        for row,z in zip(rs,zs):
            p = z.float().softmax(-1).cpu()
            assert torch.isfinite(p).all()
            pred = {'prediction_index':int(p.argmax()),'valid':True,'probabilities':p.tolist()}
            if row['task'] == 'score':
                ds = decisions(support(row),p.tolist())
                pred.update(prediction_index=ds['indices']['median'],score_decision_rule='median',
                            argmax_prediction_index=ds['indices']['argmax'],score_mean=ds['values']['mean'],
                            mean_nearest_prediction_index=ds['indices']['mean_nearest'])
            result.append(pred)
        self.clear()
        return result
