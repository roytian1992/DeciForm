"""Unchanged D2 architecture, separate continuous Score supervision."""
from models_v6_ablation import FormalModel as Architecture
from models_score_distribution import A4_FLAGS
from score_specialist import *


class FormalModel(Architecture):
    def __init__(self,arm,base,checkpointing=True,ablation=None,distribution_objective=None):
        if ablation!=A4_FLAGS:
            raise ValueError('Keep all D2 architecture mechanisms')
        super().__init__(arm,base,checkpointing=checkpointing,ablation=ablation)
        self.distribution_objective=SpecialistObjective(**distribution_objective)
        self.collect_loss_diagnostics=False
        self._loss_diagnostics=None

    def scores(self,rs,es):
        if any(r['task']!='score' or e['tasks']!=['score'] or e['labels']!=[None] for r,e in zip(rs,es)):
            raise ValueError('Continuous score labels live in rows, never integer transport labels')
        return [v[0] for v in self.net.forward_batch(es,numeric_supports=[support(r) for r in rs])]

    def losses(self,rs,es):
        zs=self.scores(rs,es);losses=[]
        for r,z in zip(rs,zs):
            losses.append(data_loss(z,support(r),r['label'],self.distribution_objective))
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics=torch.zeros((1,7),device=z.device,dtype=torch.float64)
                self._loss_diagnostics[0]+=logit_diagnostics(z,support(r),r['label'],self.distribution_objective)
        return torch.stack(losses)+self.net.regularization()

    def take_loss_diagnostics(self):
        value=self._loss_diagnostics;self._loss_diagnostics=None
        return value if value is not None else torch.zeros((1,7),device=self.device,dtype=torch.float64)

    @torch.no_grad()
    def predict(self,rs,es,max_new_tokens=0):
        out=[]
        for r,z in zip(rs,self.scores(rs,es)):
            p=z.float().softmax(0).cpu().tolist();ds=decisions(support(r),p)
            out.append({'valid':True,'prediction_index':ds['indices']['median'],'probabilities':p,
                        'score_decision_rule':'median','score_mean':ds['values']['mean']})
        self.clear();return out
