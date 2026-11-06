"""Small matched prefix CE/CDF + normalized predictive-mean MSE.

No added parameters and no numeric-sequence supervision. Keep the completed
control's causal forward/packing to isolate the regression gradient. Its
teacher-forced suffix has zero direct loss weight; the prefix is audited to
be independent of gold/counterfactual suffixes. Inference is prompt-only.
"""
from dataclasses import asdict, dataclass
import math
import torch
from models_numeric_trie_pilot import FormalModel as PilotModel
from numeric_trie import TrieObjective, components
from score_specialist import interpolation, support


@dataclass(frozen=True)
class R1Objective:
    kind: str = 'pilot_trie_ce_cdf_r1'
    ordinal_scale: float = 3.0383850790577127
    cdf_weight: float = .5
    mean_weight: float = 1.
    token_weight: float = 0.

    def __post_init__(self):
        if self.kind != 'pilot_trie_ce_cdf_r1' or self.token_weight != 0.:
            raise ValueError('R1 has no numeric-sequence supervision')
        if not math.isfinite(self.mean_weight) or self.mean_weight < 0:
            raise ValueError('Finite nonnegative mean weight required')
        self.distribution()

    def distribution(self):
        return TrieObjective('trie_ce_cdf', self.ordinal_scale, self.cdf_weight)

    def inventory(self):
        return asdict(self)


def mean_regression(logits, values, label):
    if logits.ndim != 1 or len(values) != len(logits) or len(values) < 2:
        raise ValueError('One logit per native support point required')
    if any(not math.isfinite(v) for v in values) or any(a >= b for a,b in zip(values,values[1:])):
        raise ValueError('Finite strictly increasing support required')
    if not math.isfinite(label) or not values[0] <= label <= values[-1]:
        raise ValueError('Finite label within native support required')
    z = logits if logits.dtype == torch.float64 else logits.float()
    # Normalize Python values first, preserving tiny ranges at large offsets.
    span = values[-1]-values[0]
    t = z.new_tensor([(v-values[0])/span for v in values])
    target = (float(label)-values[0])/span
    return ((z.softmax(0)*t).sum()-target).square()


def r1_components(logits, values, label, objective):
    base, ce, cdf = components(logits,values,label,objective.distribution())
    regression = mean_regression(logits,values,label)
    return base+objective.mean_weight*regression, ce, cdf, regression


def diagnostics(logits, values, label, objective):
    with torch.enable_grad():
        z=logits.detach().float().requires_grad_(True)
        total,ce,_,_=r1_components(z,values,label,objective)
        grad=torch.autograd.grad(total,z,retain_graph=True)[0]
        reference_grad=torch.autograd.grad(ce,z)[0]
    p=z.detach().softmax(0);q=p.new_tensor(interpolation(values,label))
    return torch.stack((p.new_ones(()),total.detach(),ce.detach(),grad.norm(),
                        reference_grad.norm(),(p*q).sum(),p.new_tensor(len(p)))).double()


class FormalModel(PilotModel):
    def __init__(self,*args,distribution_objective=None,**kwargs):
        obj=R1Objective(**distribution_objective)
        super().__init__(*args,distribution_objective={
            'kind':'pilot_trie_ce_cdf','ordinal_scale':obj.ordinal_scale,
            'cdf_weight':obj.cdf_weight,'token_weight':0.},**kwargs)
        self.distribution_objective=obj
        self.ablation_inventory.update(score_research_architecture='small_numeric_trie_r1',
                                       mean_weight=obj.mean_weight,token_weight=0.)

    def losses(self,rows,encodings):
        scores,_=self.joint_scores_and_token_loss(rows,encodings)
        losses=[]
        for row,z in zip(rows,scores):
            total,ce,cdf,regression=r1_components(z,support(row),row['label'],self.distribution_objective)
            losses.append(total)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics=torch.zeros((1,7),device=z.device,dtype=torch.float64)
                    self.objective_component_sums=torch.zeros(4,device=z.device,dtype=torch.float64)
                self._loss_diagnostics[0]+=diagnostics(z,support(row),row['label'],self.distribution_objective)
                self.objective_component_sums+=torch.stack((z.new_ones(()),ce.detach(),cdf.detach(),regression.detach())).double()
        return torch.stack(losses)+self.net.regularization()
