"""A bounded small experiment: prefix soft CE plus normalized CRPS.

Use the identical prefix architecture and computational path as the completed
CE/CDF control. No learned extra head, regression, or numeric token supervision.
CRPS uses full native spacings and the existing interpolated fractional target.
The omitted target-only interpolation constant has no parameter gradient.
"""
from dataclasses import asdict, dataclass
import math
import torch
from models_numeric_trie_pilot import FormalModel as PilotModel
from score_specialist import interpolation, support, SpecialistObjective, data_loss


@dataclass(frozen=True)
class CECRPSObjective:
    kind: str = 'pilot_trie_ce_crps'
    ordinal_scale: float = 3.0383850790577127
    crps_weight: float = .5
    token_weight: float = 0.
    mean_weight: float = 0.

    def __post_init__(self):
        if self.kind != 'pilot_trie_ce_crps' or self.token_weight != 0. or self.mean_weight != 0.:
            raise ValueError('Only CE and CRPS are enabled')
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale <= 0:
            raise ValueError('Positive finite scale required')
        if not math.isfinite(self.crps_weight) or not 0 <= self.crps_weight <= 1:
            raise ValueError('Mixture weight must lie in [0,1]')

    def inventory(self):
        return asdict(self)


def normalized_crps(logits, values, label):
    if logits.ndim!=1 or len(values)!=len(logits) or len(values)<2:
        raise ValueError('One logit per full native support point')
    if any(not math.isfinite(v) for v in values) or any(a>=b for a,b in zip(values,values[1:])):
        raise ValueError('Finite strictly increasing support required')
    if not math.isfinite(label) or not values[0]<=label<=values[-1]:
        raise ValueError('Finite label within native range required')
    z=logits if logits.dtype==torch.float64 else logits.float()
    span=values[-1]-values[0]
    t=z.new_tensor([(v-values[0])/span for v in values])
    q=z.new_tensor(interpolation(values,label))
    f=z.softmax(0).cumsum(0)[:-1]
    return (t.diff()*(f-q.cumsum(0)[:-1]).square()).sum()


def loss_components(logits,values,label,obj):
    ce=data_loss(logits,values,label,SpecialistObjective('soft_ce'))
    crps=normalized_crps(logits,values,label)
    scaled=2*obj.ordinal_scale*crps
    total=(1-obj.crps_weight)*ce+obj.crps_weight*scaled
    return total,ce,crps,scaled


def diagnostics(logits,values,label,obj):
    with torch.enable_grad():
        z=logits.detach().float().requires_grad_(True)
        total,ce,_,_=loss_components(z,values,label,obj)
        gradient=torch.autograd.grad(total,z,retain_graph=True)[0]
        ce_gradient=torch.autograd.grad(ce,z)[0]
    p=z.detach().softmax(0);q=p.new_tensor(interpolation(values,label))
    return torch.stack((p.new_ones(()),total.detach(),ce.detach(),gradient.norm(),
                        ce_gradient.norm(),(p*q).sum(),p.new_tensor(len(p)))).double()


class FormalModel(PilotModel):
    def __init__(self,*args,distribution_objective=None,**kwargs):
        obj=CECRPSObjective(**distribution_objective)
        super().__init__(*args,distribution_objective={
            'kind':'pilot_trie_ce_cdf','ordinal_scale':obj.ordinal_scale,
            'cdf_weight':.5,'token_weight':0.},**kwargs)
        self.distribution_objective=obj
        self.ablation_inventory.update(score_research_architecture='small_numeric_trie_cecrps',
                                       crps_weight=obj.crps_weight,mean_weight=0.,token_weight=0.)

    def losses(self,rows,encodings):
        scores,_=self.joint_scores_and_token_loss(rows,encodings)
        losses=[]
        for row,z in zip(rows,scores):
            total,ce,crps,scaled=loss_components(z,support(row),row['label'],self.distribution_objective)
            losses.append(total)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics=torch.zeros((1,7),device=z.device,dtype=torch.float64)
                    self.objective_component_sums=torch.zeros(4,device=z.device,dtype=torch.float64)
                self._loss_diagnostics[0]+=diagnostics(z,support(row),row['label'],self.distribution_objective)
                self.objective_component_sums+=torch.stack((z.new_ones(()),ce.detach(),crps.detach(),scaled.detach())).double()
        return torch.stack(losses)+self.net.regularization()
