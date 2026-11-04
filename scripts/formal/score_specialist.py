"""Isolated continuous Score data contract, objectives and native-value metrics."""
from bisect import bisect_left
from dataclasses import asdict, dataclass
import math

import torch

from common import PROJECT, BASE, SEED, atomic, sha, cost
from score_distribution import decisions

PROTOCOL = 'score_specialist_continuous_20261104_v1'
TASKS = ('score',)


def support(row):
    values = row['score_support']
    if row['task'] != 'score' or len(values) < 2 or any(not math.isfinite(v) for v in values):
        raise ValueError('Expected a complete finite Score support')
    if any(a >= b for a,b in zip(values,values[1:])):
        raise ValueError('Score support must be strictly increasing')
    return values


def interpolation(values, label):
    y=float(label)
    if not math.isfinite(y) or not values[0] <= y <= values[-1]:
        raise ValueError('Raw target outside the declared score range')
    j=bisect_left(values,y)
    q=[0.] * len(values)
    if values[j] == y:
        q[j]=1.
    else:
        q[j-1]=(values[j]-y)/(values[j]-values[j-1]);q[j]=1-q[j-1]
    return q


def encode_score(row, tokenizer):
    from kev.model import encode, user_tokens
    values=support(row)
    interpolation(values,row['label'])  # validate; no rounded target is encoded
    options=[f'level_{v:g}' for v in values]
    rec={'state':row['state']['text'],'questions':[
        {'instr':row['question']['instructions'],'options':options,'label':None}]}
    n=len(user_tokens(tokenizer,rec['state']))+1
    branch=len(user_tokens(tokenizer,rec['questions'][0]['instr']))+sum(
        len(user_tokens(tokenizer,s))+2 for s in options)+2
    result=encode(tokenizer,rec,max_state=n,max_branch=n+branch+8,strict=True)
    result['tasks']=['score']
    assert result['labels']==[None] and not result['state_truncated']
    assert len(result['opt_idx'][0])==len(values)
    return result


@dataclass(frozen=True)
class SpecialistObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127

    def __post_init__(self):
        if self.kind not in ('soft_ce','soft_cdf') or not math.isfinite(self.ordinal_scale) or self.ordinal_scale<=0:
            raise ValueError('Invalid specialist objective')

    def inventory(self):
        return asdict(self)


def data_loss(logits, values, label, objective):
    if logits.ndim!=1 or len(logits)!=len(values):
        raise ValueError('Logits must cover the entire numeric grid')
    z=logits if logits.dtype==torch.float64 else logits.float()
    q=z.new_tensor(interpolation(values,label))
    logz=z.logsumexp(0)
    if objective.kind=='soft_ce':
        return -(q*(z-logz)).sum()
    left=z.logcumsumexp(0)[:-1]
    right=z.flip(0).logcumsumexp(0).flip(0)[1:]
    qcdf=q.cumsum(0)[:-1]
    widths=z.new_tensor([b-a for a,b in zip(values,values[1:])])/(values[-1]-values[0])
    return objective.ordinal_scale*(widths*(qcdf*(logz-left)+(1-qcdf)*(logz-right))).sum()


def logit_diagnostics(logits, values, label, objective):
    with torch.enable_grad():
        z=logits.detach().float().requires_grad_(True)
        value=data_loss(z,values,label,objective)
        ce=data_loss(z,values,label,SpecialistObjective('soft_ce'))
        grad=torch.autograd.grad(value,z,retain_graph=True)[0]
        ce_grad=torch.autograd.grad(ce,z)[0]
    p=z.detach().softmax(0);q=p.new_tensor(interpolation(values,label))
    return torch.stack((p.new_ones(()),value.detach(),ce.detach(),grad.norm(),ce_grad.norm(),
                        (p*q).sum(),p.new_tensor(len(p)))).double()


def prediction_row(row,prediction):
    values=support(row); y=float(row['label'])
    if not prediction['valid'] or len(prediction['probabilities'])!=len(values):
        raise ValueError('Missing/invalid score prediction; do not omit it')
    ds=decisions(values,prediction['probabilities'])
    if ds['indices']['median']!=prediction['prediction_index']:
        raise ValueError('Headline decision must remain the native grid median')
    return {**prediction,'id':row['id'],'source':row['source_family'],'group_id':row['group_id'],
        'task':'score','label':y,'support':values,'k':len(values),'score_decision_rule':'median',
        'prediction_value':ds['values']['median'],'score_decisions':ds['values'],
        'mae':abs(ds['values']['median']-y),'quantization_floor':min(abs(v-y) for v in values),
        'rater_std':row.get('rater_std'),'original_label_preserved':True}


def spearman(labels, predictions):
    """Pearson correlation of average ranks; constant inputs are undefined."""
    import numpy as np
    def ranks(values):
        values=np.asarray(values,dtype=np.float64)
        order=np.argsort(values,kind='stable');sorted_values=values[order]
        starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
        ends=np.r_[starts[1:],len(values)]
        ranked=np.empty(len(values),dtype=np.float64)
        for start,end in zip(starts,ends):ranked[order[start:end]]=(start+end-1)/2
        return ranked-ranked.mean()
    if len(labels)!=len(predictions) or not labels:
        raise ValueError('Correlation requires nonempty paired observations')
    x=ranks(labels);y=ranks(predictions)
    denom=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.clip(np.dot(x,y)/denom,-1,1)) if denom else None


def metrics(rows):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:
        groups['all'].append(r);groups['task/score'].append(r);groups['source/'+r['source']].append(r)
    result={}
    for group,items in groups.items():
        n=len(items);errs=[r['mae'] for r in items]
        labels=[r['label'] for r in items];preds=[r['prediction_value'] for r in items]
        rho=spearman(labels,preds)
        result[group]={'rows':n,'mae':sum(errs)/n,'nmae':sum(r['mae']/(r['support'][-1]-r['support'][0]) for r in items)/n,
            'rmse':math.sqrt(sum(e*e for e in errs)/n),'spearman':rho,
            'quantization_floor_mae':sum(r['quantization_floor'] for r in items)/n,
            'rules':{rule:{'mae':sum(abs(r['score_decisions'][rule]-r['label']) for r in items)/n}
                     for rule in ('median','argmax','mean','mean_nearest')}}
    source=[v['nmae'] for k,v in result.items() if k.startswith('source/')]
    result['macro_source_nmae']=sum(source)/len(source)
    return result


def score_rule_report(rows,split):
    report=metrics(rows)
    return {'evaluation_protocol':PROTOCOL,'split':split,'rows':len(rows),
            'primary_rule':'median','primary_metric':'MAE against original real-valued label',
            'primary_mae':report['task/score']['mae'],'rules':report['task/score']['rules'],
            'by_source':{k[7:]:v for k,v in report.items() if k.startswith('source/')},
            'macro_source_nmae':report['macro_source_nmae'],'complete_output_grid':True,
            'continuous_targets_not_rounded':True}
