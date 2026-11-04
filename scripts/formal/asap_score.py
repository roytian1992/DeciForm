"""Native ASAP score supports and supplementary, prompt-specific metrics."""
import collections
import math

from score_specialist import *
from score_specialist import metrics as base_metrics
from score_specialist import prediction_row as base_prediction_row

PROTOCOL = 'score_specialist_asap_fold0_20261104_v1'
RANGES = {1:(2,12),2:(1,6),3:(0,3),4:(0,3),5:(0,4),6:(0,4),7:(0,30),8:(0,60)}


def quadratic_kappa(labels, predictions, values):
    """Quadratic weighted kappa on a single prompt's native integer support."""
    import numpy as np
    if not labels or len(labels)!=len(predictions):
        raise ValueError('Expected paired nonempty observations')
    if values != list(range(values[0],values[-1]+1)):
        raise ValueError('QWK requires this experiment\'s contiguous native scale')
    ix={v:i for i,v in enumerate(values)};k=len(values);n=len(labels)
    observed=np.zeros((k,k),dtype=np.float64)
    for y,p in zip(labels,predictions):observed[ix[y],ix[p]]+=1
    weight=(np.arange(k)[:,None]-np.arange(k)[None,:])**2/(k-1)**2
    expected=observed.sum(1)[:,None]*observed.sum(0)[None,:]/n
    denominator=float((weight*expected).sum())
    return 1-float((weight*observed).sum())/denominator if denominator else None


def prediction_row(row,prediction):
    values=support(row)
    lo,hi=RANGES[row['essay_set']]
    if type(row['label']) is not int or values!=list(range(lo,hi+1)):
        raise ValueError('ASAP must preserve native domain1_score and complete support')
    result=base_prediction_row(row,prediction)
    y=values.index(row['label']);p=result['probabilities'];cumulative=0.;rps=0.
    for k,pk in enumerate(p[:-1]):
        cumulative+=pk;rps+=(cumulative-float(y<=k))**2/(len(p)-1)
    return {**result,'label':row['label'],'essay_set':row['essay_set'],
            'nll':-math.log(max(p[y],1e-300)),'normalized_rps':rps,
            'original_native_support':True}


def metrics(rows):
    report=base_metrics(rows);groups=collections.defaultdict(list)
    for r in rows:groups['source/'+r['source']].append(r)
    for key,items in groups.items():
        values=items[0]['support'];assert all(r['support']==values for r in items)
        report[key].update(qwk=quadratic_kappa([r['label'] for r in items],
            [r['prediction_value'] for r in items],values),
            nll=sum(r['nll'] for r in items)/len(items),
            normalized_rps=sum(r['normalized_rps'] for r in items)/len(items),
            boundary_target_rows=sum(r['label'] in (values[0],values[-1]) for r in items),
            large_error_fraction=sum(r['mae']/(values[-1]-values[0])>=.2 for r in items)/len(items))
    report['macro_prompt_nmae']=report['macro_source_nmae']
    return report


def score_rule_report(rows,split):
    report=metrics(rows)
    return {'evaluation_protocol':PROTOCOL,'split':split,'rows':len(rows),
            'primary_rule':'median','primary_metric':'native MAE per prompt; equal-prompt NMAE',
            'primary_mae':report['task/score']['mae'],'pooled_raw_mae_is_not_cross_prompt_headline':True,
            'rules':report['task/score']['rules'],
            'by_source':{k[7:]:v for k,v in report.items() if k.startswith('source/')},
            'macro_source_nmae':report['macro_source_nmae'],
            'macro_prompt_nmae':report['macro_prompt_nmae'],
            'complete_output_grid':True,'native_integer_labels':True}
