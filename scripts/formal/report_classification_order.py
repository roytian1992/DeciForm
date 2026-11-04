"""Current Choice/Noul report; preserve the superseded score/noul journals."""
import copy
import json
from pathlib import Path
from common import PROJECT, atomic, sha


def classification_only(report):
    result=copy.deepcopy(report)
    for field in ('groups','by_source'):
        result[field]={key:value for key,value in result.get(field,{}).items()
                       if value['task'] in ('choice','noul')}
        for value in result[field].values():
            value.pop('invalid_score_penalty',None)
            for stats in value['per_view'].values():
                stats.pop('mae',None)
            if value.get('order_metrics'):
                for key in ('mean_absolute_score_change_on_valid_pairs','valid_score_pairs'):
                    value['order_metrics'].pop(key,None)
    result['repeat_controls']=[r for r in result.get('repeat_controls',[]) if r['task'] in ('choice','noul')]
    result['evaluated_predictions']=sum(g['evaluated_predictions'] for g in result['groups'].values())
    return result


def main():
    previous=PROJECT/'runs/formal_answer_order_20261104_v1'
    current=PROJECT/'runs/formal_choice_order_20261104_v1'
    manifest=json.loads((previous/'experiment_manifest.json').read_text())
    models={};provenance={};pending=[]
    for name in manifest['models']:
        path=previous/name/'report.json'
        result=classification_only(json.loads(path.read_text()))
        provenance[str(path)]=sha(path)
        choice=current/name/'report.json'
        if choice.exists() and (current/name/'complete.json').exists():
            extra=classification_only(json.loads(choice.read_text()))
            if extra['checkpoint_sha256']!=result['checkpoint_sha256']:
                raise ValueError('Choice/Noul must use the same checkpoint')
            for field in ('groups','by_source'):
                assert not (result[field].keys() & extra[field].keys())
                result[field].update(extra[field])
            result['evaluated_predictions']+=extra['evaluated_predictions']
            result['repeat_controls']+=extra['repeat_controls']
            provenance[str(choice)]=sha(choice)
        else:
            pending.append(name)
        # Combined provenance is at top level; do not leave a misleading single
        # protocol hash or prediction-journal hash on a two-protocol report.
        for key in ('protocol_sha256','predictions_sha256','seconds','peak_memory_gib'):
            result.pop(key,None)
        models[name]=result
    report={'status':'partial' if pending else 'complete','tasks':['choice','noul'],
        'pending_choice_models':pending,'models':models,'source_reports_sha256':provenance,
        'notes':['Score flip rates are excluded by user correction; original files remain provenance.',
                 'Native and redundant instruction-list panels are different interventions and must stay separate.',
                 'Unsupported native Noul interfaces remain N/A; no zero-flip superiority claim.',
                 'Choice uses all 5000 rows and all native candidates; Noul reuses the completed fixed panel.']}
    output=previous/'classification_report.json'
    atomic(output,report)
    print(output,report['status'])


if __name__=='__main__':
    main()
