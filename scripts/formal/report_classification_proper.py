"""Report only completed matched pilot arms; never select on test or relaunch."""
import argparse
import collections
import datetime
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from common import PROJECT, atomic, sha

ROOT=PROJECT/'runs/pilot_classification_proper_20261106_v1'
BASELINE=PROJECT/'runs/pilot_classification_rlcd_20261106_v1/ce'
def arm_path(arm):return BASELINE if arm=='ce' else ROOT/arm
DOC=PROJECT/'docs/experiment_record_final_d2_20261105.md'
START='<!-- CLASSIFICATION_PROPER_PILOT_START -->'
END='<!-- CLASSIFICATION_PROPER_PILOT_END -->'


def predictions(arm,split):
    rows=[]
    for p in sorted((arm_path(arm)/'evaluations').glob(f'epoch_1.{split}.rank*.jsonl')):
        rows.extend(json.loads(line) for line in p.open())
    assert len(rows)==(1000 if split=='dev' else 2000)
    assert len({r['id'] for r in rows})==len(rows)
    return {r['id']:r for r in rows}


def paired(base,other,task):
    assert base.keys()==other.keys()
    keys=sorted(k for k,r in base.items() if r['task']==task)
    d=np.array([int(other[k]['correct'])-int(base[k]['correct']) for k in keys],dtype=float)
    for k in keys:
        assert (base[k]['label'],base[k]['support'])==(other[k]['label'],other[k]['support'])
    groups=collections.defaultdict(list)
    for i,k in enumerate(keys):groups[base[k]['source']].append(i)
    rng=np.random.default_rng(20261101);draws=np.zeros(2000)
    for indices in groups.values():
        v=d[indices];draws+=v[rng.integers(0,len(v),size=(2000,len(v)))].sum(1)
    draws/=len(d)
    return {'rows':len(d),'accuracy_delta':float(d.mean()),
            'paired_source_stratified_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),
            'gained_correct':int((d==1).sum()),'lost_correct':int((d==-1).sum()),
            'uncertainty':'conditional on these checkpoints and frozen sample; not cross-seed stability'}


def report():
    arms={};all_done=True;failed=False
    for arm in ['ce','rlcd_both','rlcd_scaled','proper']:
        p=arm_path(arm);status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
        entry={'status':status.get('phase','not_started')}
        if (p/'progress.json').exists():entry['progress']=json.loads((p/'progress.json').read_text())
        if status.get('phase')=='complete' and (p/'complete.json').exists():
            entry['complete']=json.loads((p/'complete.json').read_text())
            entry['metrics']={}
            for split in ['dev','test_in_domain']:
                f=p/'evaluations'/f'epoch_1.{split}.json';entry['metrics'][split]=json.loads(f.read_text())
                entry.setdefault('result_sha256',{})[str(f)]=sha(f)
        else:all_done=False
        failed |= entry['status']=='failed'
        arms[arm]=entry
    comparisons={}
    if 'metrics' in arms['ce']:
        for arm,task in [('rlcd_both',None),('rlcd_scaled',None),('proper',None)]:
            if 'metrics' in arms[arm]:
                comparisons[arm]={split:{t:paired(predictions('ce',split),predictions(arm,split),t) for t in ['choice','noul']} for split in ['dev','test_in_domain']}
    if 'metrics' in arms['rlcd_both']:
        for arm in ['rlcd_scaled','proper']:
            if 'metrics' in arms[arm]:
                comparisons[arm+'_vs_joint_native']={split:{t:paired(predictions('rlcd_both',split),predictions(arm,split),t) for t in ['choice','noul']} for split in ['dev','test_in_domain']}
    result={'updated':datetime.datetime.now().astimezone().isoformat(),'arms':arms,'paired_comparisons':comparisons,
            'references':{'ce':str(BASELINE)},'complete':all_done,'has_failure':failed,'exploratory':True,'test_selected':False}
    atomic(ROOT/'results_summary.json',result)
    lines=[START,'','## Choice/Noul 适当评分与RLCD尺度对照','',
        '独立探索：10,000训练＋1,000开发＋2,000测试，均为Choice/Noul各半；从原始Base初始化，1轮，157步。',
        '固定D2模块和原训练负样本；所有评测保留完整候选。Score不参与，主模型和正式结果不替换。','',
        '| 配置 | Choice accuracy ↑ | Noul accuracy ↑ | 状态 |','|---|---:|---:|---|']
    for arm,label in [('ce','原CE/BCE参考'),('rlcd_both','两类任务原RLCD'),('rlcd_scaled','两类任务RLCD增量×0.1'),('proper','两类任务直接适当目标α=0.75')]:
        e=arms[arm];m=e.get('metrics',{}).get('test_in_domain')
        vals=[f"{m['task/'+t]['accuracy']:.4f}" if m else 'Placeholder' for t in ['choice','noul']]
        lines.append('| '+label+' | '+' | '.join(vals)+' | '+e['status']+' |')
    lines+=['','每类测试1,000条。按来源报告Accuracy/NLL/Brier；配对区间在运行目录，单seed小样本只用于筛选方向。',
            '运行根：runs/pilot_classification_proper_20261106_v1；详见 [执行记录](experiment_log_20261106_classification_proper.md)。','',END]
    rendered='\n'.join(lines)
    (ROOT/'report.md').write_text(rendered+'\n')
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);text=DOC.read_text()
        if START in text:
            a=text.index(START);b=text.index(END,a)+len(END);text=text[:a]+rendered+text[b:]
        else:text=text.rstrip()+'\n\n'+rendered+'\n'
        tmp=DOC.with_suffix('.classification_proper.tmp');tmp.write_text(text);tmp.replace(DOC)
    return all(e['status'] in ('complete','failed') for e in arms.values())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    while True:
        finished=report()
        if finished or not a.watch:break
        time.sleep(30)
