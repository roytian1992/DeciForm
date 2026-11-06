"""Overwrite the R1 pilot block from actual completed native-score artifacts."""
import argparse,collections,fcntl,json,time
from pathlib import Path
import numpy as np
from common import PROJECT,atomic,sha
START='<!-- NUMERIC_TRIE_R1_SMALL_START -->'
END='<!-- NUMERIC_TRIE_R1_SMALL_END -->'


def read(p):return json.loads(p.read_text()) if p.exists() else None


def predictions(out,split):
    rows=[]
    for p in sorted((out/'evaluations').glob('epoch_1.'+split+'.rank*.jsonl')):
        rows.extend(json.loads(x) for x in p.open())
    assert rows and len(rows)==len({r['id'] for r in rows})
    return {r['id']:r for r in rows}


def paired(reference,out,split):
    a,b=predictions(reference,split),predictions(out,split)
    assert a.keys()==b.keys()
    groups=collections.defaultdict(list)
    for k in sorted(a):
        x,y=a[k],b[k];assert x['label']==y['label'] and x['support']==y['support']
        groups[x['source']].append((y['mae']-x['mae'])/(x['support'][-1]-x['support'][0]))
    rng=np.random.default_rng(20261101);draws=np.zeros(2000)
    for v in groups.values():
        v=np.asarray(v);draws+=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)/len(groups)
    return {'rows':len(a),'r1_minus_control_macro_nmae':float(np.mean([np.mean(v) for v in groups.values()])),
            'paired_source_stratified_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),
            'uncertainty':'Conditional on these single-seed models and fixed small evaluation population'}


def outcome(out):
    cfg=read(out/'config.json');status=read(out/'status.json') or {'phase':'queued'};done=read(out/'complete.json')
    e={'path':str(out),'status':status,'progress':read(out/'progress.json'),'complete':done,'metrics':{},'result_sha256':{}}
    if done:
        assert done['steps']==cfg['total_steps'] and done['sample_visits']==cfg['train_rows']
        for split in ['dev','test_in_domain']:
            p=out/'evaluations'/('epoch_1.'+split+'.json');m=read(p)
            assert m['rows']==cfg['evaluation_counts'][split] and m['complete_output_grid'] and m['score_decision_rule']=='median'
            e['metrics'][split]=m;e['result_sha256'][str(p)]=sha(p)
    return e


def update(root):
    top=read(root/'experiment_manifest.json');results={};comparison={}
    lines=[START,'','## Score 小样本：前缀 CE/CDF＋R1 均值回归','',
        '固定相同小样本：ASAP训练512/开发64/测试128；STS-B与WMT英中联合训练2048/开发256/测试512。',
        '复用已完成前缀CE/CDF对照；新增R1只加归一化预测均值MSE，权重1。fresh Base、1轮、batch16；全文与合法分数支持完整。',
        '主解码仍为原生分布中位数，期望只用于训练辅助；未加数字序列监督。','',
        '| 数据 | 方法 | 测试MAE ↓ | NMAE ↓ | 状态 |','|---|---|---:|---:|---|']
    for name,item in top['arms'].items():
        panel=item['panel'];reference=Path(item['reference']);current=root/name
        entries={'control':outcome(reference),'r1':outcome(current)};results[name]=entries
        for variant,e in entries.items():
            m=e['metrics'].get('test_in_domain')
            labels=[('ASAP','task/score')] if panel=='asap' else [('STS-B','source/stsb'),('WMT英中','source/wmt_en_zh')]
            for label,key in labels:
                mae=f"{m[key]['mae']:.6f}" if m else 'Placeholder'
                nmae=f"{m['macro_source_nmae'] if panel=='asap' else m[key]['nmae']:.6f}" if m else 'Placeholder'
                lines.append('| '+label+' | '+('前缀 CE/CDF' if variant=='control' else '前缀 CE/CDF＋R1')+' | '+mae+' | '+nmae+' | '+e['status']['phase']+' |')
        if all(e['complete'] for e in entries.values()):comparison[panel]={s:paired(reference,current,s) for s in ['dev','test_in_domain']}
    lines+=['','ASAP NMAE按八题等权；小样本单seed探索，不声称稳定泛化，不依据测试改变解码。',
        '执行记录：docs/experiment_log_20261106_numeric_trie_r1.md；根目录：runs/pilot_numeric_trie_r1_20261106_v1。','',END]
    block='\n'.join(lines);(root/'report.md').write_text(block+'\n')
    atomic(root/'results_summary.json',{'updated_unix':time.time(),'arms':results,'paired_comparisons':comparison})
    doc=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);s=doc.read_text()
        if START in s:
            i=s.index(START);j=s.index(END,i)+len(END);s=s[:i]+block+s[j:]
        else:s=s.rstrip()+'\n\n'+block+'\n'
        tmp=doc.with_suffix('.numeric_trie_r1.tmp');tmp.write_text(s);tmp.replace(doc)
    return all(e['r1']['complete'] for e in results.values()) or any(e['r1']['status']['phase']=='failed' for e in results.values())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=update(a.root.resolve())
        if done or not a.watch:break
        time.sleep(30)
