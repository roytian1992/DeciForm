"""Report matched small Score arms; never compare to full-data metrics."""
import argparse
import collections
import fcntl
import json
import time
from pathlib import Path
import numpy as np
from common import PROJECT, atomic, sha

START='<!-- SCORE_PROPER_SMALL_START -->';END='<!-- SCORE_PROPER_SMALL_END -->'
def read(path):return json.loads(path.read_text()) if path.exists() else None

def paired(root,panel,split):
    def rows(variant):
        output=[]
        for path in sorted((root/(panel+'_'+variant)/'evaluations').glob('epoch_1.'+split+'.rank*.jsonl')):
            output.extend(json.loads(line) for line in path.open())
        assert len(output)==len({r['id'] for r in output})
        return {r['id']:r for r in output}
    base,other=rows('cdf'),rows('proper');assert base.keys()==other.keys() and base
    groups=collections.defaultdict(list)
    for k in sorted(base):
        a,b=base[k],other[k];assert a['label']==b['label'] and a['support']==b['support']
        groups[a['source']].append((b['mae']-a['mae'])/(a['support'][-1]-a['support'][0]))
    rng=np.random.default_rng(20261101);draws=np.zeros(2000)
    for values in groups.values():
        values=np.asarray(values);draws+=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)/len(groups)
    return {'rows':len(base),'proper_minus_cdf_macro_source_nmae':float(np.mean([np.mean(x) for x in groups.values()])),
       'paired_source_stratified_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),
       'uncertainty':'conditional on frozen small sample and single-seed checkpoints; no stability claim'}

def update(root):
    top=read(root/'experiment_manifest.json');arms={};comparisons={}
    for name,item in top['arms'].items():
        out=root/name;entry={'panel':item['panel'],'variant':item['variant'],'status':read(out/'status.json') or {'phase':'queued'},'complete':read(out/'complete.json')}
        if (out/'progress.json').exists():entry['progress']=read(out/'progress.json')
        if entry['complete']:
            cfg=read(out/'config.json');assert entry['complete']['sample_visits']==cfg['train_rows'] and entry['complete']['steps']==cfg['total_steps']
            entry['metrics']={};entry['result_sha256']={}
            for split in ['dev','test_in_domain']:
                path=out/'evaluations'/('epoch_1.'+split+'.json');m=read(path)
                assert m['rows']==cfg['evaluation_counts'][split] and m['complete_output_grid']
                entry['metrics'][split]=m;entry['result_sha256'][str(path)]=sha(path)
        arms[name]=entry
    for panel in ['continuous','asap']:
        if all(arms[panel+'_'+v]['complete'] for v in ['cdf','proper']):
            comparisons[panel]={split:paired(root,panel,split) for split in ['dev','test_in_domain']}
    lines=[START,'','## Score 小样本：原D2 CDF与直接适当评分目标','',
       '固定小样本、同Base随机初始化、相同架构与优化器；每组1轮、batch16。连续评分训练2,048/开发256/测试512；ASAP训练512/开发64/测试128。',
       '每个划分保留原始全文、分数和完整评分档位；原生中位数MAE为主，不能与全量训练结果作因果比较。','',
       '| 数据 | 损失 | 小测试集 MAE ↓ | 小测试集 NMAE ↓ | 状态 |','|---|---|---:|---:|---|']
    for panel in ['continuous','asap']:
        labels=[('STS-B','source/stsb'),('WMT英中','source/wmt_en_zh')] if panel=='continuous' else [('ASAP','task/score')]
        for variant in ['cdf','proper']:
            e=arms[panel+'_'+variant];m=e.get('metrics',{}).get('test_in_domain')
            for label,key in labels:
                mae=f"{m[key]['mae']:.6f}" if m else 'Placeholder'
                nmae=f"{m['macro_source_nmae'] if panel=='asap' else m[key]['nmae']:.6f}" if m else 'Placeholder'
                lines.append('| '+label+' | '+('原D2 CDF-log' if variant=='cdf' else '直接log+spherical')+' | '+mae+' | '+nmae+' | '+e['status']['phase']+' |')
    lines+=['','ASAP的NMAE按8个题目等权；连续评分按各来源单独汇报。单seed探索，不依据测试选择解码或调参。',
       '根目录：runs/pilot_score_proper_20261106_v1。详见 [执行记录](experiment_log_20261106_score_proper_pilot.md)。','',END]
    block='\n'.join(lines);(root/'report.md').write_text(block+'\n')
    atomic(root/'results_summary.json',{'updated_unix':time.time(),'arms':arms,'paired_comparisons':comparisons,'queue':read(root/'queue_status.json')})
    doc=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);s=doc.read_text()
        if START in s:
            a=s.index(START);b=s.index(END,a)+len(END);s=s[:a]+block+s[b:]
        else:s=s.rstrip()+'\n\n'+block+'\n'
        tmp=doc.with_suffix('.score_proper_small.tmp');tmp.write_text(s);tmp.replace(doc)
    return all(e['complete'] for e in arms.values()) or (root/'queue_alert.json').exists()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        done=update(args.root.resolve())
        if done or not args.watch:break
        time.sleep(30)
