"""Four-GPU inference supervisor, failure/stall visibility and locked result updates."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from common import PROJECT,atomic,sha
from zero_shot_logits import RULES,PANELS,summarize


def reports(root,manifest):
    complete=[]
    for panel in PANELS:
        if (root/f'{panel}.json').exists():complete.append(panel);continue
        dirs=[root/f'shard_{i}' for i in range(manifest['shards'])]
        if not all((d/f'{panel}.json').exists() for d in dirs):continue
        records=[];evidence={}
        for d in dirs:
            path=d/f'{panel}.predictions.jsonl';digest=sha(path)
            assert json.loads((d/f'{panel}.json').read_text())['predictions_sha256']==digest
            evidence[str(path)]=digest
            records.extend(json.loads(l) for l in path.open())
        assert len(records)==manifest['panels'][panel]['records_both_rules']
        assert len({(r['id'],r['view'],r['scoring_rule']) for r in records})==len(records)
        atomic(root/f'{panel}.json',{'status':'complete','panel':panel,'rows':manifest['panels'][panel]['rows'],
            'reports':{rule:summarize(panel,[r for r in records if r['scoring_rule']==rule]) for rule in RULES},
            'predictions_sha256':evidence,'optimizer_updates':0,'primary_rule':'sequence_likelihood'})
        complete.append(panel)
    return complete


def update_document(root,manifest,phase,complete):
    path=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    begin,end='<!-- ZERO_SHOT_LOGITS_START -->','<!-- ZERO_SHOT_LOGITS_END -->'
    text=['## 无训练 LLM logits 对照（2026-11-06）','',
          f'状态：{phase}。根目录：{root.relative_to(PROJECT)}。官方 Qwen3.5-9B 指令模型，0 次任务训练；0–3 卡。',
          '','全合法 JSON 答案似然为主，逐节点受约束归一化为单列对照。主测试 15,000；ASAP 2,598；STS-B 1,379；WMT en-zh 1,000。Choice 全候选 10 个顺序，Noul 原序/倒序。',
          '','Score 原始 MAE 使用预先固定的概率中位数，期望/argmax 同时记录。这里只是合法规范答案条件分布，不保证概率校准，也不保证顺序不变。','']
    if 'main' in complete:
        data=json.loads((root/'main.json').read_text())['reports']
        text+=['| 规则 | Choice Acc | Noul Acc | Score MAE ↓ |','|---|---:|---:|---:|']
        for rule,r in data.items():text.append(f"| {rule} | {r['task/choice']['accuracy']:.4f} | {r['task/noul']['accuracy']:.4f} | {r['task/score']['mae']:.4f} |")
    else:text+=['主测试：待完成。']
    for panel in ['asap','continuous']:
        if panel not in complete:text+=['',panel+'：待完成。'];continue
        text+=['',f'{panel}，结果源：{root.relative_to(PROJECT)}/{panel}.json','',
               '| 规则 / 来源 | MAE ↓ |','|---|---:|']
        for rule,r in json.loads((root/f'{panel}.json').read_text())['reports'].items():
            for source,m in r.items():
                if source.startswith('source/'):text.append(f"| {rule} / {source[7:]} | {m['mae']:.6f} |")
            text.append(f"| {rule} / 跨来源 NMAE | {r['macro_source_nmae']:.6f} |")
    for panel in ['choice_order','noul_order']:
        if panel not in complete:text+=['',panel+'：待完成。'];continue
        text+=['',f'{panel}，结果源：{root.relative_to(PROJECT)}/{panel}.json']
        for rule,r in json.loads((root/f'{panel}.json').read_text())['reports'].items():
            r=r.get('new_d2_seven_orders',r)
            text.append(f"- {rule}：倒序改判率 {r['order_metrics']['reverse_flip_rate_on_valid_pairs']:.6f}；完整多顺序结果见 JSON。")
    block=begin+'\n'+'\n'.join(text)+'\n'+end
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=path.read_text()
        if begin in old:
            a=old.index(begin);b=old.index(end,a)+len(end);new=old[:a]+block+old[b:]
        else:new=old.rstrip()+'\n\n'+block+'\n'
        tmp=path.with_suffix('.logits.tmp');tmp.write_text(new);tmp.replace(path)


def run(root):
    lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((root/'experiment_manifest.json').read_text())
    for p,d in m['code_sha256'].items():assert sha(PROJECT/p)==d,p
    for p,s in m['model']['verified_file_stats'].items():
        st=Path(p).stat();assert st.st_size==s['bytes'] and st.st_mtime_ns==s['mtime_ns'],p
    children={};handles=[];finished={};started=time.time()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(g):int(v) for g,v in (l.split(',') for l in used.splitlines())}
    for shard,gpu in enumerate(m['gpus']):
        out=root/f'shard_{shard}';out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():finished[shard]=0;continue
        if (out/'failure.json').exists():raise RuntimeError(f'Failure requires review before resume: {out}')
        if (out/'allocation.json').exists():
            old=json.loads((out/'allocation.json').read_text())
            cmd=Path(f"/proc/{old['pid']}/cmdline")
            if cmd.exists() and str(root).encode() in cmd.read_bytes():raise RuntimeError('Worker already alive')
        assert used[gpu]<1000,(gpu,used[gpu])
        cmd=[m['python'],str(PROJECT/'scripts/formal/worker_zero_shot_logits.py'),'--root',str(root),'--shard',str(shard)]
        log=(out/'worker.log').open('a',buffering=1);handles.append(log)
        child=subprocess.Popen(cmd,cwd=PROJECT,env={**os.environ,**m['environment_variables'],'CUDA_VISIBLE_DEVICES':str(gpu)},
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[shard]=child
        atomic(out/'allocation.json',{'gpu':gpu,'pid':child.pid,'command':cmd,'started_unix':time.time()})
    last_document=None
    while children:
        states={}
        for shard,child in list(children.items()):
            code=child.poll();out=root/f'shard_{shard}'
            if code is not None:
                finished[shard]=code if code else (0 if (out/'complete.json').exists() else -1)
                del children[shard]
            if (out/'progress.json').exists():states[shard]=json.loads((out/'progress.json').read_text())
        completed=reports(root,m);failed=any(c!=0 for c in finished.values())
        stalled=[s for s,r in states.items() if s in children and time.time()-r['updated_unix']>900]
        phase='failed' if failed else 'stalled' if stalled else 'running'
        atomic(root/'status.json',{'phase':phase,'updated_unix':time.time(),'active_pids':{s:c.pid for s,c in children.items()},
            'workers':states,'completed_exit_codes':finished,'available_reports':completed})
        if failed or stalled:
            atomic(root/'alerts.json',{'phase':phase,'updated_unix':time.time(),'failed_shards':{s:c for s,c in finished.items() if c},'stalled_shards':stalled,
                'action':'Inspect shard failure/progress/log. Existing other shards may finish; no hidden retry or omitted record.'})
        signature=(phase,tuple(completed))
        if signature!=last_document:update_document(root,m,phase,completed);last_document=signature
        if children:time.sleep(10)
    assert all(c==0 for c in finished.values()),finished
    completed=reports(root,m);assert set(completed)==set(PANELS)
    atomic(root/'complete.json',{'status':'complete','seconds':time.time()-started,'predictions':m['expected_predictions'],
        'records':m['expected_records'],'optimizer_updates':0,'finished_unix':time.time()})
    atomic(root/'status.json',{'phase':'complete','exit_code':0,'updated_unix':time.time(),'available_reports':completed})
    update_document(root,m,'complete',completed)
    for h in handles:h.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root.resolve())
    except BaseException as exc:
        atomic(a.root/'alerts.json',{'updated_unix':time.time(),'error':str(exc),'traceback':traceback.format_exc()});raise
