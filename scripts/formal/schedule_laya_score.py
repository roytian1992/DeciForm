"""Two persistent GPU-pair queues; frozen preflight/train/eval and locked reporting."""
import argparse,fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from common import PROJECT,atomic,sha


def update_report(root,m,statuses):
    summary={}
    lines=['## Laya 多档位 Score 专项（2026-11-06）','',
        '从原始 ModernBERT-large 和随机 Laya head 重新训练。每个布局/数据面板独立初始化；不继承20万条主实验 checkpoint。原 Laya RLCD、全参数训练，专项数据1轮/global batch64。','',
        'ASAP训练7,784/测试2,598；STS-B＋WMT英中联合训练8,129，分别测试1,379/1,000。完整输入及合法档位；原始小数标签保留。主指标为中位数解码原生MAE，附均值/argmax诊断。','',
        '| 布局 | ASAP MAE ↓ | ASAP跨题NMAE ↓ | STS-B MAE ↓ | WMT英中 MAE ↓ | 状态 |',
        '|---|---:|---:|---:|---:|---|']
    for layout,jobs in m['queues'].items():
        vals={}
        for job in jobs:
            out=root/job['name'];result=out/'evaluations/test_in_domain.original.json'
            if (out/'complete.json').exists() and result.exists():
                d=json.loads(result.read_text());summary[job['name']]={'result':str(result),'sha256':sha(result),'report':d,
                    'complete':json.loads((out/'complete.json').read_text())}
                if job['name'].endswith('_asap'):
                    vals.update(asap=f"{d['task/score']['mae']:.6f}",nmae=f"{d['macro_source_nmae']:.6f}")
                else:
                    vals.update(stsb=f"{d['source/stsb']['mae']:.6f}",wmt=f"{d['source/wmt_en_zh']['mae']:.6f}")
        state='; '.join(j['name'].split('_',1)[1]+': '+statuses.get(j['name'],{}).get('phase','queued') for j in jobs)
        lines.append('| '+layout+' | '+' | '.join(vals.get(k,'Placeholder') for k in ('asap','nmae','stsb','wmt'))+' | '+state+' |')
    lines+=['',f'结果根目录：{root.relative_to(PROJECT)}。各实验 evaluations/test_in_domain.original.json、score_rules.json 与逐样本文件为证据。','',
        '与当前 D2/生成式复跑对齐数据和训练暴露；backbone、全参数/LoRA、原生优化目标不同。历史 Kev 连续面板含额外英德训练来源，保留为历史参考，不称完全同数据比较。']
    atomic(root/'results_summary.json',{'updated_unix':time.time(),'completed':summary,'statuses':statuses})
    begin,end='<!-- LAYA_SCORE_START -->','<!-- LAYA_SCORE_END -->'
    block=begin+'\n'+'\n'.join(lines)+'\n'+end
    path=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=path.read_text()
        if begin in text:
            a=text.index(begin);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
        else:text=text.rstrip()+'\n\n'+block+'\n'
        tmp=path.with_suffix('.laya_score.tmp');tmp.write_text(text);tmp.replace(path)


def run(root):
    lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((root/'experiment_manifest.json').read_text())
    for p,d in m['code_sha256'].items():assert sha(p)==d,p
    for p,d in m['base_files_sha256'].items():assert sha(p)==d,p
    for jobs in m['queues'].values():
        for job in jobs:
            assert sha(job['config'])==job['config_sha256']
            out=root/job['name']
            if (out/'failure.json').exists():raise RuntimeError(f'Review previous failure: {out}')
            alloc=out/'allocation.json'
            if alloc.exists():
                pid=json.loads(alloc.read_text())['pid'];proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists() and str(root).encode() in proc.read_bytes():raise RuntimeError(f'Existing active child {pid}')
    active={};handles=[];failed={};statuses={};last_report=0
    def start(queue,job,mode):
        out=root/job['name'];cfg=json.loads(Path(job['config']).read_text())
        cmd=[m['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             str(PROJECT/'scripts/formal/worker_laya_score.py'),'--config',job['config'],'--mode',mode]
        if mode=='train' and (out/'resume.pt').exists():cmd.append('--resume')
        log=(out/f'{mode}.log').open('a',buffering=1);handles.append(log)
        child=subprocess.Popen(cmd,cwd=PROJECT,env={**os.environ,**m['environment_variables'],'CUDA_VISIBLE_DEVICES':job['gpus']},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        desc={'child':child,'job':job,'mode':mode,'started':time.time()}
        active[queue]=desc
        atomic(out/'allocation.json',{'pid':child.pid,'command':cmd,'gpus':job['gpus'],'mode':mode,'started_unix':desc['started']})
    while True:
        for queue,desc in list(active.items()):
            child=desc['child'];job=desc['job'];out=root/job['name'];mode=desc['mode']
            cfg=json.loads(Path(job['config']).read_text());code=child.poll()
            pp=out/('preflight.progress.json' if mode=='preflight' else 'progress.json')
            progress=json.loads(pp.read_text()) if pp.exists() else {}
            state={'phase':mode,'gpus':job['gpus'],'pid':child.pid,'progress':progress,'updated_unix':time.time(),'exit_code':code}
            if code is not None:
                del active[queue]
                proof=out/('preflight.json' if mode=='preflight' else 'complete.json')
                if code!=0 or not proof.exists():
                    failed[queue]=job['name'];state.update(phase='failed')
                    atomic(out/'failure.json',{'exit_code':code,'updated_unix':time.time(),'error':f'{mode} failed; inspect rank failure files and log'})
                elif mode=='train':
                    d=json.loads(proof.read_text())
                    assert d['sample_visits']==cfg['train_rows'] and d['steps']==cfg['total_steps']
                    state.update(phase='complete')
                else:
                    d=json.loads(proof.read_text());assert d['status']=='passed' and d['optimizer_updates']==0
                    state.update(phase='preflight_passed')
            elif time.time()-max(desc['started'],progress.get('updated_unix',0))>600:
                atomic(out/'alerts.json',{'updated_unix':time.time(),'error':'No progress for10minutes','pid':child.pid})
                if time.time()-max(desc['started'],progress.get('updated_unix',0))>1800:
                    os.killpg(child.pid,signal.SIGTERM)
            statuses[job['name']]=state;atomic(out/'status.json',state)
        remaining=False
        used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={g.strip():int(v) for g,v in (l.split(',') for l in used.splitlines())}
        for queue,jobs in m['queues'].items():
            if queue in failed:continue
            pending=[j for j in jobs if not (root/j['name']/'complete.json').exists()]
            if not pending:continue
            remaining=True
            if queue in active:continue
            job=pending[0]
            if any(used[g]>1000 for g in job['gpus'].split(',')):continue
            mode='train' if (root/job['name']/'preflight.json').exists() else 'preflight'
            start(queue,job,mode)
        for jobs in m['queues'].values():
            for j in jobs:
                if j['name'] not in statuses:
                    statuses[j['name']]={'phase':'complete' if (root/j['name']/'complete.json').exists() else 'queued'}
        atomic(root/'status.json',{'phase':'failed' if failed else 'running','updated_unix':time.time(),
            'active':{q:{'pid':d['child'].pid,'name':d['job']['name'],'mode':d['mode']} for q,d in active.items()},
            'failed':failed,'arms':statuses})
        if time.time()-last_report>30:
            update_report(root,m,statuses);last_report=time.time()
        if not remaining and not active:break
        time.sleep(5)
    if failed:raise RuntimeError(f'Failed queues {failed}; other queues may have completed')
    update_report(root,m,statuses)
    atomic(root/'complete.json',{'status':'complete','finished_unix':time.time(),'arms':list(statuses),'parent_checkpoint':None})
    atomic(root/'status.json',{'phase':'complete','exit_code':0,'updated_unix':time.time(),'arms':statuses})
    for handle in handles:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root.resolve())
    except BaseException as exc:
        atomic(a.root/'alerts.json',{'updated_unix':time.time(),'error':str(exc),'traceback':traceback.format_exc()});raise
