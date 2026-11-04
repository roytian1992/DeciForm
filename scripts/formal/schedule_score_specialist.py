"""Start S3 on 4/5, then dispatch S4 to a released matching two-GPU group."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT,atomic,sha


def memory():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {a.strip():int(b.strip()) for a,b in (line.split(',') for line in s.splitlines())}


def success(path):
    if not Path(path).exists():return False
    x=json.loads(Path(path).read_text())
    return x.get('phase')=='complete' and x.get('exit_code',0)==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'experiment_manifest.json').read_text());policy=plan['allocation_policy']
    active={};finished={};launched=set();states={};handles=[]
    def launch(name,gpus):
        original=Path(plan['arms'][name]['manifest']);m=json.loads(original.read_text());out=original.parent
        runtime=out/'runtime_manifest.json'
        if runtime.exists():raise FileExistsError('Existing allocation: inspect/resume the specific arm rather than silently relaunch')
        m.update(gpus=','.join(gpus),allocation_unix=time.time(),prepared_manifest_sha256=sha(original))
        m['launch_command']=[m['python'],m['supervisor'],'--manifest',str(runtime)]
        atomic(runtime,m)
        log=(out/'supervisor.log').open('a',buffering=1);handles.append(log)
        child=subprocess.Popen(m['launch_command'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[name]=child;launched.add(name);states[name]={'phase':'running','gpus':gpus,'pid':child.pid,'runtime_manifest':str(runtime)}
    while len(finished)<2:
        for name,child in list(active.items()):
            code=child.poll()
            if code is not None:
                finished[name]=code;states[name].update(phase='complete' if code==0 else 'failed',exit_code=code)
                del active[name]
        if any(code for code in finished.values()):raise RuntimeError('A specialist arm failed; inspect its status/alerts; pending launches remain stopped')
        used=memory();free=lambda g:all(used[i]<policy['max_memory_mib_before_launch'] for i in g)
        if 's3_continuous_ce' not in launched and free(policy['s3']):launch('s3_continuous_ce',policy['s3'])
        s3=root/'s3_continuous_ce'
        if 's4_continuous_cdf' not in launched and (s3/'preflight_passed.json').exists():
            for option in policy['s4_preference']:
                if success(option['prerequisite']) and free(option['gpus']):
                    launch('s4_continuous_cdf',option['gpus']);break
        atomic(root/'status.json',{'phase':'complete' if len(finished)==2 else 'running',
            'updated_unix':time.time(),'arms':states,'pending':[n for n in plan['arms'] if n not in launched],
            'gpu_memory_mib':used})
        if len(finished)<2:time.sleep(10)
    reports={}
    for name in plan['arms']:
        p=root/name;reports[name]={split:json.loads((p/'evaluations'/f'epoch_1.{split}.json').read_text()) for split in ('dev','test_in_domain')}
    atomic(root/'results_summary.json',{'status':'complete','models':reports,'primary':'native MAE and equal-source NMAE','test_selection':False})
    for h in handles:h.close()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys
        root=Path(sys.argv[sys.argv.index('--root')+1]);atomic(root/'alerts.json',{'error':str(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()})
        raise
