"""Own GPU pairs 0/1 and 2/3 only after existing D2 jobs finish successfully."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT,atomic


def memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {a.strip():int(b.strip()) for a,b in (line.split(',') for line in out.splitlines())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((root/'experiment_manifest.json').read_text())
    assert not (root/'status.json').exists(),'Existing queue: recover individual runs explicitly'
    pending=list(m['priority']);active={};finished={};handles=[]
    while pending or active:
        for name,desc in list(active.items()):
            code=desc['child'].poll()
            if code is not None:finished[name]=code;del active[name]
        failed=any(code!=0 for code in finished.values())
        used=memory();reserved={desc['gpus'] for desc in active.values()}
        if not failed:
            for pair in m['gpu_pairs']:
                if not pending or pair in reserved:continue
                dep=Path(m['release_dependencies'][pair])
                if not dep.exists():continue
                released=json.loads(dep.read_text())
                if released.get('phase')!='complete' or released.get('exit_code')!=0:continue
                if any(used[g]>=m['max_memory_mib_before_launch'] for g in pair.split(',')):continue
                name=pending.pop(0);path=Path(m['arms'][name]['manifest'])
                run=copy.deepcopy(json.loads(path.read_text()));run['gpus']=pair
                run['allocated_unix']=time.time();runtime=path.parent/'runtime_manifest.json'
                assert not runtime.exists();atomic(runtime,run)
                log=(path.parent/'supervisor.log').open('a',buffering=1);handles.append(log)
                child=subprocess.Popen(run['launch_command'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]={'child':child,'gpus':pair};reserved.add(pair)
        atomic(root/'status.json',{'phase':'failed' if failed else 'running','updated_unix':time.time(),
            'pending':pending,'active':{n:{'pid':d['child'].pid,'gpus':d['gpus']} for n,d in active.items()},
            'completed_exit_codes':finished})
        if failed and not active:raise RuntimeError('Baseline failed; inspect per-run alerts and explicit recovery command')
        if pending or active:time.sleep(10)
    reports={}
    for name in m['arms']:
        reports[name]={s:json.loads((root/name/'evaluations'/f'epoch_1.{s}.json').read_text()) for s in ('dev','test_in_domain')}
    refs={}
    for panel,paths in m['references'].items():
        refs[panel]={Path(p).name:json.loads((Path(p)/'evaluations/epoch_1.test_in_domain.json').read_text()) for p in paths}
    atomic(root/'results_summary.json',{'status':'complete','models':reports,'references':refs,
        'primary':m['primary'],'wmt_reporting':m['wmt_reporting'],'test_selection':False})
    atomic(root/'status.json',{'phase':'complete','updated_unix':time.time(),'completed_exit_codes':finished})
    for h in handles:h.close()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys
        root=Path(sys.argv[sys.argv.index('--root')+1])
        atomic(root/'alerts.json',{'error':str(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()})
        raise
