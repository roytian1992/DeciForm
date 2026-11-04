"""Own the two native-score arms in tmux, with per-arm resumable supervisors."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT,atomic


def memory():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {a.strip():int(b.strip()) for a,b in (s.split(',') for s in value.splitlines())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'experiment_manifest.json').read_text());pending=set(manifest['arms']);active={};finished={};handles=[]
    assert not (root/'status.json').exists(),'Existing run: recover individual arm using its manifest and --resume'
    while pending or active:
        for name,child in list(active.items()):
            code=child.poll()
            if code is not None:finished[name]=code;del active[name]
        # Continue observing an already-started peer, but never launch more after failure.
        failed=any(code!=0 for code in finished.values());used=memory()
        if not failed:
            for name in sorted(pending.copy()):
                desc=manifest['arms'][name]
                if all(used[g]<manifest['max_memory_mib_before_launch'] for g in desc['gpus'].split(',')):
                    p=Path(desc['manifest']);m=json.loads(p.read_text());log=(p.parent/'supervisor.log').open('a',buffering=1);handles.append(log)
                    active[name]=subprocess.Popen(m['launch_command'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    pending.remove(name)
        atomic(root/'status.json',{'phase':'failed' if failed else 'running','updated_unix':time.time(),
            'pending':sorted(pending),'active':{n:p.pid for n,p in active.items()},'completed_exit_codes':finished})
        if failed and not active:raise RuntimeError('ASAP arm failed; inspect per-arm alerts before explicit resume')
        if pending or active:time.sleep(10)
    reports={}
    for name in manifest['arms']:
        p=root/name
        reports[name]={s:json.loads((p/'evaluations'/f'epoch_1.{s}.json').read_text()) for s in ('dev','test_in_domain')}
    atomic(root/'results_summary.json',{'status':'complete','models':reports,'primary':'native MAE per prompt; equal-prompt NMAE','test_selection':False})
    atomic(root/'status.json',{'phase':'complete','updated_unix':time.time(),'completed_exit_codes':finished})
    for h in handles:h.close()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys
        root=Path(sys.argv[sys.argv.index('--root')+1]);atomic(root/'alerts.json',{'error':str(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()})
        raise
