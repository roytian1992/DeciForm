"""Scheduling-only amendment: two matched GPU pairs, unchanged workers/configs."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from common import PROJECT, atomic, sha

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--lane',choices=['cdf','proper'],required=True);a=ap.parse_args()
    root=a.root.resolve();top=read(root/'experiment_manifest.json');lane=a.lane
    plan=top['parallel_lanes'][lane];names=plan['arms'];gpu_ids=[int(x) for x in plan['gpus'].split(',')]
    status_path=root/('lane_'+lane+'_status.json')
    try:
        for name in names:
            out=root/name;m=read(out/'experiment_manifest.json');assert m['gpus']==plan['gpus']
            status=read(out/'status.json')
            if status.get('phase')=='failed':raise RuntimeError(name+' previously failed; inspect before resume')
            if status.get('phase')!='complete':
                if status.get('phase') in ('train','preflight','preflight_complete','train_complete'):
                    # Adopt the existing supervisor without changing its process, RNG or step.
                    while True:
                        status=read(out/'status.json')
                        atomic(status_path,{'phase':'observing_existing','active':name,'gpus':plan['gpus'],'updated_unix':time.time()})
                        if status.get('phase') in ('complete','failed'):break
                        if time.time()-status.get('updated_unix',0)>120:raise RuntimeError('Existing supervisor heartbeat stale: '+name)
                        time.sleep(5)
                    if status['phase']=='failed':raise RuntimeError(name+' existing supervisor failed')
                else:
                    for path,h in top['code_sha256'].items():assert sha(PROJECT/path)==h,path
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    memory={int(x):int(y) for x,y in (line.split(',') for line in raw.splitlines())}
                    if any(memory[i]>1000 for i in gpu_ids):raise RuntimeError('Assigned GPUs occupied before '+name+': '+str(memory))
                    cmd=m['resume_command'] if (out/'resume.pt').exists() else m['launch_command']
                    with (out/'supervisor.log').open('a',buffering=1) as log:
                        child=subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            atomic(status_path,{'phase':'running','active':name,'gpus':plan['gpus'],'pid':child.pid,'updated_unix':time.time()})
                            time.sleep(5)
                    if child.returncode:raise RuntimeError(name+' supervisor failed')
            assert read(out/'status.json')['phase']=='complete' and (out/'complete.json').exists()
            if lane=='cdf' and name=='continuous_cdf':
                # The old dispatcher was stopped, not its child supervisor.
                old=top.get('suspended_dispatcher_pid')
                proc=Path('/proc')/str(old)
                if old and proc.exists():
                    command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
                    if 'queue_score_proper_pilot.py' not in command:raise RuntimeError('Old dispatcher PID changed')
                    os.kill(old,signal.SIGKILL)
                atomic(root/'old_dispatcher_retired.json',{'pid':old,'after':'continuous_cdf complete','time':time.time()})
            time.sleep(5)
        atomic(status_path,{'phase':'complete','gpus':plan['gpus'],'arms':names,'updated_unix':time.time()})
    except BaseException as exc:
        atomic(root/('lane_'+lane+'_alert.json'),{'error':str(exc),'time':time.time()})
        atomic(status_path,{'phase':'failed','error':str(exc),'updated_unix':time.time()})
        raise

if __name__=='__main__':main()
