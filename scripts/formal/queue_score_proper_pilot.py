"""Run the four frozen small arms sequentially on the authorized GPU pair."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from common import PROJECT, atomic, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();top=json.loads((root/'experiment_manifest.json').read_text())
    try:
        for name in top['queue']:
            out=root/name;m=json.loads((out/'experiment_manifest.json').read_text())
            if (out/'complete.json').exists():
                assert json.loads((out/'status.json').read_text())['phase']=='complete'
                continue
            if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['phase']=='failed':
                raise RuntimeError('A prior failed arm requires inspection before queue recovery: '+name)
            for path,h in top['code_sha256'].items():assert sha(PROJECT/path)==h,path
            assert m['gpus']=='6,7'
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(a):int(b) for a,b in (line.split(',') for line in raw.splitlines())}
            if any(memory[i]>1000 for i in [6,7]):
                raise RuntimeError('GPU6/7 occupied before next arm: '+str(memory))
            cmd=m['resume_command'] if (out/'resume.pt').exists() else m['launch_command']
            atomic(root/'queue_status.json',{'phase':'running','active':name,'remaining':top['queue'][top['queue'].index(name):],'updated_unix':time.time()})
            with (out/'supervisor.log').open('a',buffering=1) as log:
                child=subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    atomic(root/'queue_status.json',{'phase':'running','active':name,'pid':child.pid,'updated_unix':time.time()})
                    time.sleep(5)
            if child.returncode:raise RuntimeError(name+' supervisor failed; inspect arm alerts/logs')
            assert json.loads((out/'status.json').read_text())['phase']=='complete'
            time.sleep(5)
        atomic(root/'queue_status.json',{'phase':'complete','updated_unix':time.time(),'arms':top['queue']})
    except BaseException as exc:
        atomic(root/'queue_alert.json',{'error':str(exc),'time':time.time()})
        atomic(root/'queue_status.json',{'phase':'failed','error':str(exc),'updated_unix':time.time()})
        raise

if __name__=='__main__':main()
