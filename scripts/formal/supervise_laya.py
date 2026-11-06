"""Durable Laya supervisor with source checks, heartbeat and resumable state."""
import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from common import PROJECT, atomic, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    m = json.loads(args.manifest.read_text())
    out = args.manifest.parent
    try:
        assert not (out/'complete.json').exists(), 'Already complete'
        assert sha(m['config']) == m['config_sha256']
        for path,digest in m['code_sha256'].items():
            assert sha(path)==digest, f'Frozen code changed: {path}'
        if args.resume:
            assert (out/'resume.pt').exists() and (out/'preflight.json').exists()
        else:
            assert not (out/'resume.pt').exists(), 'Use --resume'
        env = {**os.environ, **m['environment_variables'], 'CUDA_VISIBLE_DEVICES':m['gpus']}
        for mode in (['train'] if args.resume else ['preflight','train']):
            cmd = [m['python'],'-m','torch.distributed.run','--standalone',
                   f"--nproc_per_node={m['world_size']}",m['worker'],'--config',m['config'],'--mode',mode]
            if args.resume:
                cmd.append('--resume')
            started = time.time()
            with (out/f'{mode}.log').open('a',buffering=1) as log:
                child = subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                while True:
                    code = child.poll()
                    path = out/('preflight.progress.json' if mode=='preflight' else 'progress.json')
                    progress = json.loads(path.read_text()) if path.exists() else {}
                    atomic(out/'status.json',{'phase':mode,'gpus':m['gpus'],'pid':child.pid,
                        'command':cmd,'progress':progress,'started_unix':started,
                        'updated_unix':time.time(),'exit_code':code})
                    if code is not None:
                        if code:
                            raise RuntimeError(f'{mode} exited {code}; see {out}/{mode}.log')
                        break
                    stale = time.time()-max(started,progress.get('updated_unix',started))
                    if stale>600:
                        atomic(out/'alerts.json',{'error':'No progress for 10 minutes','stale_seconds':stale,
                                                 'pid':child.pid,'updated_unix':time.time()})
                    if stale>1800:
                        os.killpg(child.pid,signal.SIGTERM)
                        raise RuntimeError('Worker stopped after 30-minute stall; resume checkpoint preserved')
                    time.sleep(5)
            if mode=='preflight':
                proof = json.loads((out/'preflight.json').read_text())
                assert proof['status']=='passed' and proof['optimizer_updates']==0
        done = json.loads((out/'complete.json').read_text())
        assert done['sample_visits']==200000 and done['steps']==3125
        atomic(out/'status.json',{'phase':'complete','gpus':m['gpus'],'updated_unix':time.time(),
                                 'exit_code':0,'complete':done})
    except BaseException as exc:
        failure = {'phase':'failed','error':str(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()}
        atomic(out/'alerts.json',failure)
        atomic(out/'status.json',failure)
        raise


if __name__=='__main__':
    main()
