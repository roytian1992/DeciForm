"""Wait for two released GPUs, then evaluate four fixed checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT, atomic, sha


def gpu_memory():
    result = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    return {i.strip():int(memory.strip()) for i,memory in
            (line.split(',') for line in result.strip().splitlines())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    lock=(root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    queues=manifest['gpu_queues']
    while True:
        path=Path(manifest['prerequisite_status'])
        prerequisite=json.loads(path.read_text()) if path.exists() else {}
        if prerequisite.get('phase')=='failed':
            raise RuntimeError('Prerequisite A2 failed; order audit remains unstarted')
        memory=gpu_memory()
        released=all(memory[gpu]<10000 for gpu in queues)
        if prerequisite.get('phase')=='complete' and prerequisite.get('exit_code')==0 and released:
            break
        atomic(root/'status.json',{'phase':'waiting_for_gpus','updated_unix':time.time(),
            'prerequisite_phase':prerequisite.get('phase'),'gpus':list(queues),
            'memory_mib':{gpu:memory[gpu] for gpu in queues}})
        time.sleep(30)

    def run_queue(gpu,names):
        for name in names:
            out=root/name
            if (out/'complete.json').exists():
                if args.resume:
                    continue
                raise FileExistsError('Use --resume to continue an existing audit')
            for path,digest in manifest['code_sha256'].items():
                assert sha(PROJECT/path)==digest,path
            command=[manifest['python'],str(PROJECT/'scripts/formal/worker_choice_order_audit.py'),
                     '--root',str(root),'--model',name]
            if args.resume and (out/'predictions.jsonl').exists():
                command.append('--resume')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4',
                 'TOKENIZERS_PARALLELISM':'false','FLA_DISABLE_TENSOR_CACHE':'1',
                 'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            with (out/'worker.log').open('a',buffering=1) as log:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
                started=time.time()
                while True:
                    code=child.poll()
                    progress_file=out/'progress.json'
                    progress=json.loads(progress_file.read_text()) if progress_file.exists() else {}
                    atomic(root/f'queue_gpu{gpu}.json',{'phase':'running' if code is None else 'worker_exited',
                        'gpu':gpu,'model':name,'pid':child.pid,'exit_code':code,
                        'updated_unix':time.time(),'progress':progress,'command':command})
                    if code is not None:
                        if code:
                            raise RuntimeError(f'{name} exited {code}; inspect {out}/worker.log')
                        break
                    if time.time()-max(started,progress.get('updated_unix',started))>900:
                        atomic(out/'alerts.json',{'error':'No evaluation progress for 15 minutes',
                                                  'pid':child.pid,'updated_unix':time.time()})
                    time.sleep(5)
            complete=json.loads((out/'complete.json').read_text())
            assert complete['evaluated_predictions']==manifest['models'][name]['expected_predictions']
        atomic(root/f'queue_gpu{gpu}.json',{'phase':'complete','gpu':gpu,'models':names})
        return names

    failures={}
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures={pool.submit(run_queue,gpu,names):gpu for gpu,names in queues.items()}
        remaining=set(futures)
        while remaining:
            for future in list(remaining):
                if future.done():
                    remaining.remove(future)
                    try:
                        future.result()
                    except Exception as exc:
                        failures[futures[future]]=str(exc)
                        atomic(root/'alerts.json',{'failures':failures,'updated_unix':time.time()})
            atomic(root/'status.json',{'phase':'evaluating' if remaining else 'summarizing',
                'active_gpu_queues':[futures[f] for f in remaining],
                'failures':failures,'updated_unix':time.time()})
            if remaining:
                time.sleep(10)
    if failures:
        raise RuntimeError(str(failures))
    reports={name:json.loads((root/name/'report.json').read_text()) for name in manifest['models']}
    atomic(root/'comparison.json',{'status':'complete','protocol_sha256':manifest['protocol_sha256'],
        'models':reports,'report_sha256':{name:sha(root/name/'report.json') for name in reports},
        'notes':['This completion evaluates Choice only; no Score flip-rate claim.',
                 'Every native candidate is retained in all five orders.',
                 'Candidate IDs remain attached to semantic content; no relabeling.',
                 'Noul is reused from the preceding frozen audit, not inferred from Choice.',
                 'Five orders do not exhaust all permutations or balance every candidate position.',
                 'This experiment measures fixed-checkpoint robustness; it does not establish training-seed stability.']})
    subprocess.run([manifest['python'],str(PROJECT/'scripts/formal/report_classification_order.py')],
                   cwd=PROJECT,check=True)
    atomic(root/'status.json',{'phase':'complete','updated_unix':time.time(),
        'models':list(reports),'comparison_sha256':sha(root/'comparison.json')})


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        import sys
        try:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            atomic(root/'alerts.json',{'error':str(exc),'traceback':traceback.format_exc(),
                                      'updated_unix':time.time()})
            atomic(root/'status.json',{'phase':'failed','error':str(exc),'updated_unix':time.time()})
        except Exception:
            pass
        raise
