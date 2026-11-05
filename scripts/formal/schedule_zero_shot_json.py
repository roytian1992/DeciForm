"""Persistent inference queue; acquire only successful, released Score GPUs."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT, atomic, sha
from zero_shot_protocol import PANELS, summarize


def memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                  '--format=csv,noheader,nounits'], text=True)
    return {g.strip(): int(m.strip()) for g, m in (line.split(',') for line in raw.splitlines())}


def reports(root, manifest):
    completed = {}
    for model in manifest['models']:
        out = root / model
        for panel in PANELS:
            report_path = out / f'{panel}.json'
            if report_path.exists(): continue
            dirs = [out / f'shard_{i}' for i in range(manifest['shards_per_model'])]
            if not all((d / f'{panel}.json').exists() for d in dirs): continue
            records, evidence = [], {}
            for d in dirs:
                path = d / f'{panel}.predictions.jsonl'
                digest = sha(path)
                assert json.loads((d / f'{panel}.json').read_text())['predictions_sha256'] == digest
                evidence[str(path)] = digest
                records.extend(json.loads(line) for line in path.open())
            assert len(records) == manifest['panels'][panel]['predictions']
            assert len({(r['id'], r['view']) for r in records}) == len(records)
            atomic(report_path, {'status': 'complete', 'model': model, 'panel': panel,
                'predictions': len(records), 'rows': manifest['panels'][panel]['rows'],
                'report': summarize(panel, records), 'prediction_files_sha256': evidence,
                'model_files_sha256': manifest['models'][model]['files_sha256'],
                'optimizer_updates': 0})
        completed[model] = [p for p in PANELS if (out / f'{p}.json').exists()]
    return completed


def run(root):
    if (root / 'cancellation.json').exists():
        raise RuntimeError('This queue was explicitly cancelled; use its documented replacement')
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    for path, digest in manifest['code_sha256'].items(): assert sha(PROJECT / path) == digest, path
    jobs = [(model, i) for i in range(manifest['shards_per_model']) for model in manifest['models']]
    active, handles, finished = {}, [], {}
    # Refuse to launch duplicate workers after a scheduler interruption.
    for model, shard in jobs:
        out = root / model / f'shard_{shard}'
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'complete.json').exists(): finished[(model, shard)] = 0
        elif (out / 'allocation.json').exists():
            alloc = json.loads((out / 'allocation.json').read_text())
            proc = Path(f"/proc/{alloc['pid']}/cmdline")
            if proc.exists() and str(root).encode() in proc.read_bytes():
                raise RuntimeError(f"Existing worker still active: {alloc['pid']}; do not duplicate it")
            if (out / 'failure.json').exists():
                raise RuntimeError(f'Previous failed worker requires review: {out}')
    pending = [job for job in jobs if job not in finished]
    while pending or active:
        for job, desc in list(active.items()):
            code = desc['child'].poll()
            if code is not None:
                done = root / job[0] / f'shard_{job[1]}' / 'complete.json'
                finished[job] = code if code else (0 if done.exists() else -1)
                del active[job]
        failed = any(code != 0 for code in finished.values())
        if not failed:
            used = memory()
            reserved = {d['gpu'] for d in active.values()}
            for gpu, dependency in manifest['gpu_release_dependencies'].items():
                if not pending or gpu in reserved: continue
                dep = Path(dependency)
                if not dep.exists(): continue
                state = json.loads(dep.read_text())
                if state.get('phase') != 'complete' or state.get('exit_code') != 0: continue
                if not (dep.parent / 'complete.json').exists(): continue
                if used[gpu] >= manifest['max_memory_mib_before_launch']: continue
                model, shard = pending.pop(0)
                out = root / model / f'shard_{shard}'
                cmd = [manifest['python'], manifest['worker'], '--root', str(root),
                       '--model', model, '--shard', str(shard)]
                log = (out / 'worker.log').open('a', buffering=1)
                handles.append(log)
                env = {**os.environ, **manifest['environment_variables'], 'CUDA_VISIBLE_DEVICES': gpu}
                child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                atomic(out / 'allocation.json', {'gpu': gpu, 'pid': child.pid, 'command': cmd,
                       'started_unix': time.time(), 'dependency': dependency,
                       'dependency_status_sha256': sha(dep)})
                active[(model, shard)] = {'gpu': gpu, 'child': child}
                reserved.add(gpu)
        available_reports = reports(root, manifest)
        atomic(root / 'status.json', {'phase': 'failed' if failed else 'running' if active else 'waiting_for_gpu',
            'updated_unix': time.time(), 'pending': [{'model': m, 'shard': s} for m, s in pending],
            'active': {f'{m}/{s}': {'pid': d['child'].pid, 'gpu': d['gpu']}
                       for (m, s), d in active.items()},
            'completed_exit_codes': {f'{m}/{s}': code for (m, s), code in finished.items()},
            'available_reports': available_reports})
        if failed:
            atomic(root / 'alerts.json', {'updated_unix': time.time(), 'error': 'Inference worker failed; inspect shard failure.json. Other active workers may finish; new launches are stopped.'})
            if not active: raise RuntimeError('Inference failed; inspect saved failure evidence')
        if pending or active: time.sleep(10)
    completed = reports(root, manifest)
    assert all(len(panels) == len(PANELS) for panels in completed.values())
    atomic(root / 'complete.json', {'status': 'complete', 'models': completed, 'optimizer_updates': 0,
           'predictions_per_model': manifest['expected_predictions_per_model'], 'finished_unix': time.time()})
    atomic(root / 'status.json', {'phase': 'complete', 'updated_unix': time.time(), 'models': completed, 'exit_code': 0})
    for handle in handles: handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.root.resolve())
    except BaseException as exc:
        atomic(args.root / 'alerts.json', {'updated_unix': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
