"""Persistent GPU-pair queues with explicit completion and release dependencies."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from common import PROJECT, atomic


def run(root, gpus):
    root = root.resolve()
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    queue = manifest['gpu_queues'][gpus]
    lock = (root / ('queue_' + gpus.replace(',', '') + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root / ('queue_' + gpus.replace(',', '') + '.json')
    dependency = manifest.get('queue_dependencies', {}).get(gpus)
    while True:
        if dependency:
            path = Path(dependency)
            d = json.loads(path.read_text()) if path.exists() else {}
            if d.get('phase') == 'complete' and d.get('exit_code') == 0:
                assert (path.parent / 'complete.json').exists()
            else:
                if d.get('phase') in ('failed', 'cancelled_for_efficiency_fix'):
                    atomic(status, {'phase': 'dependency_failed', 'dependency': dependency})
                    raise RuntimeError(f'Dependency requires review: {dependency}')
                atomic(status, {'phase': 'waiting_for_dependency', 'dependency': dependency,
                                'gpus': gpus, 'updated_unix': time.time()})
                time.sleep(10)
                continue
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        memory = {x.strip(): int(y.strip()) for x, y in (line.split(',') for line in raw.splitlines())}
        if all(memory[g] < 1024 for g in gpus.split(',')):
            break
        atomic(status, {'phase': 'waiting_for_memory_release', 'gpus': gpus,
                        'memory_mib': memory, 'updated_unix': time.time()})
        time.sleep(10)
    for arm in queue:
        out = root / arm
        job = json.loads((out / 'experiment_manifest.json').read_text())
        assert job['gpus'] == gpus
        if (out / 'complete.json').exists():
            assert json.loads((out / 'status.json').read_text())['phase'] == 'complete'
            continue
        command = list(job['launch_command'])
        if (out / 'resume.pt').exists():
            command.append('--resume')
        atomic(status, {'phase': 'running', 'arm': arm, 'gpus': gpus,
                        'updated_unix': time.time(), 'command': command})
        with (out / 'supervisor.log').open('a', buffering=1) as log:
            code = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            atomic(status, {'phase': 'failed', 'arm': arm, 'exit_code': code,
                            'updated_unix': time.time()})
            raise SystemExit(code)
    atomic(status, {'phase': 'complete', 'gpus': gpus, 'completed_arms': queue,
                    'updated_unix': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    run(args.root, args.gpus)
