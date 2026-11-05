"""Preflight, run and audit one independent-D2 ablation with recovery state."""
import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from common import PROJECT, atomic, sha


def run_mode(manifest, root, mode, resume):
    config = Path(manifest['config'])
    env = {**os.environ, **manifest['environment_variables'], 'CUDA_VISIBLE_DEVICES': manifest['gpus']}
    cmd = [manifest['python'], '-m', 'torch.distributed.run', '--standalone',
           f"--nproc_per_node={manifest['world_size']}", manifest['worker'],
           '--config', str(config), '--mode', mode]
    if resume:
        cmd.append('--resume')
    logfile = root / f'{mode}.log'
    with logfile.open('a', buffering=1) as log:
        child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        started = time.time()
        while True:
            code = child.poll()
            progress_path = root / ('preflight.progress.json' if mode == 'preflight' else 'progress.json')
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            atomic(root / 'status.json', {'phase': mode, 'arm': manifest['experiment_name'],
                'gpus': manifest['gpus'], 'pid': child.pid, 'command': cmd,
                'started_unix': started, 'updated_unix': time.time(), 'progress': progress,
                'exit_code': code, 'log': str(logfile)})
            if code is not None:
                if code:
                    raise RuntimeError(f'{mode} exited {code}; see {logfile}')
                return
            stale = time.time() - max(started, progress.get('updated_unix', started))
            if stale > 900:
                atomic(root / 'alerts.json', {'time': time.time(), 'error': 'No worker progress for over 15 minutes',
                                               'seconds': stale, 'pid': child.pid, 'log': str(logfile)})
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    try:
        config = Path(manifest['config'])
        assert sha(config) == manifest['config_sha256']
        assert not (root / 'complete.json').exists(), 'Run is already complete'
        if args.resume:
            assert (root / 'resume.pt').exists()
            assert json.loads((root / 'preflight_passed.json').read_text())['status'] == 'passed'
        else:
            assert not (root / 'resume.pt').exists(), 'Use --resume for recovery'
        for path, digest in manifest['code_sha256'].items():
            assert sha(PROJECT / path) == digest, f'Code changed since manifest: {path}'
        run_mode(manifest, root, 'train' if args.resume else 'preflight', args.resume)
        if not args.resume:
            proof = json.loads((root / 'preflight.json').read_text())
            assert proof['status'] == 'passed' and proof['optimizer_updates'] == 0
            baseline = json.loads(Path(manifest['baseline_initialization']).read_text())
            for key in ('lora_sha256', 'pointer_sha256', 'head_sha256', 'base_config_sha256'):
                assert proof['initialization'][key] == baseline[key], key
            assert proof['checkpoint_policy']['policy'] == 'half'
            assert proof['ablation']['ablation'] == manifest['ablation']
            assert proof['ablation']['trainable_parameters'] == manifest['expected_trainable_parameters']
            assert proof['distribution_objective'] == manifest['distribution_objective']
            assert proof['evaluation_protocol'] == manifest['evaluation_protocol']
            atomic(root / 'preflight_passed.json', {
                'status': 'passed', 'same_fresh_initialization_as_d2': True,
                'checkpoint_policy': 'half', 'config_sha256': sha(config), 'proof': proof})
            run_mode(manifest, root, 'train', False)
        complete = json.loads((root / 'complete.json').read_text())
        cfg = json.loads(config.read_text())
        assert complete['steps'] == cfg['total_steps'] and complete['sample_visits'] == cfg['train_rows']
        assert complete['evaluated_splits'] == ['dev', 'test_in_domain']
        atomic(root / 'status.json', {'phase': 'complete', 'arm': manifest['experiment_name'],
            'gpus': manifest['gpus'], 'updated_unix': time.time(), 'exit_code': 0, 'complete': complete})
    except Exception as exc:
        atomic(root / 'alerts.json', {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        atomic(root / 'status.json', {'phase': 'failed', 'error': str(exc), 'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
