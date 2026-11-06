"""Audited score distribution comparison: fixed preflight, fresh training, internal evaluation."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import traceback

from common import PROJECT, atomic, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    try:
        config = Path(manifest['config'])
        cfg = json.loads(config.read_text())
        env = {**os.environ, **manifest['environment_variables'],
               'CUDA_VISIBLE_DEVICES': manifest['gpus']}
        if args.resume:
            assert (root / 'resume.pt').exists()
            assert json.loads((root / 'preflight_passed.json').read_text())['status'] == 'passed'
        else:
            assert not (root / 'resume.pt').exists(), 'Use --resume for recovery'
        assert not (root / 'complete.json').exists(), 'Run is already complete'
        for mode in (('train',) if args.resume else ('preflight', 'train')):
            assert sha(config) == manifest['config_sha256'], 'Config changed since freeze'
            for name, digest in manifest['code_sha256'].items():
                if sha(PROJECT / name) != digest:
                    raise RuntimeError(f'Code changed since manifest: {name}')
            cmd = [manifest['python'], '-m', 'torch.distributed.run', '--standalone',
                   f"--nproc_per_node={manifest['world_size']}", manifest['worker'],
                   '--config', str(config), '--mode', mode]
            if args.resume:
                cmd.append('--resume')
            logfile = root / f'{mode}.log'
            with logfile.open('a', buffering=1) as log:
                child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                started = time.time()
                while True:
                    code = child.poll()
                    path = root / ('preflight.progress.json' if mode == 'preflight' else 'progress.json')
                    progress = json.loads(path.read_text()) if path.exists() else {}
                    state = {'phase': mode, 'arm': manifest['experiment_name'],
                             'gpus': manifest['gpus'], 'pid': child.pid, 'command': cmd,
                             'started_unix': started, 'updated_unix': time.time(),
                             'progress': progress, 'exit_code': code, 'log': str(logfile)}
                    if code is not None:
                        if code:
                            raise RuntimeError(f'{mode} exited {code}; see {logfile}')
                        state['phase'] = mode + '_complete'
                        atomic(root / 'status.json', state)
                        break
                    stale = time.time() - max(started, progress.get('updated_unix', started))
                    if stale > 900:
                        atomic(root / 'alerts.json', {'time': time.time(),
                               'error': 'No worker progress for over 15 minutes',
                               'seconds': stale, 'pid': child.pid, 'log': str(logfile)})
                    if stale>1800:
                        os.killpg(child.pid,signal.SIGTERM)
                        raise RuntimeError('Worker stopped after 30-minute stall; own resume checkpoint preserved')
                    atomic(root / 'status.json', state)
                    time.sleep(5)
            if mode == 'preflight':
                proof = json.loads((root / 'preflight.json').read_text())
                assert proof['status'] == 'passed' and proof['optimizer_updates'] == 0
                assert sha(manifest['baseline_initialization'])==manifest['baseline_initialization_sha256']
                baseline = json.loads(Path(manifest['baseline_initialization']).read_text())
                for key in ('lora_sha256', 'pointer_sha256', 'head_sha256', 'base_config_sha256'):
                    assert proof['initialization'][key] == baseline[key], key
                assert proof['checkpoint_policy']['policy'] == cfg.get('checkpoint_policy', 'half')
                assert proof['ablation']['ablation'] == manifest['ablation']
                assert proof['distribution_objective'] == manifest['distribution_objective']
                assert cfg['rlcd_tasks'] == manifest['rlcd_tasks']
                assert json.loads((root/'preflight.classification_objective.json').read_text()) == cfg['classification_objective'] == manifest['classification_objective']
                assert proof['evaluation_protocol'] == manifest['evaluation_protocol']
                assert proof['ablation']['trainable_parameters'] == manifest['expected_trainable_parameters']
                atomic(root / 'preflight_passed.json', {
                    'status': 'passed', 'same_initial_lora_head_pointer_as_d2_fresh_initialization': True,
                    'checkpoint_policy': cfg.get('checkpoint_policy', 'half'),
                    'config_sha256': sha(config), 'proof': proof})
        complete = json.loads((root / 'complete.json').read_text())
        cfg = json.loads(config.read_text())
        assert complete['steps'] == cfg['total_steps'] and complete['sample_visits'] == cfg['train_rows']
        assert complete['evaluated_splits'] == ['dev', 'test_in_domain']
        atomic(root / 'status.json', {'phase': 'complete', 'arm': manifest['experiment_name'],
                                     'gpus': manifest['gpus'], 'updated_unix': time.time(),
                                     'exit_code': 0, 'complete': complete})
        return 0
    except Exception as exc:
        atomic(root / 'alerts.json', {'time': time.time(), 'error': str(exc),
                                     'traceback': traceback.format_exc()})
        atomic(root / 'status.json', {'phase': 'failed', 'error': str(exc),
                                     'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
