"""Immutable, measured RQ-Score launches with architecture-aware preflight checks."""
import argparse
import json
import os
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
    m = json.loads(args.manifest.read_text())
    out = args.manifest.resolve().parent
    try:
        assert not (out / 'cancellation.json').exists()
        assert not (out / 'complete.json').exists()
        cfg = json.loads(Path(m['config']).read_text())
        if args.resume:
            assert (out / 'resume.pt').exists() and (out / 'preflight_passed.json').exists()
        else:
            assert not (out / 'resume.pt').exists()
        speed = json.loads(Path(m['execution_benchmark']).read_text())
        assert sha(m['execution_benchmark']) == m['execution_benchmark_sha256']
        assert speed['status'] == 'passed' and speed['optimizer_updates'] == 0
        assert speed['variant'] == cfg['score_architecture']
        env = {**os.environ, **m['environment_variables'], 'CUDA_VISIBLE_DEVICES': m['gpus']}
        for mode in (['train'] if args.resume else ['preflight', 'train']):
            assert sha(m['config']) == m['config_sha256']
            for path, digest in m['code_sha256'].items():
                assert sha(PROJECT / path) == digest, path
            command = [m['python'], '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=2', m['worker'], '--config', m['config'], '--mode', mode]
            if args.resume:
                command.append('--resume')
            start = time.time()
            with (out / f'{mode}.log').open('a', buffering=1) as log:
                child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    code = child.poll()
                    p = out / ('preflight.progress.json' if mode == 'preflight' else 'progress.json')
                    progress = json.loads(p.read_text()) if p.exists() else {}
                    atomic(out / 'status.json', {'phase': mode, 'gpus': m['gpus'], 'pid': child.pid,
                        'updated_unix': time.time(), 'started_unix': start, 'command': command,
                        'exit_code': code, 'progress': progress})
                    if code is not None:
                        if code:
                            raise RuntimeError(f'{mode} exited {code}; inspect {mode}.log')
                        break
                    if time.time()-max(start, progress.get('updated_unix', start)) > 900:
                        atomic(out / 'alerts.json', {'error': 'No progress for 15 minutes',
                               'pid': child.pid, 'updated_unix': time.time()})
                    time.sleep(5)
            if mode == 'preflight':
                proof = json.loads((out / 'preflight.json').read_text())
                baseline = json.loads(Path(m['baseline_initialization']).read_text())
                assert proof['status'] == 'passed' and proof['optimizer_updates'] == 0
                for key in ['lora_sha256', 'pointer_sha256', 'base_config_sha256']:
                    assert proof['initialization'][key] == baseline[key], key
                observed = (proof['initialization']['head_sha256'] if cfg['score_architecture'] == 'd2_batched'
                            else proof['shared_head_sha256_before_replacement'])
                assert observed == baseline['head_sha256']
                assert proof['ablation']['trainable_parameters'] == m['expected_trainable_parameters']
                assert proof['checkpoint_policy']['policy'] == cfg['checkpoint_policy']
                assert proof['distribution_objective'] == cfg['distribution_objective']
                assert proof['evaluation_protocol'] == cfg['evaluation_protocol']
                atomic(out / 'preflight_passed.json', {'status': 'passed', 'proof': proof,
                    'shared_fresh_d2_initialization_verified': True,
                    'new_score_module': cfg['score_architecture'] == 'numeric_cdf'})
        complete = json.loads((out / 'complete.json').read_text())
        assert complete['sample_visits'] == cfg['train_rows'] and complete['steps'] == cfg['total_steps']
        assert complete['evaluated_splits'] == ['dev', 'test_in_domain']
        atomic(out / 'status.json', {'phase': 'complete', 'exit_code': 0,
                                    'gpus': m['gpus'], 'updated_unix': time.time()})
    except BaseException as exc:
        atomic(out / 'alerts.json', {'error': str(exc), 'traceback': traceback.format_exc(),
                                    'updated_unix': time.time()})
        atomic(out / 'status.json', {'phase': 'failed', 'error': str(exc), 'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
