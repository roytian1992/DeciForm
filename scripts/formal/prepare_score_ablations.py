"""Freeze single-component ablations against the completed D2 full run."""

import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from models_score_ablation import ABLATIONS


ROOT = PROJECT / 'runs/formal_d2_ablations_20261104_v1'
FULL = PROJECT / 'runs/formal_score_distribution_20261104_v1/d2_ordinal_main'
COUNTS = {'a1_no_task_routing': 46113798,
          'a2_no_quadratic': 46025670,
          'a3_no_head_geometry': 46107654}
GPU_QUEUES = {'0,1,2,3': ['a1_no_task_routing', 'a3_no_head_geometry'],
              '4,5,6,7': ['a2_no_quadratic']}


def main():
    full_manifest = json.loads((FULL / 'experiment_manifest.json').read_text())
    full_config = json.loads((FULL / 'config.json').read_text())
    complete = json.loads((FULL / 'complete.json').read_text())
    status = json.loads((FULL / 'status.json').read_text())
    assert complete['steps'] == 3125 and complete['sample_visits'] == 200000
    assert status['phase'] == 'complete' and status['exit_code'] == 0
    assert full_manifest['world_size'] == 4 and full_config['world_size'] == 4
    assert full_config['ablation'] == {k: True for k in ('task_routing', 'quadratic', 'geometry')} | {'auxiliary_losses': False}
    assert full_config['distribution_objective']['kind'] == 'ordinal_main'
    assert sha(FULL / 'config.json') == full_manifest['config_sha256']
    for path, digest in full_manifest['code_sha256'].items():
        assert sha(PROJECT / path) == digest, f'Frozen D2 source changed: {path}'

    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    ROOT.mkdir(exist_ok=False)
    additions = ('scripts/formal/models_score_ablation.py',
                 'scripts/formal/worker_score_ablation.py',
                 'scripts/formal/prepare_score_ablations.py',
                 'scripts/formal/schedule_score_ablations.py',
                 'tests/test_score_ablations.py')
    hashes = dict(full_manifest['code_sha256'])
    hashes.update({path: sha(PROJECT / path) for path in additions})
    for path, digest in hashes.items():
        snapshot = ROOT / 'code' / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, snapshot)
        assert sha(snapshot) == digest

    references = ('config.json', 'initialization.json', 'complete.json',
                  'epoch_1.pt', 'evaluations/epoch_1.dev.json',
                  'evaluations/epoch_1.test_in_domain.json',
                  'evaluations/epoch_1.test_in_domain.score_rules.json')
    atomic(ROOT / 'full_reference.json', {'run': str(FULL),
          'checkpoint_loaded_for_initialization': False,
          'files_sha256': {str(FULL / name): sha(FULL / name) for name in references},
          'protocol_sha256': sha(PROJECT / 'runs/formal_score_distribution_20261104_v1/evaluation_protocol.json')})

    inverse_gpus = {arm: gpus for gpus, arms in GPU_QUEUES.items() for arm in arms}
    arms = {}
    for name, flags in ABLATIONS.items():
        output = ROOT / name
        output.mkdir()
        config = copy.deepcopy(full_config)
        config.update(output=str(output), method_variant=name, experiment_name=name,
                      label=f'D2 architecture ablation: {name}', ablation=flags,
                      reference_run=str(FULL),
                      worker_variant=str(PROJECT / 'scripts/formal/worker_score_ablation.py'))
        assert {key: value for key, value in config.items() if key not in
                {'output', 'method_variant', 'experiment_name', 'label', 'ablation',
                 'reference_run', 'worker_variant'}} == {
                 key: value for key, value in full_config.items() if key not in
                 {'output', 'method_variant', 'experiment_name', 'label', 'ablation',
                  'reference_run', 'worker_variant'}}
        atomic(output / 'config.json', config)
        manifest = copy.deepcopy(full_manifest)
        manifest.update(created=created, gpus=inverse_gpus[name], world_size=4,
                        config=str(output / 'config.json'),
                        config_sha256=sha(output / 'config.json'),
                        worker=config['worker_variant'], code_sha256=hashes,
                        code_snapshot=str(ROOT / 'code'), experiment_name=name,
                        ablation=flags, expected_trainable_parameters=COUNTS[name],
                        baseline_initialization=str(FULL / 'initialization.json'),
                        reference_run=str(FULL), parent_checkpoint=None,
                        tmux_session='deciform_d2_ablation_' + name,
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                            cwd=PROJECT, text=True).strip(),
                        comparison='Identical D2 data, objective, 4-GPU world size, seed, full-native evaluation and fresh-base initialization; only one architecture component is disabled.')
        manifest['launch_command'] = [sys.executable, manifest['supervisor'],
                                      '--manifest', str(output / 'experiment_manifest.json')]
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': inverse_gpus[name], 'ablation': flags,
                      'manifest': str(output / 'experiment_manifest.json'),
                      'expected_trainable_parameters': COUNTS[name],
                      'state': 'prepared'}
    queue_commands = {gpus: [sys.executable,
        str(PROJECT / 'scripts/formal/schedule_score_ablations.py'),
        '--root', str(ROOT), '--arms', *names]
        for gpus, names in GPU_QUEUES.items()}
    atomic(ROOT / 'experiment_manifest.json', {
        'created': created, 'status': 'prepared', 'full_reference': str(ROOT / 'full_reference.json'),
        'arms': arms, 'gpu_queues': GPU_QUEUES, 'queue_commands': queue_commands,
        'train_rows_per_arm': 200000, 'epochs': 1, 'steps_per_arm': 3125,
        'world_size_per_arm': 4, 'evaluation_splits': ['dev', 'test_in_domain'],
        'evaluation_candidates': 'complete_native',
        'evaluation_protocol': full_config['evaluation_protocol'],
        'score_decision_rule': full_config['score_decision_rule'],
        'distribution_objective': full_config['distribution_objective'],
        'initialization': 'fresh_base_only',
        'data': full_config['data'], 'cache_audit_sha256': full_config['cache_audit_sha256'],
        'code_sha256': hashes})
    print(ROOT)


if __name__ == '__main__':
    main()
