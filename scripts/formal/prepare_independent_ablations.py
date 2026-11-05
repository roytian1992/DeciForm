"""Freeze three fresh-base ablations against the completed independent D2 run."""
import copy
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha


FULL = PROJECT / 'runs/formal_independent_20261105_v1'
ROOT = PROJECT / 'runs/formal_independent_ablations_20261105_v1'
ARMS = {
    'A1_no_task_routing': {'task_routing': False, 'quadratic': True, 'geometry': True, 'auxiliary_losses': False},
    'A2_no_quadratic': {'task_routing': True, 'quadratic': False, 'geometry': True, 'auxiliary_losses': False},
    'A3_no_head_geometry': {'task_routing': True, 'quadratic': True, 'geometry': False, 'auxiliary_losses': False},
}
COUNTS = {
    'A1_no_task_routing': 46113798,
    'A2_no_quadratic': 46025670,
    'A3_no_head_geometry': 46107654,
}
GPUS = {'A1_no_task_routing': '2,3', 'A2_no_quadratic': '4,5', 'A3_no_head_geometry': '6,7'}


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    full_manifest = json.loads((FULL / 'experiment_manifest.json').read_text())
    full_config = json.loads((FULL / 'config.json').read_text())
    complete = json.loads((FULL / 'complete.json').read_text())
    assert complete['steps'] == 3125 and complete['sample_visits'] == 200000
    assert full_config['initialization'] == 'fresh_base_only'
    assert full_config['ablation'] == {'task_routing': True, 'quadratic': True,
                                       'geometry': True, 'auxiliary_losses': False}
    assert full_config['distribution_objective']['kind'] == 'ordinal_main'
    assert sha(FULL / 'config.json') == full_manifest['config_sha256']
    for path, digest in full_manifest['code_sha256'].items():
        assert sha(PROJECT / path) == digest, f'Frozen D2 source changed: {path}'

    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    ROOT.mkdir()
    additions = (
        'scripts/formal/models_independent_ablation.py',
        'scripts/formal/worker_independent_ablation.py',
        'scripts/formal/prepare_independent_ablations.py',
        'scripts/formal/schedule_independent_ablations.py',
        'scripts/formal/supervise_independent_ablation.py',
    )
    hashes = dict(full_manifest['code_sha256'])
    hashes.update({path: sha(PROJECT / path) for path in additions})
    for path, digest in hashes.items():
        snapshot = ROOT / 'code' / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, snapshot)
        assert sha(snapshot) == digest

    arms = {}
    for name, flags in ARMS.items():
        output = ROOT / name
        output.mkdir()
        config = copy.deepcopy(full_config)
        config.update(
            output=str(output), method_variant=name, experiment_name=name,
            label=f'Independent D2 architecture ablation: {name}',
            ablation=flags, reference_run=str(FULL),
            worker_variant=str(PROJECT / 'scripts/formal/worker_independent_ablation.py'),
            world_size=2, total_steps=3125,
            order_audit='full-native test Choice/Noul, original and reverse; Score excluded',
        )
        # Exactly these fields differ from the frozen D2 protocol.
        changed = {k for k in config if config[k] != full_config.get(k)}
        assert changed <= {'output', 'method_variant', 'experiment_name', 'label',
                           'ablation', 'reference_run', 'worker_variant', 'world_size',
                           'total_steps', 'order_audit'}
        atomic(output / 'config.json', config)
        manifest = copy.deepcopy(full_manifest)
        supervisor = str(PROJECT / 'scripts/formal/supervise_independent_ablation.py')
        manifest.update(
            created=created, python=sys.executable, python_version=platform.python_version(),
            packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'triton')},
            config=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
            worker=config['worker_variant'], supervisor=supervisor, code_sha256=hashes,
            code_snapshot=str(ROOT / 'code'), experiment_name=name, ablation=flags,
            expected_trainable_parameters=COUNTS[name], baseline_initialization=str(FULL / 'initialization.json'),
            reference_run=str(FULL), parent_checkpoint=None, gpus=GPUS[name], world_size=2,
            tmux_session='deciform_independent_ablation_' + name,
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
            comparison='Identical independent D2 data, objective, seed, fixed negative IDs, global batches and full-native evaluation; fresh base only; exactly one architecture component is disabled.',
            initialization='fresh_base_only', candidate_encoder=full_config['candidate_encoder'],
            data_counts={'train': 200000, 'dev': 6000, 'test_in_domain': 15000},
            cache_audit_sha256=full_config['cache_audit_sha256'], changed_reference_config_fields=sorted(changed),
        )
        manifest['launch_command'] = [sys.executable, supervisor, '--manifest', str(output / 'experiment_manifest.json')]
        manifest['resume_command'] = [sys.executable, supervisor, '--manifest', str(output / 'experiment_manifest.json'), '--resume']
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': GPUS[name], 'world_size': 2, 'manifest': str(output / 'experiment_manifest.json'),
                      'ablation': flags, 'expected_trainable_parameters': COUNTS[name], 'state': 'prepared'}

    atomic(ROOT / 'experiment_manifest.json', {
        'created': created, 'status': 'prepared', 'full_reference': str(FULL),
        'arms': arms, 'gpu_queues': {v: [k] for k, v in GPUS.items()},
        'train_rows_per_arm': 200000, 'epochs': 1, 'steps_per_arm': 3125, 'world_size_per_arm': 2,
        'evaluation_splits': ['dev', 'test_in_domain'], 'evaluation_candidates': 'complete_native',
        'evaluation_protocol': full_config['evaluation_protocol'], 'score_decision_rule': full_config['score_decision_rule'],
        'distribution_objective': full_config['distribution_objective'], 'initialization': 'fresh_base_only',
        'data': full_config['data'], 'cache_audit_sha256': full_config['cache_audit_sha256'], 'code_sha256': hashes,
        'note': 'Three arms run concurrently on GPU pairs 2-3, 4-5 and 6-7; GPU 0-1 are reserved for the D2 position audit.',
    })
    print(ROOT)


if __name__ == '__main__':
    main()
