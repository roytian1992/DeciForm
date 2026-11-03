"""Freeze isolated choice/score mechanisms with two four-GPU training groups."""
import copy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from decision_mechanisms import Mechanism, extra_parameters


def main():
    reference = PROJECT / 'runs/formal_v6_ablations_20261103_v1/a4_ce_only'
    previous = json.loads((reference / 'experiment_manifest.json').read_text())
    base_config = json.loads((reference / 'config.json').read_text())
    complete = json.loads((reference / 'complete.json').read_text())
    assert complete['steps'] == 3125 and complete['sample_visits'] == 200000
    assert previous['world_size'] == 2
    assert sha(reference / 'config.json') == previous['config_sha256']
    for path, digest in previous['code_sha256'].items():
        assert sha(PROJECT / path) == digest, f'A4 source changed: {path}'
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    root = PROJECT / 'runs/formal_decision_mechanisms_20261103_v1'
    root.mkdir(exist_ok=False)
    new_sources = [
        'scripts/formal/decision_mechanisms.py', 'scripts/formal/score_decoding.py',
        'scripts/formal/models_decision_mechanism.py', 'scripts/formal/worker_decision_mechanism.py',
        'scripts/formal/supervise_decision_mechanism.py', 'scripts/formal/prepare_decision_mechanism.py',
        'tests/test_decision_mechanisms.py',
    ]
    hashes = dict(previous['code_sha256'])
    hashes.update({name: sha(PROJECT / name) for name in new_sources})
    for name in hashes:
        dest = root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        assert sha(dest) == hashes[name]
    reference_files = ['config.json', 'initialization.json', 'complete.json',
                       'evaluations/epoch_1.dev.json', 'evaluations/epoch_1.test_in_domain.json']
    atomic(root / 'reference.json', {'run': str(reference), 'checkpoint_inheritance': None,
           'files': {str(reference / f): sha(reference / f) for f in reference_files}})
    arms = {}
    control_audit = {}
    variants = [
        ('m1_score_regression', '0,1,2,3', 'Score CE + 0.5 native MAE median-regression auxiliary head',
         Mechanism('score_regression'), 'deciform_score_reg_20261103'),
        ('m2_choice_metric', '4,5,6,7', 'Choice question-conditioned normalized positive QF precision',
         Mechanism('choice_metric'), 'deciform_choice_metric_20261103'),
    ]
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'triton')}
    for name, gpus, label, objective, session in variants:
        output = root / name
        output.mkdir()
        config = copy.deepcopy(base_config)
        config.update(label=label, output=str(output), method_variant=name,
                      worker_variant=str(PROJECT / 'scripts/formal/worker_decision_mechanism.py'),
                      mechanism=objective.inventory(), loss_diagnostics_every=50,
                      reference_run=str(reference), experiment_name=name,
                      task_losses={'choice': 'categorical CE',
                                   'noul': 'Bernoulli NLL',
                                   'score': label if objective.kind=='score_regression' else 'categorical CE'})
        config['mechanism_description'] = label
        config['world_size'] = 4
        config['score_regression_diagnostic'] = 'dev only; no primary decoding change'
        config.pop('ablation_name')
        declared_changes = {'label', 'output', 'method_variant', 'worker_variant',
                            'task_losses', 'ablation_name', 'reference_run'}
        fixed_fields = sorted(set(base_config) - declared_changes)
        assert all(config[field] == base_config[field] for field in fixed_fields)
        control_audit[name] = {'fixed_config_fields': fixed_fields,
            'status': 'matched_except_declared_mechanism_and_hardware',
            'noul_loss': 'Bernoulli NLL', 'global_batch': 64,
            'reference_world_size': 2, 'world_size': 4,
            'parent_checkpoint': None, 'same_negative_ids': True,
            'data_sha256_verified_by_worker': True}
        atomic(output / 'config.json', config)
        manifest = copy.deepcopy(previous)
        manifest.update(created=created, gpus=gpus, world_size=4, python=sys.executable,
                        python_version=platform.python_version(), packages=packages,
                        config=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
                        worker=config['worker_variant'],
                        supervisor=str(PROJECT / 'scripts/formal/supervise_decision_mechanism.py'),
                        code_sha256=hashes, code_snapshot=str(root / 'code'),
                        experiment_name=name, mechanism=objective.inventory(),
                        baseline_initialization=str(reference / 'initialization.json'),
                        expected_trainable_parameters=46304262 + extra_parameters(objective),
                        reference_run=str(reference), tmux_session=session,
                        launch_command=[sys.executable, str(PROJECT / 'scripts/formal/supervise_decision_mechanism.py'),
                                        '--manifest', str(output / 'experiment_manifest.json')],
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                        comparison='Same data/global batches/initial base as A4; one named mechanism per arm. Four GPUs per arm vs A4 two: rank/RNG/reduction differences recorded, not a matched-hardware speed comparison.')
        manifest.pop('ablation_name', None)
        # This field is the source location only; no old checkpoint is loaded.
        manifest['parent_checkpoint'] = None
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': gpus, 'label': label, 'tmux_session': session,
                      'manifest': str(output / 'experiment_manifest.json'), 'output': str(output),
                      'mechanism': objective.inventory(), 'state': 'prepared'}
    atomic(root / 'experiment_manifest.json', {
        'created': created, 'arms': arms, 'reference': str(root / 'reference.json'),
        'train_rows': 200000, 'epochs': 1, 'steps': 3125, 'world_size_per_arm': 4,
        'initialization': 'fresh_base_only', 'code_sha256': hashes,
        'data': base_config['data'], 'cache_audit_sha256': base_config['cache_audit_sha256'],
        'evaluated_splits': ['dev', 'test_in_domain'], 'score_prediction': base_config['score_prediction'],
        'dev_only_decoding_diagnostic': 'predictive median on saved native probabilities; no headline changes',
        'scope': 'score auxiliary native MAE or conditional choice QF; noul BCE; all categorical CE coefficients 1; frozen inputs and argmax outputs',
    })
    atomic(root / 'matched_control_audit.json', control_audit)
    print(root)


if __name__ == '__main__':
    main()
