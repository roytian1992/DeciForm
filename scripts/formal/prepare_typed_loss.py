"""Freeze two noul/score loss comparisons against the completed, two-GPU A4 arm."""
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
from typed_losses import TypedObjective


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
    root = PROJECT / 'runs/formal_typed_losses_20261103_v1'
    root.mkdir(exist_ok=False)
    new_sources = [
        'scripts/formal/typed_losses.py', 'scripts/formal/score_decoding.py', 'scripts/formal/models_typed_loss.py',
        'scripts/formal/worker_typed_loss.py', 'scripts/formal/supervise_typed_loss.py',
        'scripts/formal/prepare_typed_loss.py', 'tests/test_typed_loss_experiments.py',
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
    variants = [
        ('l3_noul_brier', '0,1', 'Noul BCE + binary Brier, rho=0.5',
         TypedObjective('noul_brier'), 'deciform_loss_noul_20261103'),
        ('l4_score_ordinal', '2,3', 'Score CE + ordinal threshold log loss, weight=0.1',
         TypedObjective('score_ordinal'), 'deciform_loss_score_20261103'),
    ]
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'triton')}
    for name, gpus, label, objective, session in variants:
        output = root / name
        output.mkdir()
        config = copy.deepcopy(base_config)
        config.update(label=label, output=str(output), method_variant=name,
                      worker_variant=str(PROJECT / 'scripts/formal/worker_typed_loss.py'),
                      typed_objective=objective.inventory(), loss_diagnostics_every=50,
                      reference_run=str(reference), experiment_name=name,
                      task_losses={'choice': 'categorical CE',
                                   'noul': label if objective.kind=='noul_brier' else 'Bernoulli NLL',
                                   'score': label if objective.kind=='score_ordinal' else 'categorical CE'})
        config.pop('ablation_name')
        atomic(output / 'config.json', config)
        manifest = copy.deepcopy(previous)
        manifest.update(created=created, gpus=gpus, world_size=2, python=sys.executable,
                        python_version=platform.python_version(), packages=packages,
                        config=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
                        worker=config['worker_variant'],
                        supervisor=str(PROJECT / 'scripts/formal/supervise_typed_loss.py'),
                        code_sha256=hashes, code_snapshot=str(root / 'code'),
                        experiment_name=name, typed_objective=objective.inventory(),
                        baseline_initialization=str(reference / 'initialization.json'),
                        expected_trainable_parameters=46304262,
                        reference_run=str(reference), tmux_session=session,
                        launch_command=[sys.executable, str(PROJECT / 'scripts/formal/supervise_typed_loss.py'),
                                        '--manifest', str(output / 'experiment_manifest.json')],
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                        comparison='Same two-GPU partition/global batches as A4; only the named noul or score objective changes.')
        manifest.pop('ablation_name', None)
        # This field is the source location only; no old checkpoint is loaded.
        manifest['parent_checkpoint'] = None
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': gpus, 'label': label, 'tmux_session': session,
                      'manifest': str(output / 'experiment_manifest.json'), 'output': str(output),
                      'typed_objective': objective.inventory(), 'state': 'prepared'}
    atomic(root / 'experiment_manifest.json', {
        'created': created, 'arms': arms, 'reference': str(root / 'reference.json'),
        'train_rows': 200000, 'epochs': 1, 'steps': 3125, 'world_size_per_arm': 2,
        'initialization': 'fresh_base_only', 'code_sha256': hashes,
        'data': base_config['data'], 'cache_audit_sha256': base_config['cache_audit_sha256'],
        'evaluated_splits': ['dev', 'test_in_domain'], 'score_prediction': base_config['score_prediction'],
        'dev_only_decoding_diagnostic': 'predictive median on saved native probabilities; no headline changes',
        'scope': 'one typed objective per arm; A4 architecture/choice/regularization and data unchanged; score CE coefficient stays 1',
    })
    print(root)


if __name__ == '__main__':
    main()
