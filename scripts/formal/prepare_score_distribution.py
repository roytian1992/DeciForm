"""Freeze score-distribution objectives and common decoder rules before test replay."""
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
from score_distribution import DistributionObjective, PROTOCOL
from calibrate_score_objective import calibrate_ordinal_scale


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
    root = PROJECT / 'runs/formal_score_distribution_20261104_v1'
    root.mkdir(exist_ok=False)
    scale_audit = calibrate_ordinal_scale(Path(base_config['data']))
    atomic(root / 'ordinal_scale.json', scale_audit)
    protocol = {'name': PROTOCOL, 'primary_score_rule': 'median', 'primary_score_metric': 'native MAE',
                'reported_rules': ['argmax', 'median', 'mean', 'mean_nearest'],
                'native_rules': ['argmax', 'median', 'mean_nearest'],
                'continuous_diagnostic_rules': ['mean'], 'frozen_at': created,
                'selection': 'median prescribed by MAE decision theory; no rule chosen on test',
                'baseline_application': 'identical four-rule replay for saved full-native probability baselines',
                'legacy_reports': 'preserved separately', 'noul': 'unchanged BCE and argmax',
                'choice': 'unchanged A4 QF/CE and argmax',
                'user_authorization': 'User requested probability/expectation evaluation or a justified alternative and new training.'}
    atomic(root / 'evaluation_protocol.json', protocol)
    new_sources = [
        'scripts/formal/score_distribution.py', 'scripts/formal/typed_losses.py',
        'scripts/formal/models_score_distribution.py', 'scripts/formal/worker_score_distribution.py',
        'scripts/formal/supervise_score_distribution.py', 'scripts/formal/prepare_score_distribution.py',
        'scripts/formal/calibrate_score_objective.py', 'scripts/formal/replay_score_decisions.py',
        'tests/test_score_distribution.py',
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
        ('d1_mean_moment', '0,1,2,3', 'Score CE + 0.5 squared error of predictive mean divided by native span',
         DistributionObjective('mean_moment'), 'deciform_mean_moment_20261104'),
        ('d2_ordinal_main', '4,5,6,7', 'Score pure ordered CDF log loss with frozen uniform-gradient scale',
         DistributionObjective('ordinal_main', ordinal_scale=scale_audit['scale']), 'deciform_ordinal_main_20261104'),
    ]
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'triton')}
    for name, gpus, label, objective, session in variants:
        output = root / name
        output.mkdir()
        config = copy.deepcopy(base_config)
        config.update(label=label, output=str(output), method_variant=name,
                      worker_variant=str(PROJECT / 'scripts/formal/worker_score_distribution.py'),
                      distribution_objective=objective.inventory(), loss_diagnostics_every=50,
                      reference_run=str(reference), experiment_name=name,
                      task_losses={'choice': 'categorical CE',
                                   'noul': 'Bernoulli NLL',
                                   'score': label})
        config['objective_description'] = label
        config['world_size'] = 4
        config['evaluation_protocol'] = PROTOCOL
        config['score_decision_rule'] = 'median'
        config['score_prediction'] = 'native predictive median; MAE primary; fixed four-rule reports'
        config['score_rule_reports'] = ['argmax', 'median', 'mean', 'mean_nearest']
        config.pop('ablation_name')
        declared_changes = {'label', 'output', 'method_variant', 'worker_variant',
                            'task_losses', 'ablation_name', 'reference_run', 'score_prediction'}
        fixed_fields = sorted(set(base_config) - declared_changes)
        assert all(config[field] == base_config[field] for field in fixed_fields)
        control_audit[name] = {'fixed_config_fields': fixed_fields,
            'status': 'matched_except_score_objective_decoder_and_hardware',
            'noul_loss': 'Bernoulli NLL', 'global_batch': 64, 'evaluation_protocol': PROTOCOL,
            'reference_world_size': 2, 'world_size': 4,
            'parent_checkpoint': None, 'same_negative_ids': True,
            'data_sha256_verified_by_worker': True}
        atomic(output / 'config.json', config)
        manifest = copy.deepcopy(previous)
        manifest.update(created=created, gpus=gpus, world_size=4, python=sys.executable,
                        python_version=platform.python_version(), packages=packages,
                        config=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
                        worker=config['worker_variant'],
                        supervisor=str(PROJECT / 'scripts/formal/supervise_score_distribution.py'),
                        code_sha256=hashes, code_snapshot=str(root / 'code'),
                        experiment_name=name, distribution_objective=objective.inventory(),
                        baseline_initialization=str(reference / 'initialization.json'),
                        expected_trainable_parameters=46304262, evaluation_protocol=PROTOCOL,
                        reference_run=str(reference), tmux_session=session,
                        launch_command=[sys.executable, str(PROJECT / 'scripts/formal/supervise_score_distribution.py'),
                                        '--manifest', str(output / 'experiment_manifest.json')],
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                        comparison='Same data/global batches/initial base as A4; one score objective per arm; matched four-rule evaluation for all baselines. Four GPUs per arm vs A4 two: rank/RNG/reduction differences recorded, not a matched-hardware speed comparison.')
        manifest.pop('ablation_name', None)
        # This field is the source location only; no old checkpoint is loaded.
        manifest['parent_checkpoint'] = None
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': gpus, 'label': label, 'tmux_session': session,
                      'manifest': str(output / 'experiment_manifest.json'), 'output': str(output),
                      'distribution_objective': objective.inventory(), 'state': 'prepared'}
    atomic(root / 'experiment_manifest.json', {
        'created': created, 'arms': arms, 'reference': str(root / 'reference.json'),
        'train_rows': 200000, 'epochs': 1, 'steps': 3125, 'world_size_per_arm': 4,
        'initialization': 'fresh_base_only', 'code_sha256': hashes,
        'data': base_config['data'], 'cache_audit_sha256': base_config['cache_audit_sha256'],
        'evaluated_splits': ['dev', 'test_in_domain'], 'score_prediction': 'native predictive median', 'evaluation_protocol': PROTOCOL,
        'score_rules': protocol['reported_rules'], 'protocol_file_sha256': sha(root / 'evaluation_protocol.json'),
        'scope': 'fixed A4 architecture and data, choice CE/noul BCE; direct distribution mean regression or ordinal main score objective; matched median headline and four-rule evaluation',
    })
    atomic(root / 'matched_control_audit.json', control_audit)
    print(root)


if __name__ == '__main__':
    main()
