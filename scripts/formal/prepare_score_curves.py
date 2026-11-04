"""Freeze D2-full size curves, matching completed Kev data and evaluation IDs."""
import copy
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from curve_contract import load_subset, require_nested


def main():
    reference = PROJECT / 'runs/formal_score_distribution_20261104_v1/d2_ordinal_main'
    root = PROJECT / 'runs/formal_d2_curves_20261104_v1'
    previous = json.loads((reference / 'experiment_manifest.json').read_text())
    base = json.loads((reference / 'config.json').read_text())
    complete = json.loads((reference / 'complete.json').read_text())
    assert complete['steps'] == 3125 and complete['sample_visits'] == 200000
    assert sha(reference / 'config.json') == previous['config_sha256']
    assert base['distribution_objective']['kind'] == 'ordinal_main'
    assert base['parent_checkpoint'] is None and 'continuation' not in base
    for path, digest in previous['code_sha256'].items():
        assert sha(PROJECT / path) == digest, f'D2 source changed: {path}'
    audit_path = Path(base['data']) / 'cache/audit.json'
    assert sha(audit_path) == base['cache_audit_sha256']
    audit = json.loads(audit_path.read_text())
    desc = audit['splits']['train']
    assert sha(desc['views_path']) == desc['views_sha256']
    rows = [json.loads(line) for line in open(desc['views_path'])]
    assert len(rows) == desc['rows'] == 200000
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    arms, subsets, controls = {}, {}, {}
    cases = [(25000, '0,1'), (50000, '0,1'), (100000, '2,3')]
    # Validate every comparison before creating a new experiment root.
    for size, _ in cases:
        kev = PROJECT / f'runs/formal_next_20261102/kev_{size // 1000}k'
        cfg = json.loads((kev / 'config.json').read_text())
        done = json.loads((kev / 'complete.json').read_text())
        init = json.loads((kev / 'initialization.json').read_text())
        assert done['sample_visits'] == size and done['steps'] == math.ceil(size / 64)
        assert init['mode'] == 'fresh_base_only'
        assert init['inherited_adapter'] is None and init['inherited_head'] is None
        fixed = ['data', 'cache_audit_sha256', 'base', 'initialization', 'seed',
                 'global_batch', 'epochs', 'token_budget', 'eval_token_budget',
                 'base_lr', 'warmup_fraction', 'candidate_sampling',
                 'evaluation_candidates', 'task_balanced']
        assert all(cfg[key] == base[key] for key in fixed)
        path = Path(cfg['train_subset_indices'])
        ids = load_subset(path, sha(path), rows, size)
        subsets[size] = ids
        controls[str(size)] = {'kev_run': str(kev), 'subset': str(path),
            'subset_sha256': sha(path), 'fixed_config_fields': fixed,
            'files_sha256': {str(kev / name): sha(kev / name) for name in
                ('config.json', 'initialization.json', 'complete.json')},
            'historical_continuation_field': 'unused stale metadata in Kev config; worker_next loads only its own resume.pt with --resume; fresh initialization verified',
            'task_counts': json.loads(path.read_text())['task_counts']}
    require_nested(subsets)
    root.mkdir(exist_ok=False)
    hashes = dict(previous['code_sha256'])
    for path in ['scripts/formal/curve_contract.py', 'scripts/formal/worker_score_curve.py',
                 'scripts/formal/prepare_score_curves.py', 'tests/test_curve_contract.py',
                 'scripts/formal/supervise_score_specialist.py',
                 'scripts/formal/schedule_score_ablations.py',
                 'scripts/formal/replay_curve_references.py']:
        hashes[path] = sha(PROJECT / path)
    for path, digest in hashes.items():
        dest = root / 'code' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, dest)
        assert sha(dest) == digest
    protocol_path = reference.parent / 'evaluation_protocol.json'
    shutil.copy2(protocol_path, root / 'evaluation_protocol.json')
    reference_files = ['config.json', 'initialization.json', 'complete.json',
        'evaluations/epoch_1.dev.json', 'evaluations/epoch_1.test_in_domain.json']
    atomic(root / 'reference.json', {'run': str(reference), 'checkpoint_inheritance': None,
        'reuse_200k_endpoint': True,
        'files_sha256': {str(reference / name): sha(reference / name) for name in reference_files}})
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'triton')}
    for size, gpus in cases:
        name = f'd2_{size // 1000}k'
        output = root / name
        output.mkdir()
        control = controls[str(size)]
        config = copy.deepcopy(base)
        config.update(label=f'D2 full {size // 1000}k', output=str(output),
            worker_variant=str(PROJECT / 'scripts/formal/worker_score_curve.py'),
            reference_run=str(reference), experiment_name=name, world_size=2,
            train_rows=size, total_steps=math.ceil(size / 64),
            train_subset_indices=control['subset'], train_subset_sha256=control['subset_sha256'],
            train_views_sha256=desc['views_sha256'],
            schedule=f'OneCycleLR, {math.ceil(size / 64)} steps, pct_start=0.3')
        changed = {key for key in base if config[key] != base[key]}
        allowed = {'label', 'output', 'worker_variant', 'reference_run', 'experiment_name',
                   'world_size', 'train_rows', 'schedule'}
        assert changed <= allowed
        control['changed_fields_vs_full_200k'] = sorted(changed)
        control['unchanged_d2_fields'] = sorted(set(base) - allowed)
        atomic(output / 'config.json', config)
        session = f'deciform_d2_curves_g{gpus.replace(",", "")}_20261104'
        supervisor = str(PROJECT / 'scripts/formal/supervise_score_specialist.py')
        manifest = copy.deepcopy(previous)
        manifest.update(created=created, gpus=gpus, world_size=2,
            python=sys.executable, python_version=platform.python_version(), packages=packages,
            config=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
            worker=config['worker_variant'], supervisor=supervisor, code_sha256=hashes,
            code_snapshot=str(root / 'code'), experiment_name=name,
            baseline_initialization=str(reference / 'initialization.json'),
            reference_run=str(reference), tmux_session=session,
            launch_command=[sys.executable, supervisor, '--manifest',
                            str(output / 'experiment_manifest.json')],
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
            subset_sha256=control['subset_sha256'], train_rows=size,
            comparison='Exact Kev nested record IDs and frozen candidate views; unchanged D2 full architecture/loss/decoder; fresh base for each size. Two GPUs per new arm, 200k D2 reused from four GPUs; not a matched-hardware timing study.')
        manifest['parent_checkpoint'] = None
        atomic(output / 'experiment_manifest.json', manifest)
        arms[name] = {'gpus': gpus, 'train_rows': size, 'steps': config['total_steps'],
            'manifest': str(output / 'experiment_manifest.json'), 'tmux_session': session,
            'state': 'prepared', 'output': str(output)}
    atomic(root / 'matched_control_audit.json', controls)
    atomic(root / 'experiment_manifest.json', {'created': created, 'arms': arms,
        'gpu_queues': {'0,1': ['d2_25k', 'd2_50k'], '2,3': ['d2_100k']},
        'initialization': 'fresh_base_only', 'epochs': 1, 'global_batch': 64,
        'world_size_per_arm': 2, 'seed': base['seed'], 'code_sha256': hashes,
        'data': base['data'], 'train_views_sha256': desc['views_sha256'],
        'cache_audit_sha256': base['cache_audit_sha256'], 'nested_subsets_verified': True,
        'evaluation_protocol': base['evaluation_protocol'],
        'protocol_file_sha256': sha(root / 'evaluation_protocol.json'),
        'evaluated_splits': ['dev', 'test_in_domain'],
        'evaluation_rows': {'dev': 6000, 'test_in_domain': 15000},
        'primary_metrics': {'choice': 'accuracy', 'noul': 'accuracy', 'score': 'native median MAE'},
        'reference_200k': str(reference), 'parent_checkpoint': None,
        'scope': 'Three independent one-epoch size points; automatic full-native internal eval; reuse completed Kev and D2 200k endpoint; external panel deferred'})
    print(root)


if __name__ == '__main__':
    main()
