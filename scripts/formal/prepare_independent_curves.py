"""Reuse matched frozen subsets; only replace the old causal D2 with final D2."""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from curve_contract import load_subset, require_nested


def main():
    ref = PROJECT / 'runs/formal_independent_20261105_v1'
    root = PROJECT / 'runs/formal_independent_curves_20261105_v1'
    prior = json.loads((ref / 'experiment_manifest.json').read_text())
    full = json.loads((ref / 'config.json').read_text())
    done = json.loads((ref / 'complete.json').read_text())
    assert done['sample_visits'] == 200000 and done['steps'] == 3125
    assert sha(ref / 'config.json') == prior['config_sha256']
    audit = json.loads((Path(full['data']) / 'cache/audit.json').read_text())
    train = audit['splits']['train']
    assert sha(train['views_path']) == train['views_sha256']
    rows = [json.loads(line) for line in open(train['views_path'])]
    cases, subsets = [], {}
    for size, gpus in [(25000, '0,1'), (50000, '0,1'), (100000, '2,3')]:
        old = PROJECT / f'runs/formal_d2_curves_20261104_v1/d2_{size // 1000}k'
        cfg = json.loads((old / 'config.json').read_text())
        subsets[size] = load_subset(cfg['train_subset_indices'], cfg['train_subset_sha256'], rows, size)
        cases.append((size, gpus, cfg))
    require_nested(subsets)
    paths = set(prior['code_sha256'])
    paths.update(['scripts/formal/worker_score_curve.py', 'scripts/formal/curve_contract.py',
                  'scripts/formal/worker_independent_curve.py', 'scripts/formal/prepare_independent_curves.py',
                  'scripts/formal/schedule_research_queue.py', 'tests/test_curve_contract.py'])
    # Frozen main-model implementation must remain the actual reference.
    for name in prior['code_sha256']:
        if Path(name).name.startswith(('models_', 'kev_ours', 'independent_candidates', 'score_distribution')):
            assert sha(PROJECT / name) == prior['code_sha256'][name], name
    hashes = {name: sha(PROJECT / name) for name in sorted(paths)}
    root.mkdir(exist_ok=False)
    for name in hashes:
        dest = root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
    arms = {}
    for size, gpus, old in cases:
        name = f'd2_{size // 1000}k'
        out = root / name
        out.mkdir()
        cfg = copy.deepcopy(full)
        cfg.update(output=str(out), label=f'Final independent D2 {size // 1000}k',
                   experiment_name=name, world_size=2, train_rows=size,
                   total_steps=math.ceil(size / 64),
                   train_subset_indices=old['train_subset_indices'],
                   train_subset_sha256=old['train_subset_sha256'],
                   train_views_sha256=train['views_sha256'], reference_run=str(ref),
                   worker_variant=str(PROJECT / 'scripts/formal/worker_independent_curve.py'),
                   schedule=f'OneCycleLR, {math.ceil(size / 64)} steps, pct_start=0.3',
                   order_audit='not repeated in size curves; full-native dev/test only')
        assert cfg['parent_checkpoint'] is None
        atomic(out / 'config.json', cfg)
        job = copy.deepcopy(prior)
        supervisor = str(PROJECT / 'scripts/formal/supervise_score_specialist.py')
        job.update(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   python=sys.executable, gpus=gpus, world_size=2, config=str(out / 'config.json'),
                   config_sha256=sha(out / 'config.json'), worker=cfg['worker_variant'],
                   code_sha256=hashes, code_snapshot=str(root / 'code'), experiment_name=name,
                   baseline_initialization=str(ref / 'initialization.json'), reference_run=str(ref),
                   data_counts={'train': size, 'dev': 6000, 'test_in_domain': 15000},
                   launch_command=[sys.executable, supervisor, '--manifest', str(out / 'experiment_manifest.json')],
                   comparison='Final independent D2, unchanged loss/decoder and frozen Kev nested subset/negative IDs; fresh base each size. Two GPUs vs four-GPU reused 200k endpoint; not a timing comparison.')
        atomic(out / 'experiment_manifest.json', job)
        arms[name] = {'gpus': gpus, 'train_rows': size, 'steps': cfg['total_steps'], 'manifest': str(out / 'experiment_manifest.json')}
    atomic(root / 'experiment_manifest.json', {
        'arms': arms, 'gpu_queues': {'0,1': ['d2_25k', 'd2_50k'], '2,3': ['d2_100k']},
        'queue_dependencies': {'2,3': str(PROJECT / 'runs/formal_independent_score_completion_20261105_v1/asap_ce/status.json')},
        'code_sha256': hashes, 'reference_200k': str(ref), 'reuse_200k_endpoint': True,
        'fresh_base_only': True, 'nested_subsets_verified': True,
        'evaluation': 'unchanged complete-native 6000 dev / 15000 test',
        'scope': 'User-authorized 4 GPU final-D2 size curves; no full-200k or ablation reruns'})
    print(root)


if __name__ == '__main__':
    main()
