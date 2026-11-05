"""Freeze a reproducible multi-order audit for the final independent D2."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha


def main():
    full = PROJECT / 'runs/formal_independent_20261105_v1'
    root = PROJECT / 'runs/formal_independent_position_20261105_v1'
    if root.exists():
        raise FileExistsError(root)
    config = json.loads((full / 'config.json').read_text())
    manifest = json.loads((full / 'experiment_manifest.json').read_text())
    complete = json.loads((full / 'complete.json').read_text())
    assert complete['steps'] == 3125 and complete['sample_visits'] == 200000
    checkpoint = full / 'epoch_1.pt'
    data = Path(config['data'])
    audit = json.loads((data / 'cache/audit.json').read_text())
    assert audit['status'] == 'passed' and audit['splits']['test_in_domain']['native_evaluation']
    root.mkdir()
    orders = ['original', 'reverse', 'rotate_left_1', 'rotate_right_1', 'shuffle_11', 'shuffle_29', 'shuffle_47']
    code_paths = [
        'scripts/formal/common.py', 'scripts/formal/models_independent.py',
        'scripts/formal/independent_candidates.py', 'scripts/formal/score_distribution.py',
        'scripts/formal/order_audit_independent.py', 'scripts/formal/prepare_independent_order_audit.py',
    ]
    code_hashes = {path: sha(PROJECT / path) for path in code_paths}
    cfg = {
        'output': str(root), 'data': config['data'], 'cache_audit_sha256': config['cache_audit_sha256'],
        'base': config['base'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'ablation': config['ablation'], 'distribution_objective': config['distribution_objective'],
        'evaluation_candidates': 'complete_native', 'eval_token_budget': config['eval_token_budget'],
        'orders': orders, 'tasks': ['choice', 'noul'], 'world_size': 2,
    }
    atomic(root / 'config.json', cfg)
    manifest_out = {
        'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'status': 'prepared',
        'python': sys.executable, 'python_version': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'triton')},
        'gpus': '0,1', 'world_size': 2, 'config': str(root / 'config.json'),
        'config_sha256': sha(root / 'config.json'), 'worker': str(PROJECT / 'scripts/formal/order_audit_independent.py'),
        'code_sha256': code_hashes, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'reference_run': str(full), 'data': config['data'], 'cache_audit_sha256': config['cache_audit_sha256'],
        'orders': orders, 'tasks': ['choice', 'noul'], 'native_full_candidates': True,
        'launch_command': [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                           str(PROJECT / 'scripts/formal/order_audit_independent.py'), '--config', str(root / 'config.json')],
    }
    atomic(root / 'experiment_manifest.json', manifest_out)
    atomic(root / 'input_hashes.json', {
        'checkpoint_sha256': sha(checkpoint), 'cache_audit_sha256': config['cache_audit_sha256'],
        'test_views_sha256': audit['splits']['test_in_domain']['views_sha256'],
        'test_encoding_sha256': audit['splits']['test_in_domain']['arms']['kev']['sha256'],
        'native_evaluation': True, 'candidate_order_only': True,
    })
    print(root)


if __name__ == '__main__':
    main()
