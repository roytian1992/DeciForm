"""Freeze candidate-independent D2, with identical parameters/data/losses."""
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


def main():
    reference = PROJECT / 'runs/formal_score_distribution_20261104_v1/d2_ordinal_main'
    root = PROJECT / 'runs/formal_independent_20261105_v1'
    previous = json.loads((reference / 'experiment_manifest.json').read_text())
    base = json.loads((reference / 'config.json').read_text())
    assert sha(reference / 'config.json') == previous['config_sha256']
    assert base['parent_checkpoint'] is None
    assert base['distribution_objective']['kind'] == 'ordinal_main'
    for path, digest in previous['code_sha256'].items():
        assert sha(PROJECT / path) == digest, f'Frozen D2 source changed: {path}'
    audit_path = Path(base['data']) / 'cache/audit.json'
    assert sha(audit_path) == base['cache_audit_sha256']
    audit = json.loads(audit_path.read_text())
    assert {k: audit['splits'][k]['rows'] for k in ('train','dev','test_in_domain')} == {
        'train': 200000, 'dev': 6000, 'test_in_domain': 15000}
    config = copy.deepcopy(base)
    config.update(label='D2 with shared differentiable prefix and independent candidates',
        output=str(root), experiment_name='independent_d2',
        method_variant='independent_d2', reference_run=str(reference), total_steps=3125,
        worker_variant=str(PROJECT / 'scripts/formal/worker_independent.py'),
        candidate_encoder='shared_prefix_independent_branches',
        query_encoder='state + question + isolated decide token',
        candidate_positions='each option starts at the actual unpadded prefix length',
        classification_ties='exact ties use stable semantic ID, never display position',
        order_audit='full-native test Choice/Noul, original and reverse; Score excluded')
    changed = {k for k in base if config[k] != base[k]}
    assert changed <= {'label','output','experiment_name','method_variant','reference_run','worker_variant'}
    hashes = dict(previous['code_sha256'])
    added = ['scripts/formal/independent_candidates.py', 'scripts/formal/models_independent.py',
        'scripts/formal/worker_independent.py', 'scripts/formal/prepare_independent.py',
        'scripts/formal/supervise_score_specialist.py', 'tests/test_independent_candidates.py']
    for path in added:
        hashes[path] = sha(PROJECT / path)
    root.mkdir(exist_ok=False)
    for path, digest in hashes.items():
        dest = root / 'code' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, dest)
        assert sha(dest) == digest
    atomic(root / 'config.json', config)
    shutil.copy2(reference.parent / 'evaluation_protocol.json', root / 'evaluation_protocol.json')
    supervisor = str(PROJECT / 'scripts/formal/supervise_score_specialist.py')
    manifest = copy.deepcopy(previous)
    manifest.update(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        python=sys.executable, python_version=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ('torch','transformers','peft','triton')},
        config=str(root / 'config.json'), config_sha256=sha(root / 'config.json'),
        worker=config['worker_variant'], supervisor=supervisor, code_sha256=hashes,
        code_snapshot=str(root / 'code'), reference_run=str(reference),
        baseline_initialization=str(reference / 'initialization.json'),
        experiment_name='independent_d2', gpus='4,5,6,7', world_size=4,
        tmux_session='deciform_independent_20261105',
        launch_command=[sys.executable, supervisor, '--manifest', str(root / 'experiment_manifest.json')],
        resume_command=[sys.executable, supervisor, '--manifest', str(root / 'experiment_manifest.json'), '--resume'],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        comparison='Identical fresh base, parameter initialization, all-target LoRA/routing/QF/geometry/CDF, losses, fixed records/negative IDs and global batches as D2; only causal encoding topology and exact-tie classification rule change. Shared prefix dropout is shared across branches; hardware and global batch match D2.',
        candidate_encoder=config['candidate_encoder'], parent_checkpoint=None,
        data_counts={'train':200000,'dev':6000,'test_in_domain':15000},
        cache_audit_sha256=sha(audit_path),
        changed_reference_config_fields=sorted(changed),
        provenance_note='Shared-prefix functional KV/convolution/recurrent cache reuses attributed vendor Kev implementation; independent candidate branching and prefix-only query are implemented here. Do not claim prefix sharing itself is novel.')
    atomic(root / 'experiment_manifest.json', manifest)
    atomic(root / 'input_hashes.json', {'data':base['data'],'cache_audit_sha256':sha(audit_path),
        'splits':{s:{'rows':audit['splits'][s]['rows'],
            'views_sha256':audit['splits'][s]['views_sha256'],
            'encoding_sha256':audit['splits'][s]['arms']['kev']['sha256']}
            for s in ('train','dev','test_in_domain')},
        'reference_config_sha256':sha(reference/'config.json'),
        'reference_initialization_sha256':sha(reference/'initialization.json'),
        'no_input_truncation':True,'evaluation_candidates':'complete native'})
    print(root)


if __name__ == '__main__':
    main()
