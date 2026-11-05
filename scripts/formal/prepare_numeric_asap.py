"""Freeze an unchanged numeric-head transfer to the native ASAP Score panel."""
import copy
from datetime import datetime
import json
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from asap_score import PROTOCOL


def main():
    parent = PROJECT / 'runs/formal_score_numeric_20261105_v1/numeric_cdf'
    reference = PROJECT / 'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap'
    root = PROJECT / 'runs/formal_score_numeric_asap_20261105_v1'
    data = PROJECT / 'data/score_specialist_asap_fold0_20261104_v1'
    cfg = json.loads((parent / 'config.json').read_text())
    job = json.loads((parent / 'experiment_manifest.json').read_text())
    audit_path = data / 'cache/audit.json'
    audit = json.loads(audit_path.read_text())
    counts = {s: v['rows'] for s, v in audit['splits'].items()}
    assert audit['status'] == 'passed' and audit['data_manifest_sha256'] == sha(data / 'manifest.json')
    assert counts == {'train': 7784, 'dev': 2593, 'test_in_domain': 2598}
    benchmark = PROJECT / 'runs/score_execution_preflight_20261105_v1/numeric_asap_budget8192_half/complete.json'
    proof = json.loads(benchmark.read_text())
    assert proof['status'] == 'passed' and proof['optimizer_updates'] == 0
    assert proof['token_budget'] == 8192 and proof['rows_per_microbatch'] == 4
    assert proof['evaluation_all_levels'] == [4, 5, 6, 11, 31, 61]
    assert max(x['peak_allocated_gib'] for x in proof['measurements']) < 64
    paths = set(job['code_sha256']) | {'scripts/formal/worker_numeric_asap.py',
        'scripts/formal/benchmark_numeric_asap.py', 'scripts/formal/prepare_numeric_asap.py',
        'scripts/formal/asap_score.py'}
    code_hashes = {p: sha(PROJECT / p) for p in sorted(paths)}
    assert all(code_hashes[n] == h for n, h in proof['code_sha256'].items())
    root.mkdir(exist_ok=False)
    for path in paths:
        dest = root / 'code' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, dest)
    cfg.update(data=str(data), cache_audit_sha256=sha(audit_path), output=str(root),
               label='RQ-Score frozen numeric CDF on native ASAP', experiment_name='numeric_asap',
               allowed_sources=['asap_set_' + str(i) for i in range(1, 9)],
               train_rows=7784, total_steps=122, max_microbatch_rows=4,
               token_budget=8192, eval_token_budget=8192,
               schedule='OneCycleLR, 122 steps, pct_start=0.3',
               worker_variant=str(PROJECT / 'scripts/formal/worker_numeric_asap.py'),
               evaluation_protocol=PROTOCOL, paired_reference=str(reference),
               score_prediction='native integer predictive median',
               primary_metrics={'score': 'native MAE per prompt; equal-prompt NMAE'},
               evaluation_candidates='complete native integer support in train/dev/test',
               candidate_sampling='none; complete native integer support',
               objective_description='Same D2 ordinal CDF objective; soft-CDF equals integer ordinal loss on native labels')
    assert cfg['parent_checkpoint'] is None
    atomic(root / 'config.json', cfg)
    job.update(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               python=sys.executable, gpus='6,7', config=str(root / 'config.json'),
               config_sha256=sha(root / 'config.json'), worker=cfg['worker_variant'],
               code_sha256=code_hashes, code_snapshot=str(root / 'code'),
               experiment_name='numeric_asap', data=str(data), data_counts=counts,
               data_manifest_sha256=sha(data / 'manifest.json'), cache_audit_sha256=sha(audit_path),
               evaluation_protocol=PROTOCOL, paired_reference=str(reference),
               execution_benchmark=str(benchmark), execution_benchmark_sha256=sha(benchmark),
               expected_trainable_parameters=proof['trainable_parameters'],
               tmux_session='deciform_score_numeric_asap_20261105',
               launch_command=[sys.executable, job['supervisor'], '--manifest', str(root / 'experiment_manifest.json')],
               comparison='Frozen numeric head/loss on original native ASAP fold0; no hyperparameter change after continuous test; historical independent D2 reference has different execution/checkpoint policy.',
               git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip())
    for stale in ('completed_dependency', 'completed_dependency_sha256', 'resume_command'):
        job.pop(stale, None)
    atomic(root / 'experiment_manifest.json', job)
    print(root)


if __name__ == '__main__':
    main()
