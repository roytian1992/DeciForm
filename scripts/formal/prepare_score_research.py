"""Freeze matched no-German RQ-Score runs after measured execution preflights."""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha


def main():
    root = PROJECT / 'runs/formal_score_numeric_20261105_v1'
    source = PROJECT / 'runs/formal_independent_score_completion_20261105_v1/continuous_cdf'
    ref = PROJECT / 'runs/formal_independent_20261105_v1'
    data = PROJECT / 'data/score_specialist_stsb_enzh_20261105_v1'
    prior = json.loads((source / 'experiment_manifest.json').read_text())
    base = json.loads((source / 'config.json').read_text())
    audit_path = data / 'cache/audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and audit['data_manifest_sha256'] == sha(data / 'manifest.json')
    counts = {s: e['rows'] for s, e in audit['splits'].items()}
    assert counts == {'train': 8129, 'dev': 2500, 'test_in_domain': 2379}
    paths = set(prior['code_sha256']) | {
        'scripts/formal/prepare_score_no_german.py', 'scripts/formal/models_score_research.py',
        'scripts/formal/numeric_score_distribution.py', 'scripts/formal/worker_score_research.py',
        'scripts/formal/supervise_score_research.py', 'scripts/formal/prepare_score_research.py',
        'scripts/formal/benchmark_score_research.py', 'tests/test_numeric_score_distribution.py'}
    code_hashes = {name: sha(PROJECT / name) for name in sorted(paths)}
    specs = [('d2_batched', '4,5', 4, 'd2_final_batch4_half'),
             ('numeric_cdf', '6,7', 16, 'numeric_final_batch16_half')]
    proofs = {}
    for name, _, micro, benchmark in specs:
        p = PROJECT / 'runs/score_execution_preflight_20261105_v1' / benchmark / 'complete.json'
        proof = json.loads(p.read_text())
        assert proof['status'] == 'passed' and proof['optimizer_updates'] == 0
        assert proof['variant'] == name and proof['rows_per_microbatch'] == micro
        assert proof['checkpoint_policy'] == 'half'
        assert max(x['peak_allocated_gib'] for x in proof['measurements']) < 64
        assert all(code_hashes[n] == h for n, h in proof['code_sha256'].items())
        proofs[name] = (p, proof)
    root.mkdir(exist_ok=False)
    for name in paths:
        dest = root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
    arms = {}
    for name, gpus, micro, _ in specs:
        out = root / name
        out.mkdir()
        cfg = copy.deepcopy(base)
        cfg.pop('score_branch_chunk_size', None)
        cfg.update(data=str(data), cache_audit_sha256=sha(audit_path), output=str(out),
                   label=f'RQ-Score {name}; STS-B + WMT en-zh', experiment_name=name,
                   method_variant='score_numeric_research', score_architecture=name,
                   allowed_sources=['stsb', 'wmt_en_zh'], max_microbatch_rows=micro,
                   train_rows=counts['train'], total_steps=math.ceil(counts['train']/64),
                   checkpoint_policy='half', checkpoint_policy_candidates=['half'],
                   token_budget=16384, eval_token_budget=16384,
                   schedule=f'OneCycleLR, {math.ceil(counts["train"]/64)} steps, pct_start=0.3',
                   worker_variant=str(PROJECT / 'scripts/formal/worker_score_research.py'),
                   evaluation_protocol='score_stsb_enzh_numeric_20261105_v1',
                   candidate_encoder='batched_shared_prefix_independent' if name == 'd2_batched' else 'single_state_rubric_query_encoding',
                   score_head='D2 ordinal CDF readout' if name == 'd2_batched' else 'adaptive 32-bin conditional CDF with endpoint atoms and linear projection',
                   score_prediction='full-support projected predictive median',
                   paired_reference=str(root / ('numeric_cdf' if name == 'd2_batched' else 'd2_batched')),
                   reference_run=str(ref))
        assert cfg['parent_checkpoint'] is None and cfg['distribution_objective']['kind'] == 'soft_cdf'
        atomic(out / 'config.json', cfg)
        benchmark_path, benchmark_proof = proofs[name]
        supervisor = str(PROJECT / 'scripts/formal/supervise_score_research.py')
        job = copy.deepcopy(prior)
        job.update(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   python=sys.executable, gpus=gpus, config=str(out / 'config.json'),
                   config_sha256=sha(out / 'config.json'), worker=cfg['worker_variant'],
                   supervisor=supervisor, code_sha256=code_hashes, code_snapshot=str(root / 'code'),
                   experiment_name=name, data=str(data), data_counts=counts,
                   data_manifest_sha256=sha(data / 'manifest.json'), cache_audit_sha256=sha(audit_path),
                   evaluation_protocol=cfg['evaluation_protocol'],
                   expected_trainable_parameters=benchmark_proof['trainable_parameters'],
                   execution_benchmark=str(benchmark_path), execution_benchmark_sha256=sha(benchmark_path),
                   baseline_initialization=str(ref / 'initialization.json'),
                   paired_reference=cfg['paired_reference'],
                   launch_command=[sys.executable, supervisor, '--manifest', str(out / 'experiment_manifest.json')],
                   tmux_session='deciform_score_rq_' + name + '_20261105',
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                   comparison='Same retained rows/labels/supports/base initialization/LoRA routing/global batches/epoch/objective/decoder; numeric arm replaces Score readout and avoids candidate token encoding. Microbatch sizes differ after measured memory/throughput checks.',
                   scope='Separate RQ-Score, not A4 or a replacement for the frozen three-task full.')
        job.pop('resume_command', None)
        atomic(out / 'experiment_manifest.json', job)
        arms[name] = {'gpus': gpus, 'manifest': str(out / 'experiment_manifest.json'),
                      'train_rows': counts['train'], 'steps': cfg['total_steps']}
    atomic(root / 'experiment_manifest.json', {
        'arms': arms, 'data': str(data), 'counts': counts, 'excluded_source': 'wmt_en_de',
        'code_sha256': code_hashes, 'main_d2_frozen': str(ref), 'fresh_base_only': True,
        'parent_checkpoint': None, 'epochs': 1, 'global_batch': 64, 'seed': base['seed'],
        'loss': base['distribution_objective'], 'scope': 'Independent RQ-Score, no new A4'})
    print(root)


if __name__ == '__main__':
    main()
