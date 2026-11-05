"""Freeze two native ASAP loss experiments with identical numeric architecture."""
import copy
from datetime import datetime
import json
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from numeric_score_objectives import NumericScoreObjective


def main():
    reference = PROJECT / 'runs/formal_score_numeric_asap_20261105_v1'
    root = PROJECT / 'runs/formal_score_objectives_20261106_v1'
    original = json.loads((reference / 'config.json').read_text())
    manifest = json.loads((reference / 'experiment_manifest.json').read_text())
    assert (reference / 'complete.json').exists()
    proof = json.loads((reference / 'preflight_passed.json').read_text())
    assert proof['status'] == 'passed' and proof['shared_fresh_d2_initialization_verified']
    data = PROJECT / 'data/score_specialist_asap_fold0_20261104_v1'
    audit = json.loads((data / 'cache/audit.json').read_text())
    assert sha(data / 'cache/audit.json') == original['cache_audit_sha256']
    assert audit['data_manifest_sha256'] == sha(data / 'manifest.json')
    assert {s: d['rows'] for s, d in audit['splits'].items()} == {
        'train': 7784, 'dev': 2593, 'test_in_domain': 2598}
    paths = set(manifest['code_sha256']) | {
        'scripts/formal/numeric_score_objectives.py', 'scripts/formal/models_numeric_objectives.py',
        'scripts/formal/worker_numeric_objectives.py', 'scripts/formal/supervise_numeric_objectives.py',
        'scripts/formal/prepare_numeric_objectives.py', 'tests/test_numeric_score_objectives.py'}
    hashes = {p: sha(PROJECT / p) for p in sorted(paths)}
    for p in ['scripts/formal/models_score_research.py', 'scripts/formal/numeric_score_distribution.py',
              'scripts/formal/independent_candidates.py']:
        assert hashes[p] == manifest['code_sha256'][p], 'Frozen model changed: '+p
    root.mkdir(exist_ok=False)
    for p in paths:
        dest = root / 'code' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / p, dest)
    specs = [('r1_cdf_mean', '4,5', NumericScoreObjective('cdf_mean', mean_weight=1.)),
             ('r2_crps', '6,7', NumericScoreObjective('crps', mean_weight=0.))]
    arms = {}
    for name, gpus, objective in specs:
        out = root / name
        out.mkdir()
        cfg = copy.deepcopy(original)
        cfg.update(output=str(out), label='ASAP numeric objective: '+name, experiment_name=name,
                   distribution_objective=objective.inventory(), task_losses={'score': objective.kind},
                   worker_variant=str(PROJECT / 'scripts/formal/worker_numeric_objectives.py'),
                   method_variant='numeric_score_objective_research', paired_reference=str(reference),
                   score_panel='asap', grid_sizes=[4, 5, 6, 11, 31, 61],
                   objective_description=('Ordered CDF log loss + squared normalized predictive-mean regression, lambda=1'
                       if objective.kind == 'cdf_mean' else 'Pure normalized CRPS, scale=2*D2 alpha'),
                   candidate_positions='not applicable: numeric supports use complete numerical projection',
                   classification_ties='not applicable: fixed native median Score decoding')
        assert cfg['epochs'] == 1 and cfg['train_rows'] == 7784 and cfg['total_steps'] == 122
        assert cfg['parent_checkpoint'] is None and cfg['global_batch'] == 64
        atomic(out / 'config.json', cfg)
        job = copy.deepcopy(manifest)
        supervisor = str(PROJECT / 'scripts/formal/supervise_numeric_objectives.py')
        job.update(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   python=sys.executable, config=str(out / 'config.json'), config_sha256=sha(out / 'config.json'),
                   worker=cfg['worker_variant'], supervisor=supervisor, code_sha256=hashes,
                   code_snapshot=str(root / 'code'), experiment_name=name, gpus=gpus,
                   distribution_objective=objective.inventory(), paired_reference=str(reference),
                   numeric_initialization_reference=str(reference / 'initialization.json'),
                   tmux_session='deciform_score_objectives_'+name+'_20261106',
                   launch_command=[sys.executable, supervisor, '--manifest', str(out / 'experiment_manifest.json')],
                   resume_command=[sys.executable, supervisor, '--manifest', str(out / 'experiment_manifest.json'), '--resume'],
                   comparison='Identical numeric architecture/full initialization/data/order/batching/global batch/epoch/optimizer/median decoder; only Score data objective changes.',
                   execution_benchmark_scope='Existing exact-architecture ASAP memory gate; new loss backward and diagnostics validated in each distributed preflight before training.',
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip())
        atomic(out / 'experiment_manifest.json', job)
        arms[name] = {'manifest': str(out / 'experiment_manifest.json'), 'gpus': gpus,
                      'objective': objective.inventory()}
    atomic(root / 'experiment_manifest.json', {
        'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'arms': arms,
        'data': str(data), 'data_counts': manifest['data_counts'], 'cache_audit_sha256': original['cache_audit_sha256'],
        'reference': str(reference), 'architecture': 'unchanged numeric CDF D2',
        'fresh_base_only': True, 'parent_checkpoint': None, 'epochs': 1, 'global_batch': 64,
        'steps': 122, 'seed': original['seed'], 'code_sha256': hashes,
        'main_d2_unchanged': True, 'source_languages_added': [],
        'references_for_results': {
            'numeric_cdf': str(reference),
            'independent_d2': str(PROJECT / 'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap'),
            'kev': str(PROJECT / 'runs/formal_score_baselines_20261105_v1/asap_kev'),
            'generative_lora': str(PROJECT / 'runs/formal_score_baselines_20261105_v1/asap_json_sft')},
        'scope': 'Two Score-only objectives on native ASAP; no A4 or main-model retraining'})
    print(root)


if __name__ == '__main__':
    main()
