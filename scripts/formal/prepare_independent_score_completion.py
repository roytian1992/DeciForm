"""Freeze the three missing Score comparisons after A1--A3 completion."""
import copy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import asap_score
import score_specialist
from common import PROJECT, atomic, sha


def main():
    reference = PROJECT / 'runs/formal_independent_20261105_v1'
    asap_reference = PROJECT / 'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap'
    source_cfg = json.loads((asap_reference / 'config.json').read_text())
    assert source_cfg['parent_checkpoint'] is None
    root = PROJECT / 'runs/formal_independent_score_completion_20261105_v1'
    # Preserve every preceding run; never reuse an existing output directory.
    root.mkdir(exist_ok=False)
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    old_manifest = json.loads((reference / 'experiment_manifest.json').read_text())
    paths = set(old_manifest['code_sha256']) | {
        'scripts/formal/models_score_specialist.py', 'scripts/formal/asap_score.py',
        'scripts/formal/score_specialist.py', 'scripts/formal/worker_score_specialist.py',
        'scripts/formal/worker_independent_asap_score.py',
        'scripts/formal/models_independent_score_specialist.py',
        'scripts/formal/worker_independent_score_specialist.py',
        'scripts/formal/prepare_independent_score_completion.py',
        'tests/test_independent_score_specialist.py',
    }
    code_hashes = {name: sha(PROJECT / name) for name in sorted(paths)}
    model_paths = [name for name in old_manifest['code_sha256']
                   if name.startswith('src/') or '/models_' in name or
                   name.endswith(('independent_candidates.py', 'common.py', 'model.py',
                                  'kev_ours.py', 'kev_ours_v4.py', 'kev_ours_v5.py'))]
    for name in model_paths:
        assert code_hashes[name] == old_manifest['code_sha256'][name], name
    for name, digest in code_hashes.items():
        dest = root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        assert sha(dest) == digest
    specs = [('asap_ce', 'asap', 'soft_ce', '2,3', 'A1_no_task_routing'),
             ('continuous_cdf', 'continuous', 'soft_cdf', '4,5', 'A2_no_quadratic'),
             ('continuous_ce', 'continuous', 'soft_ce', '6,7', 'A3_no_head_geometry')]
    arms = []
    for name, panel_name, kind, gpus, dependency in specs:
        panel = asap_score if panel_name == 'asap' else score_specialist
        data = PROJECT / 'data' / panel.PROTOCOL
        audit_path = data / 'cache/audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'passed' and sha(data / 'manifest.json') == audit['data_manifest_sha256']
        counts = {s: audit['splits'][s]['rows'] for s in ('train', 'dev', 'test_in_domain')}
        assert counts == ({'train': 7784, 'dev': 2593, 'test_in_domain': 2598} if panel_name == 'asap'
                          else {'train': 11457, 'dev': 3500, 'test_in_domain': 3379})
        out = root / name
        out.mkdir()
        cfg = copy.deepcopy(source_cfg)
        steps = (counts['train'] + 63) // 64
        cfg.update(data=str(data), cache_audit_sha256=sha(audit_path), output=str(out),
                   label=f'Independent D2 Score: {panel_name} {kind}',
                   experiment_name=name, method_variant='independent_score_specialist',
                   worker_variant=str(PROJECT / 'scripts/formal/worker_independent_score_specialist.py'),
                   score_panel=panel_name, evaluation_protocol=panel.PROTOCOL,
                   grid_sizes=[4, 5, 6, 11, 31, 61] if panel_name == 'asap' else [51, 101],
                   train_rows=counts['train'], total_steps=steps,
                   schedule=f'OneCycleLR, {steps} steps, pct_start=0.3',
                   checkpoint_policy_candidates=['all'],
                   task_losses={'score': kind},
                   distribution_objective=score_specialist.SpecialistObjective(kind).inventory(),
                   objective_description='Categorical CE' if panel_name == 'asap' else f'Adjacent-grid interpolated raw targets: {kind}',
                   candidate_sampling='none; complete native Score support',
                   evaluation_candidates='complete native Score support in train/dev/test',
                   order_audit='not applicable: Score only; no permutations',
                   primary_metrics={'score': 'median native MAE per source; equal-source NMAE'},
                   reference_run=str(reference), paired_reference=str(asap_reference) if panel_name == 'asap'
                   else str(root / ('continuous_ce' if kind == 'soft_cdf' else 'continuous_cdf')))
        atomic(out / 'config.json', cfg)
        dep = PROJECT / 'runs/formal_independent_ablations_20261105_v1' / dependency / 'evaluation_recovery_20261105_v2'
        assert json.loads((dep / 'status.json').read_text())['phase'] == 'complete'
        assert json.loads((dep / 'complete.json').read_text())['status'] == 'complete'
        manifest_path = out / 'experiment_manifest.json'
        supervisor = str(PROJECT / 'scripts/formal/supervise_score_specialist.py')
        command = [sys.executable, supervisor, '--manifest', str(manifest_path)]
        manifest = {
            'created': created, 'python': sys.executable,
            'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'triton')},
            'config': str(out / 'config.json'), 'config_sha256': sha(out / 'config.json'),
            'worker': cfg['worker_variant'], 'supervisor': supervisor,
            'code_sha256': code_hashes, 'code_snapshot': str(root / 'code'),
            'experiment_name': name, 'gpus': gpus, 'world_size': 2,
            'environment_variables': old_manifest['environment_variables'],
            'baseline_initialization': str(reference / 'initialization.json'),
            'ablation': cfg['ablation'], 'distribution_objective': cfg['distribution_objective'],
            'evaluation_protocol': panel.PROTOCOL, 'expected_trainable_parameters': 46304262,
            'data': str(data), 'data_counts': counts, 'data_manifest_sha256': sha(data / 'manifest.json'),
            'cache_audit_sha256': sha(audit_path), 'parent_checkpoint': None,
            'reference_run': str(reference), 'paired_reference': cfg['paired_reference'],
            'grid_sizes': cfg['grid_sizes'], 'completed_dependency': str(dep / 'complete.json'),
            'completed_dependency_sha256': sha(dep / 'complete.json'),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
            'tmux_session': 'deciform_ind_score_' + name + '_20261105',
            'launch_command': command, 'resume_command': command + ['--resume'],
            'comparison': 'Same fresh initialization, independent architecture, full supports, source splits, one epoch, seed, global batch, LR schedule and chunk-8 execution; only Score objective differs within each panel.',
            'reporting': 'Median native MAE; equal-source NMAE across scales; expectation diagnostic. No Score permutations. Original labels never rounded.',
        }
        atomic(manifest_path, manifest)
        arms.append({'name': name, 'manifest': str(manifest_path), 'gpus': gpus,
                     'train_rows': counts['train'], 'steps': steps})
    atomic(root / 'experiment_manifest.json', {'created': created, 'arms': arms,
           'code_sha256': code_hashes, 'reference_model_hashes_verified': model_paths,
           'live_asap_d2_reference': str(asap_reference)})
    print(json.dumps({'root': str(root), 'arms': arms}))


if __name__ == '__main__':
    main()
