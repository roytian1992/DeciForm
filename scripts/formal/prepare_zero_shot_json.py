"""Freeze unadapted-model inference panels and GPU-release prerequisites."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer
from common import PROJECT, BASE, atomic, sha
from zero_shot_protocol import PANELS, prompt_ids, views_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['base', 'instruct'], default=['base'])
    args = parser.parse_args()
    root = PROJECT / 'runs/formal_zero_shot_json_20261105_v1'
    if root.exists(): raise FileExistsError(root)
    main_audit = PROJECT / 'data/formal_200k_20261101_v1/cache/audit.json'
    score_audits = {p: PROJECT / f'data/score_baseline_encodings_20261105_v1/{p}/audit.json'
                    for p in ('asap', 'continuous')}
    panels = {}
    audits = {}
    for panel in PANELS:
        audit_path = score_audits.get(panel, main_audit)
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'passed'
        desc = audit['splits']['test_in_domain']
        assert desc['native_evaluation'] and sha(desc['views_path']) == desc['views_sha256']
        rows = [json.loads(line) for line in open(desc['views_path'])]
        task = 'choice' if panel == 'choice_order' else 'noul' if panel == 'noul_order' else None
        if task: rows = [r for r in rows if r['task'] == task]
        assert all(r['candidate_mode'] == 'complete' for r in rows)
        count = len(rows)
        assert count == {'main': 15000, 'asap': 2598, 'continuous': 3379,
                         'choice_order': 5000, 'noul_order': 5000}[panel]
        panels[panel] = {'views_path': desc['views_path'], 'views_sha256': desc['views_sha256'],
            'audit_path': str(audit_path), 'audit_sha256': sha(audit_path), 'task': task,
            'rows': count, 'views': views_for(panel), 'predictions': count * len(views_for(panel)),
            'reference_json_cache': desc['arms']['json_sft'],
            'reference_max_new_tokens': desc['max_json_response_tokens'] + 8}
        audits[str(audit_path)] = sha(audit_path)
    model_info = {
        'base': {'path': str(BASE.resolve()), 'prompt_mode': 'matched_base',
                 'label': 'Qwen3.5-9B-Base, zero-shot, no task fine-tuning',
                 'role': 'Exact same pretrained backbone and prompt interface as JSON-generative LoRA'},
        'instruct': {'path': str((PROJECT.parent / 'models/Qwen3.5-9B').resolve()),
                     'prompt_mode': 'official_chat',
                     'label': 'Qwen3.5-9B official instruct, zero-shot on our tasks',
                     'role': 'Practical off-the-shelf reference; upstream post-training, no task-data adaptation'},
    }
    models = {}
    tokenizer_checks = {}
    for name in dict.fromkeys(args.models):
        spec = model_info[name]
        path = Path(spec['path'])
        files = [p for p in path.iterdir() if p.is_file() and
                 (p.suffix in ('.json', '.safetensors') or p.name in ('merges.txt', 'README.md'))]
        assert len(list(path.glob('model*.safetensors'))) == 4
        spec['files_sha256'] = {str(p.resolve()): sha(p) for p in sorted(files)}
        spec['max_new_tokens'] = {panel: 64 if name == 'instruct' else d['reference_max_new_tokens']
                                  for panel, d in panels.items()}
        spec['max_new_tokens']['choice_order'] = 17 if name == 'base' else 64
        spec['max_new_tokens']['noul_order'] = 17 if name == 'base' else 64
        spec['training_rows'] = 0
        spec['optimizer_updates'] = 0
        spec['task_checkpoint'] = None
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        checks = {}
        for panel in ('main', 'asap', 'continuous'):
            desc = panels[panel]
            rows = [json.loads(line) for line in open(desc['views_path'])]
            cache_desc = desc['reference_json_cache']
            assert sha(cache_desc['path']) == cache_desc['sha256']
            with open(cache_desc['path'], 'rb') as handle: cache = pickle.load(handle)
            ids = sorted(set(range(min(32, len(rows)))) | {max(range(len(rows)), key=lambda i: len(cache[i]['prompt']))})
            lengths = []
            for i in ids:
                encoded = prompt_ids(rows[i], panel, 'original', tok, spec['prompt_mode'])
                if name == 'base': assert encoded == cache[i]['prompt'], (panel, rows[i]['id'])
                lengths.append(len(encoded))
            checks[panel] = {'examples': len(ids), 'max_checked_input_tokens': max(lengths),
                             'exact_prompt_token_parity_with_trained_baseline': name == 'base'}
        models[name] = spec
        tokenizer_checks[name] = checks
    root.mkdir(exist_ok=False)
    code = ['scripts/formal/' + name for name in ('common.py', 'score_baselines.py',
            'score_specialist.py', 'score_distribution.py', 'order_audit.py',
            'classification_order_audit.py', 'zero_shot_protocol.py', 'worker_zero_shot_json.py',
            'prepare_zero_shot_json.py', 'schedule_zero_shot_json.py')]
    code += ['tests/test_zero_shot_json.py']
    hashes = {path: sha(PROJECT / path) for path in code}
    for path in code:
        dst = root / 'code' / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / path, dst)
    deps = {
        '0,1': PROJECT / 'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap/status.json',
        '2,3': PROJECT / 'runs/formal_independent_score_completion_20261105_v1/asap_ce/status.json',
        '4,5': PROJECT / 'runs/formal_independent_score_completion_20261105_v1/continuous_cdf/status.json',
        '6,7': PROJECT / 'runs/formal_independent_score_completion_20261105_v1/continuous_ce/status.json',
    }
    manifest = {'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'models': models,
        'panels': panels, 'python': sys.executable, 'cwd': str(PROJECT),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers')},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        'code_sha256': hashes, 'input_audits_sha256': audits,
        'worker': str(PROJECT / 'scripts/formal/worker_zero_shot_json.py'),
        'shards_per_model': 4, 'token_budget': 16384, 'max_batch_rows': 8,
        'max_context_tokens': 262144, 'environment_variables': {'PYTHONUNBUFFERED': '1',
             'TOKENIZERS_PARALLELISM': 'false', 'FLA_DISABLE_TENSOR_CACHE': '1'},
        'gpu_release_dependencies': {g: str(dep) for pair, dep in deps.items() for g in pair.split(',')},
        'max_memory_mib_before_launch': 1000,
        'decoding': 'greedy, no few-shot examples, no constrained decoder, no JSON repair',
        'input_policy': 'Original full test inputs/candidates/ranges; no truncation, sampling or target rounding',
        'metrics': 'Choice/Noul accuracy, native Score MAE including full-range invalid penalty, validity; no fabricated probabilities',
        'score_aggregate': 'Per-source native MAE; equal-source NMAE across ranges',
        'position_protocol': 'Choice: union of frozen new-D2 seven orders and legacy five orders; report both matched subsets separately. Noul: explicit instruction list original/reverse on all 5000 rows. Score excluded.',
        'expected_predictions_per_model': sum(d['predictions'] for d in panels.values()),
        'automatic_model_selection': False, 'external_panel': 'deferred',
        'launch_command': [sys.executable, str(PROJECT / 'scripts/formal/schedule_zero_shot_json.py'), '--root', str(root)],
        'tmux_session': 'deciform_zero_shot_json_20261105'}
    atomic(root / 'tokenizer_validation.json', tokenizer_checks)
    atomic(root / 'experiment_manifest.json', manifest)
    print(json.dumps({'root': str(root), 'models': list(models),
                      'predictions_per_model': manifest['expected_predictions_per_model'], 'panels': panels}, ensure_ascii=False))


if __name__ == '__main__': main()
