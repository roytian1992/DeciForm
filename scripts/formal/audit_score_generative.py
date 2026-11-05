"""Independently audit cached prompts, split isolation and stored JSON-score errors."""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

from common import BASE, PROJECT, atomic, sha
from score_baselines import json_prompt, json_answer


def strict_number(raw, row, terminal_tokens):
    # Independent parser: strip only actual tokenizer EOS/pad suffixes.
    while True:
        token = next((x for x in terminal_tokens if raw.endswith(x)), None)
        if token is None:
            break
        raw = raw[:-len(token)]
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    def reject(_):
        raise ValueError('nonfinite')
    try:
        obj = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
        assert isinstance(obj, dict) and set(obj) == {'score'}
        value = obj['score']
        assert type(value) in (int, float) and math.isfinite(value)
        support = row['score_support']
        assert support[0] <= value <= support[-1]
        if row['label_kind'] == 'native_integer':
            assert type(value) is int and value in support
        return value
    except (ValueError, TypeError, AssertionError, OverflowError):
        return None


def main():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    root = PROJECT / 'runs/score_generative_integrity_20261106_v1'
    root.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    terminal = sorted({x for x in [tokenizer.eos_token, tokenizer.pad_token] if x}, key=len, reverse=True)
    report = {'status': 'running', 'code_sha256': sha(Path(__file__)), 'panels': {},
              'scope': 'Integrity of local data/encodings/predictions; not a pretraining-contamination audit'}
    atomic(root / 'audit.json', report)
    for panel in ['asap', 'continuous']:
        path = PROJECT / 'data/score_baseline_encodings_20261105_v1' / panel / 'audit.json'
        cache = json.loads(path.read_text())
        result = {'cache_audit_sha256': sha(path), 'splits': {}, 'recomputed_reports': {}}
        populations = {}
        for split, entry in cache['splits'].items():
            assert sha(entry['views_path']) == entry['views_sha256']
            rows = [json.loads(line) for line in open(entry['views_path'])]
            arm = entry['arms']['json_sft']
            assert sha(arm['path']) == arm['sha256']
            with open(arm['path'], 'rb') as f:
                encoded = pickle.load(f)
            assert len(rows) == len(encoded) == entry['rows']
            populations[split] = rows
            for start in range(0, len(rows), 128):
                chunk = rows[start:start+128]
                prompts = [json_prompt(r) for r in chunk]
                # Change the target while holding every input field fixed.
                for row, prompt in zip(chunk, prompts):
                    changed = dict(row)
                    changed['label'] = (row['score_support'][0] if row['label'] != row['score_support'][0]
                                        else row['score_support'][-1])
                    assert json_prompt(changed) == prompt
                    assert json_prompt({k: v for k, v in row.items() if k != 'label'}) == prompt
                prompt_ids = tokenizer(prompts, add_special_tokens=True, truncation=False).input_ids
                answers = tokenizer([json_answer(r) for r in chunk], add_special_tokens=False, truncation=False).input_ids
                for j, (p, a) in enumerate(zip(prompt_ids, answers)):
                    observed = encoded[start+j]
                    assert observed['prompt'] == p and observed['answer'] == a+[tokenizer.eos_token_id]
                atomic(root / 'progress.json', {'panel': panel, 'split': split,
                       'checked': min(start+128, len(rows)), 'total': len(rows), 'updated_unix': time.time()})
            result['splits'][split] = {'rows': len(rows), 'prompt_cache_exact': True,
                'answer_cache_exact': True, 'prompt_independent_of_label': True,
                'views_sha256': entry['views_sha256'], 'encoding_sha256': arm['sha256']}
        train = populations['train']
        heldout = populations['dev'] + populations['test_in_domain']
        for key in ['id', 'group_id']:
            count = len({r[key] for r in train} & {r[key] for r in heldout})
            result['train_heldout_'+key+'_overlap'] = count
            assert count == 0
        text_hash = lambda row: hashlib.sha256(' '.join(row['state']['text'].split()).encode()).hexdigest()
        count = len({text_hash(r) for r in train} & {text_hash(r) for r in heldout})
        result['train_heldout_normalized_state_overlap'] = count
        assert count == 0
        run = PROJECT / 'runs/formal_score_baselines_20261105_v1' / (panel+'_json_sft')
        for split in ['dev', 'test_in_domain']:
            gold = {r['id']: r for r in populations[split]}
            files = sorted((run / 'evaluations').glob('epoch_1.'+split+'.rank*.jsonl'))
            assert len(files) == 2
            predictions = [json.loads(line) for f in files for line in f.open()]
            assert len(predictions) == len(gold) and {p['id'] for p in predictions} == set(gold)
            grouped = defaultdict(list); errors = []; invalid = 0
            for prediction in predictions:
                row = gold[prediction['id']]
                assert float(row['label']) == prediction['label']
                assert row['source_family'] == prediction['source']
                assert row['score_support'] == prediction['support']
                value = strict_number(prediction['raw_output'], row, terminal)
                assert (value is not None) == prediction['valid']
                assert value == prediction['prediction_value']
                span = row['score_support'][-1]-row['score_support'][0]
                error = abs(value-row['label']) if value is not None else span
                assert abs(error-prediction['mae']) < 1e-12
                errors.append(error); grouped[row['source_family']].append(error/span)
                invalid += value is None
            mae = sum(errors)/len(errors)
            macro = sum(sum(v)/len(v) for v in grouped.values())/len(grouped)
            report_path = run / 'evaluations' / ('epoch_1.'+split+'.json')
            observed = json.loads(report_path.read_text())
            assert abs(mae-observed['task/score']['mae']) < 1e-10
            assert abs(macro-observed['macro_source_nmae']) < 1e-10
            result['recomputed_reports'][split] = {'rows': len(errors), 'invalid': invalid,
                'mae': mae, 'macro_source_nmae': macro, 'report_sha256': sha(report_path),
                'prediction_sha256': {str(f): sha(f) for f in files}}
        report['panels'][panel] = result
        atomic(root / 'audit.json', report)
    report['status'] = 'passed'
    atomic(root / 'audit.json', report)
    print(json.dumps({'status': report['status'], 'panels': {
        k: v['recomputed_reports']['test_in_domain'] for k, v in report['panels'].items()}}, indent=2))


if __name__ == '__main__':
    main()
