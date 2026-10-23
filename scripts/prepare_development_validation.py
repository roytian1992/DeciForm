"""Create disjoint, hash-pinned development panels without altering training."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path[:0] = [str(PROJECT/'src'), str(WORKSPACE/'RSI_Research'), str(PROJECT/'scripts')]
from deciform.data_protocol import raw_choice, text_sha256, _parquet_rows
from deciform.development_validation import sample_key, select_panels
from deciform.joint_data import atomic_json, file_sha256
from deciform.choice import render_choice_prompts
from generate_f2_decisions import split_configs, sample_facts, policy_and_label, TEMPLATES, OPTIONS, canonical_hash


def generated_rows(family, seed):
    rules = split_configs(family)['dev']
    rng = random.Random(f'{seed}:{family}:expanded_development')
    seen = set()
    for _ in range(100000):
        rule = rng.choice(rules)
        facts = sample_facts(family, rng)
        key = canonical_hash({'rule': rule, 'facts': facts})
        if key in seen:
            continue
        seen.add(key)
        policy, label = policy_and_label(family, rule, facts)
        text = '; '.join(f'{k}={str(v).lower()}' for k, v in facts.items())
        options = list(OPTIONS[family])
        yield {'id': f'f2_expanded/{family}/{key}', 'source': 'generated/f2_decisions_v2',
               'source_split': 'dev', 'state': {'text': TEMPLATES[2].format(policy=policy, facts=text)},
               'question': {'type': 'choice', 'instructions': 'Which decision follows from the policy and case facts?', 'options': options},
               'label': label, 'label_index': options.index(label),
               'provenance': {'label_source': 'deterministic_program', 'rule_id': f'{family}_v2',
                              'rule_config': rule, 'rule_spec_sha256': canonical_hash(rule),
                              'facts': facts, 'generator_seed': seed, 'template_id': 2}}
        if len(seen) == 12000:
            return


def source_candidates(root, seed, repairs):
    from datasets import load_from_disk
    master_path = root/'kev_upstream/manifest.json'
    master = json.loads(master_path.read_text())
    selected = {'ag_news': ('fancyzhx/ag_news','test'), 'imdb': ('stanfordnlp/imdb','test'),
                'amazon_reviews': ('SetFit/amazon_reviews_multi_en','validation'),
                'mnli': ('nyu-mll/multi_nli','validation_matched'), 'yelp': ('Yelp/yelp_review_full','test')}
    for name, (repo, split) in selected.items():
        info = next(x for x in master['datasets'] if x['repo'] == repo)
        folder = Path(info['output_path'])/split
        ds = load_from_disk(str(folder))
        labels = [str(i) for i in range(5)] if name == 'amazon_reviews' else list(ds.features['label'].names)
        rows = []
        for i, native in enumerate(ds):
            # Official test-only datasets: preregister half by native index.
            # The other half is not converted or used as development data.
            native_id = f'{name}/{split}/{i:07d}'
            if split == 'test' and int(hashlib.sha256(f'{seed}:partition:{native_id}'.encode()).hexdigest(),16) % 2:
                continue
            row = raw_choice(native, source=repo, name=name, index=i, labels=labels, revision=info['revision'])
            if row:
                row.update(id=native_id, source_split=split)
                rows.append(row)
        yield name, rows, {'native_split': split, 'native_count': len(ds), 'revision': info['revision'],
                         'test_policy': 'hash-half released for development; complementary half excluded' if split=='test' else 'official validation',
                         'raw_files': {str(p): file_sha256(p) for p in sorted(folder.glob('*.arrow'))},
                         'source_manifest_sha256': file_sha256(master_path)}
    path = root/'hf_fallback/google__civil_comments_manifest.json'
    info = json.loads(path.read_text())
    rawroot = root/'hf_fallback/google__civil_comments'
    files = {str(rawroot/x['path']): x['sha256'] for x in info['splits']['validation']['files']}
    for p, digest in list(files.items()):
        if p in repairs:
            repair = repairs[p]
            evidence = Path(repair['evidence'])
            if (repair['old_sha256']!=digest or repair['revision']!=info['revision']
                    or file_sha256(evidence)!=repair['evidence_sha256']):
                raise ValueError('invalid integrity repair provenance')
            metadata = json.loads(evidence.read_text())
            matched = [x for x in metadata['files'] if x.get('path')==str(Path(p).relative_to(rawroot))]
            if (metadata['revision']!=info['revision'] or len(matched)!=1
                    or matched[0]['lfs']['oid']!=repair['correct_sha256']):
                raise ValueError('repair differs from pinned upstream LFS metadata')
            digest = files[p] = repair['correct_sha256']
        if file_sha256(Path(p)) != digest:
            raise ValueError(f'changed source: {p}')
    rows = []
    for i, native in enumerate(_parquet_rows([Path(p) for p in files], ['text','toxicity'])):
        score = native['toxicity']
        if score is None or not math.isfinite(score):
            continue
        index = int(score >= 0.5)
        rows.append({'id': f'civil_comments/validation/{i:07d}', 'source': 'google/civil_comments', 'source_split': 'validation',
                     'state': {'text': (native['text'] or '').strip()}, 'question': {'type': 'choice',
                     'instructions': 'Is this comment toxic under the selected toxicity threshold?', 'options': ['non_toxic','toxic']},
                     'label': ['non_toxic','toxic'][index], 'label_index': index,
                     'provenance': {'label_source': 'derived_threshold', 'target_score': score, 'threshold': 0.5,
                                    'raw_row_index': i, 'source_revision': info['revision']}})
    yield 'civil_comments', rows, {'native_split': 'validation', 'native_count': info['splits']['validation']['rows'],
                                   'raw_files': files, 'source_manifest_sha256': file_sha256(path)}
    path = root/'kev_upstream/devtools_sources_manifest.json'
    info = json.loads(path.read_text())
    item = next(x for x in info['sources'] if x['group']=='aegis' and x['filename']=='validation.json')
    aart = next(x for x in info['sources'] if x['group']=='aart_filter')
    for entry in (item, aart):
        if file_sha256(Path(entry['path'])) != entry['expected_sha256']:
            raise ValueError('Aegis/AART hash mismatch')
    from pyarrow.parquet import read_table
    aart_text = {' '.join(x.casefold().split()) for x in read_table(aart['path'], columns=['prompt'])['prompt'].to_pylist() if x}
    native_rows = json.loads(Path(item['path']).read_text())
    rows = []
    for i, native in enumerate(native_rows):
        prompt, label = (native.get('prompt') or '').strip(), native.get('prompt_label')
        if (not prompt or prompt=='REDACTED' or native.get('reconstruction_id_if_redacted') or label not in ('safe','unsafe')
                or native.get('prompt_label_source')!='human' or ' '.join(prompt.casefold().split()) in aart_text):
            continue
        rows.append({'id': f"aegis/validation/{native['id']}", 'source': item['repo'], 'source_split': 'validation',
                     'state': {'text': prompt}, 'question': {'type': 'choice',
                     'instructions': 'Is this prompt safe under the content safety policy?', 'options': ['safe','unsafe']},
                     'label': label, 'label_index': ['safe','unsafe'].index(label),
                     'provenance': {'label_source': 'human_prompt_label', 'raw_row_index': i, 'source_revision': item['revision']}})
    yield 'aegis', rows, {'native_split': 'validation', 'native_count': len(native_rows),
                        'raw_files': {x['path']:x['expected_sha256'] for x in (item,aart)},
                        'source_manifest_sha256': file_sha256(path)}
    for family in ('eligibility','refund'):
        yield 'f2_'+family, list(generated_rows(family,seed)), {'native_split': 'new cases, existing dev policy partition and template 2',
                          'distribution': 'uniform dev rule and original sample_facts; unique cases; not deployment prevalence',
                          'raw_files': {str(PROJECT/'scripts/generate_f2_decisions.py'): file_sha256(PROJECT/'scripts/generate_f2_decisions.py')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint-plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--integrity-repairs',type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.joint_plan.read_text())
    pools, origins = {}, {}
    repairs=json.loads(args.integrity_repairs.read_text()) if args.integrity_repairs else {}
    for name, rows, origin in source_candidates(WORKSPACE/'RSI_Research/data', args.seed, repairs):
        # Uniform, label-independent computational presample. No hard text truncation.
        raw_count = len(rows)
        rows.sort(key=lambda row: sample_key(row, args.seed+3))
        pools[name] = rows[:12000]
        origins[name] = {**origin, 'valid_native_records': raw_count, 'presample_count': len(pools[name])}
        print(json.dumps({'event':'candidate_pool', 'source':name, 'count':len(pools[name])}), flush=True)
    candidates = {text_sha256(row) for rows in pools.values() for row in rows}
    blocked = set()
    scans = []
    paths = {}
    for source in plan['sources']:
        paths[Path(source['path'])] = (source['sha256'], source['count'], 'joint_training')
        ep = Path(source['episode_manifest'])
        if file_sha256(ep) != source['episode_manifest_sha256']:
            raise ValueError('episode manifest changed')
        manifest = json.loads(ep.read_text())
        for role in ('D_mv','D_cal_fit','D_cal_risk','D_assess_visible'):
            paths[ep.parent/f'{role}.jsonl'] = (manifest['outputs'][role]['sha256'],manifest['outputs'][role]['count'],role)
    base = WORKSPACE/'RSI_Research/data/typed_adapt_bench/protocols/base_choice_pilot_20261015_v2/manifest.json'
    for role, value in json.loads(base.read_text())['outputs'].items():
        paths[Path(value['path'])] = (value['sha256'],value['count'],'theta0_'+role)
    for path, (digest, count, role) in paths.items():
        observed = hashlib.sha256()
        n = matches = 0
        with path.open('rb') as handle:
            for line in handle:
                observed.update(line)
                if not line.strip():
                    continue
                n += 1
                key = text_sha256(json.loads(line))
                if key in candidates:
                    blocked.add(key)
                    matches += 1
        if n != count or observed.hexdigest() != digest:
            raise ValueError(f'exclusion input changed: {path}')
        scans.append({'path': str(path), 'sha256':digest, 'count':n, 'role':role, 'candidate_matches':matches})
        print(json.dumps({'event':'exclusion_scan','source':path.parent.name,'role':role,'count':n,'matches':matches}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sources = []
    global_seen = set()
    for name, rows in pools.items():
        eligible = []
        reasons = Counter()
        for row in rows:
            key = text_sha256(row)
            if not row['state']['text']:
                reasons['empty'] += 1
            elif key in blocked:
                reasons['overlap_training_or_existing_visible'] += 1
            elif key in global_seen:
                reasons['duplicate_text'] += 1
            else:
                global_seen.add(key)
                eligible.append(row)
        valid = []
        for start in range(0,len(eligible),64):
            batch = eligible[start:start+64]
            prompts = [prompt for row in batch for prompt in render_choice_prompts(row)[1]]
            lengths = [len(ids) for ids in tokenizer(prompts, truncation=False, add_special_tokens=True)['input_ids']]
            cursor = 0
            for row in batch:
                size = len(row['question']['options'])
                if max(lengths[cursor:cursor+size]) <= 1024:
                    valid.append(row)
                else:
                    reasons['over_1024_tokens'] += 1
                cursor += size
        panels = select_panels(valid, seed=args.seed, target=1000)
        for panel, records in panels.items():
            if not records:
                raise ValueError(f'empty {name}/{panel}')
            path = out/f'{name}_{panel}.jsonl'
            for row in records:
                row['split'] = 'expanded_development_'+panel
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
            sources.append({'name': name+'__'+panel, 'dataset':name, 'panel':panel, 'path':str(path),
                            'sha256':file_sha256(path),'count':len(records),'max_prompt_tokens':1024,
                            'label_counts':dict(Counter(row['label'] for row in records))})
        origins[name].update(eligible_after_exclusion_and_length=len(valid), exclusions=dict(reasons))
        print(json.dumps({'event':'panels','source':name,'sizes':{k:len(v) for k,v in panels.items()}}),flush=True)
    # Independently verify the materialized outputs against the full scanned exclusion set.
    selected_keys = []
    for source in sources:
        with Path(source['path']).open() as handle:
            selected_keys.extend(text_sha256(json.loads(line)) for line in handle)
    if len(set(selected_keys)) != len(selected_keys) or set(selected_keys)&blocked:
        raise ValueError('output overlap detected')
    atomic_json(out/'manifest.json', {'version':1,'status':'complete','purpose':'development_monitoring_only',
        'seed':args.seed,'target_per_source_per_panel':1000,'joint_plan':str(args.joint_plan.resolve()),
        'joint_plan_sha256':file_sha256(args.joint_plan),'sources':sources,
        'record_count':sum(x['count'] for x in sources), 'origins':origins, 'exclusion_audit':scans,
        'unique_text_count':len(selected_keys), 'cross_panel_text_overlap':0,
        'excluded_candidate_text_count':len(blocked), 'sealed_hidden_or_transfer_accessed':False,
        'integrity_repairs':repairs,
        'temperature_policy':'inherited checkpoint temperature; monitoring only, no fitting',
        'natural_definition':'uniform sample of deduplicated, <=1024 token eligible source; synthetic panels follow generator prior',
        'unexpanded_sources':{'cfpb':'all eligible local native data already assigned to training or existing visible roles; retain original panels, new independent data required'},
        'code': {str(p):file_sha256(p) for p in (Path(__file__),PROJECT/'src/deciform/development_validation.py',PROJECT/'src/deciform/data_protocol.py')},
        'tokenizer_config_sha256':file_sha256(args.model_dir/'tokenizer_config.json')})
    print(json.dumps({'status':'complete','manifest':str(out/'manifest.json'),'record_count':len(selected_keys)}),flush=True)


if __name__=='__main__':
    main()
