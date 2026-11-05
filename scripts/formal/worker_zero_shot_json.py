"""One GPU, one immutable unadapted model shard; resumable full-input inference."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('FLA_DISABLE_TENSOR_CACHE', '1')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import PROJECT, SEED, atomic, sha
from zero_shot_protocol import PANELS, prompt_ids, views_for, strip_terminal, record_prediction, summarize


def batches(bundles, budget, max_rows):
    cur, longest = [], 0
    for bundle in sorted(bundles, key=lambda b: (-b['length'], b['row']['id'])):
        length = bundle['length']
        if cur and ((len(cur) + 1) * max(longest, length) > budget or len(cur) >= max_rows):
            yield cur
            cur, longest = [], 0
        cur.append(bundle)
        longest = max(longest, length)
    if cur: yield cur


def pad(sequences, pad_id):
    n = max(map(len, sequences))
    ids = torch.full((len(sequences), n), pad_id, dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    for i, values in enumerate(sequences):
        ids[i, -len(values):] = torch.tensor(values, device='cuda')
        mask[i, -len(values):] = 1
    return ids, mask


def run(root, name, shard):
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    spec = manifest['models'][name]
    out = root / name / f'shard_{shard}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists(): return
    for path, digest in manifest['code_sha256'].items(): assert sha(PROJECT / path) == digest, path
    for path, digest in spec['files_sha256'].items(): assert sha(path) == digest, path
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    def progress(phase, **extra):
        atomic(out / 'progress.json', {'phase': phase, 'model': name, 'shard': shard,
               'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'updated_unix': time.time(),
               'seconds': time.time() - started, **extra})
    progress('loading_model')
    tokenizer = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(spec['path'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0})
    model.requires_grad_(False).eval()
    assert not any('lora_' in n or 'adapter' in n for n, _ in model.named_parameters())
    assert not any(p.requires_grad for p in model.parameters())
    stop_ids = [tokenizer.eos_token_id]
    if spec['prompt_mode'] == 'official_chat':
        stop_ids += [tokenizer.convert_tokens_to_ids('<|im_end|>')]
    stop_ids = sorted(set(stop_ids))
    terminal = [tokenizer.convert_ids_to_tokens(i) for i in stop_ids + [tokenizer.pad_token_id]]
    atomic(out / 'model_loaded.json', {'base_path': spec['path'], 'model_files_sha256': spec['files_sha256'],
           'optimizer_updates': 0, 'trainable_parameters': 0, 'no_adapter_or_custom_head': True,
           'prompt_mode': spec['prompt_mode'], 'eos_token_ids': stop_ids})
    def predict(rs, es, panel, view):
        ids, mask = pad(es, tokenizer.pad_token_id)
        cap = spec['max_new_tokens'][panel]
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cap,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids, logits_to_keep=1)
        result = []
        for row, enc, token_ids in zip(rs, es, output[:, ids.shape[1]:].cpu().tolist()):
            raw = tokenizer.decode(token_ids, skip_special_tokens=False)
            clean = strip_terminal(raw, terminal)
            record = record_prediction(row, panel, view, raw, clean)
            record.update(input_tokens=len(enc), generated_tokens=next(
                (i + 1 for i, token in enumerate(token_ids) if token in stop_ids), len(token_ids)),
                output_cap_reached=not any(token in stop_ids for token in token_ids),
                input_sha256=__import__('hashlib').sha256(json.dumps(enc, separators=(',', ':')).encode()).hexdigest())
            result.append(record)
        return result
    for panel in PANELS:
        desc = manifest['panels'][panel]
        assert sha(desc['views_path']) == desc['views_sha256']
        rows = [json.loads(line) for line in open(desc['views_path'])]
        if desc.get('task'): rows = [r for r in rows if r['task'] == desc['task']]
        assert len(rows) == desc['rows'] and len({r['id'] for r in rows}) == len(rows)
        assert all(r['candidate_mode'] == 'complete' for r in rows)
        mine = rows[shard::manifest['shards_per_model']]
        names = views_for(panel)
        journal_path = out / f'{panel}.predictions.jsonl'
        records = [json.loads(line) for line in journal_path.open()] if journal_path.exists() else []
        seen = {(r['id'], r['view']) for r in records}
        assert len(seen) == len(records)
        expected = {(r['id'], view) for r in mine for view in names}
        assert seen.issubset(expected)
        bundles = []
        progress('encoding', panel=panel, completed=len(seen), expected=len(expected))
        for i, row in enumerate(mine):
            encs = {view: prompt_ids(row, panel, view, tokenizer, spec['prompt_mode']) for view in names}
            lengths = [len(e) for e in encs.values()]
            if max(lengths) + spec['max_new_tokens'][panel] > manifest['max_context_tokens']:
                raise ValueError(f"Full input exceeds declared model context: {row['id']}")
            bundles.append({'row': row, 'encs': encs, 'length': max(lengths)})
            if (i + 1) % 250 == 0: progress('encoding', panel=panel, encoded=i + 1, total=len(mine))
        chunks = list(batches(bundles, manifest['token_budget'], manifest['max_batch_rows']))
        # Longest input, real output parsing, and reporting are exercised before the panel loop.
        if not (out / f'{panel}.preflight.json').exists():
            longest = chunks[0][0]
            checks = predict([longest['row']], [longest['encs']['original']], panel, 'original')
            assert len(checks) == 1 and 'valid' in checks[0]
            if not panel.endswith('_order'): summarize(panel, checks)
            atomic(out / f'{panel}.preflight.json', {'status': 'passed',
                'longest_input_tokens': longest['length'], 'prediction': checks[0],
                'no_input_truncation': True, 'optimizer_updates': 0})
        with journal_path.open('a', buffering=1) as journal:
            for batch_index, batch in enumerate(chunks):
                for view in names:
                    missing = [(b['row']['id'], view) not in seen for b in batch]
                    if not any(missing): continue
                    # Preserve original padding/batch on resume, including already recorded rows.
                    predictions = predict([b['row'] for b in batch], [b['encs'][view] for b in batch], panel, view)
                    for needed, prediction in zip(missing, predictions):
                        if needed:
                            journal.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                            records.append(prediction)
                            seen.add((prediction['id'], view))
                progress('evaluating', panel=panel, completed=len(seen), expected=len(expected),
                         batch=batch_index + 1, batches=len(chunks),
                         peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        assert seen == expected
        atomic(out / f'{panel}.json', {'status': 'complete', 'rows': len(mine), 'predictions': len(records),
               'report': summarize(panel, records), 'predictions_sha256': sha(journal_path)})
    atomic(out / 'complete.json', {'status': 'complete', 'optimizer_updates': 0, 'panels': PANELS,
           'seconds': time.time() - started, 'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30})
    progress('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--shard', required=True, type=int)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.model, args.shard)
    except BaseException as exc:
        out = args.root / args.model / f'shard_{args.shard}'
        atomic(out / 'failure.json', {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
