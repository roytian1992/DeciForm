"""Real-model, zero-update throughput/memory gate for Score-only execution."""
import argparse
import json
from pathlib import Path
import pickle
import random
import time

import torch

from common import BASE, PROJECT, SEED, atomic, sha
from models_score_distribution import A4_FLAGS
from models_score_research import BatchedD2, NumericD2
from models_independent_score_specialist import FormalModel as ChunkedD2
from score_specialist import support, prediction_row, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['d2_batched', 'numeric_cdf'], required=True)
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--policy', choices=['half', 'all', 'none'], default='half')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_paths = ['scripts/formal/benchmark_score_research.py',
                    'scripts/formal/models_score_research.py',
                    'scripts/formal/numeric_score_distribution.py',
                    'scripts/formal/independent_candidates.py']
    code_hashes = {name: sha(PROJECT / name) for name in source_paths}
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = PROJECT / 'data/score_specialist_stsb_enzh_20261105_v1'
    audit = json.loads((data / 'cache/audit.json').read_text())
    entry = audit['splits']['train']
    rows = [json.loads(line) for line in open(entry['views_path'])]
    with open(entry['arms']['kev']['path'], 'rb') as f:
        encs = pickle.load(f)
    model_type = BatchedD2 if args.variant == 'd2_batched' else NumericD2
    t0 = time.time()
    atomic(out / 'progress.json', {'phase': 'loading', 'time': time.time(), 'variant': args.variant})
    model = model_type('tcdl', BASE, ablation=A4_FLAGS,
                       distribution_objective={'kind': 'soft_cdf', 'ordinal_scale': 3.0383850790577127})
    model.set_checkpoint_policy(args.policy)
    params = [p for p in model.parameters() if p.requires_grad]
    samples = []
    for source in ['stsb', 'wmt_en_zh']:
        idx = max((i for i, r in enumerate(rows) if r['source_family'] == source), key=lambda i: len(encs[i]['ids']))
        samples.append((source + '_longest', [idx]*args.rows))
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    samples.append(('fixed_mixed_batch', order[:args.rows]))
    measurements = []
    for name, ids in samples:
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        atomic(out / 'progress.json', {'phase': 'backward', 'case': name, 'time': start})
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model.losses([rows[i] for i in ids], [encs[i] for i in ids]).sum()/64
        loss.backward()
        model.clear()
        torch.cuda.synchronize()
        seconds = time.time()-start
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        measurements.append({'case': name, 'rows': len(ids), 'seconds': seconds,
                             'rows_per_second_per_gpu': len(ids)/seconds,
                             'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                             'loss_sum_over_global_batch': loss.item()})
        print(json.dumps(measurements[-1]), flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    ids = [samples[0][1][0], samples[1][1][0]]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model.scores([rows[i] for i in ids], [encs[i] for i in ids])
        model.clear()
        predictions = model.predict([rows[i] for i in ids], [encs[i] for i in ids])
    results = [prediction_row(rows[i], p) for i, p in zip(ids, predictions)]
    assert metrics(results)['task/score']['rows'] == 2
    parity = None
    if args.variant == 'd2_batched':
        # Same real model tensors; compare deterministic evaluation against the
        # stopped chunk-8 pathway without loading a second 9B backbone.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            reference = ChunkedD2.scores(model, [rows[i] for i in ids], [encs[i] for i in ids])
        diffs = [(a.float().softmax(0)-b.float().softmax(0)).abs().max().item() for a, b in zip(logits, reference)]
        assert max(diffs) < .005, diffs
        parity = {'evaluation_probability_max_abs_difference': diffs,
                  'scope': 'fresh real-model BF16; dropout-disabled, not trained-checkpoint bitwise parity'}
    atomic(out / 'complete.json', {'status': 'passed', 'variant': args.variant,
        'code_sha256': code_hashes,
        'checkpoint_policy': args.policy, 'rows_per_microbatch': args.rows,
        'optimizer_updates': 0, 'data': str(data), 'cache_audit_sha256': sha(data / 'cache/audit.json'),
        'measurements': measurements, 'parity': parity, 'seconds': time.time()-t0,
        'trainable_parameters': sum(p.numel() for p in params),
        'evaluation_all_levels': [len(support(rows[i])) for i in ids]})


if __name__ == '__main__':
    main()
