"""Zero-update native-ASAP memory and backward gate for the frozen numeric head."""
import argparse
import json
import pickle
import random
import time

import torch

from common import BASE, PROJECT, SEED, atomic, sha
import asap_score as panel
from models_score_distribution import A4_FLAGS
from models_score_research import NumericD2
from worker_score_specialist import batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=4)
    args = ap.parse_args()
    out = PROJECT / 'runs/score_execution_preflight_20261105_v1/numeric_asap_budget8192_half'
    out.mkdir(exist_ok=False)
    files = ['scripts/formal/benchmark_numeric_asap.py', 'scripts/formal/models_score_research.py',
             'scripts/formal/numeric_score_distribution.py', 'scripts/formal/independent_candidates.py']
    code_hashes = {p: sha(PROJECT / p) for p in files}
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = PROJECT / 'data/score_specialist_asap_fold0_20261104_v1'
    audit = json.loads((data / 'cache/audit.json').read_text())
    desc = audit['splits']['train']
    rows = [json.loads(line) for line in open(desc['views_path'])]
    with open(desc['arms']['kev']['path'], 'rb') as f:
        encs = pickle.load(f)
    model = NumericD2('tcdl', BASE, ablation=A4_FLAGS,
                      distribution_objective={'kind': 'soft_cdf', 'ordinal_scale': 3.0383850790577127})
    model.set_checkpoint_policy('half')
    longest = max(range(len(rows)), key=lambda i: len(encs[i]['ids']))
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    measurements = []
    cases = [('longest_native_asap', [longest]*args.rows),
             ('fixed_mixed_batch', order[:args.rows])]
    cases = [(f'{name}_{j}', chunk) for name, indices in cases
             for j, chunk in enumerate(batches(indices, encs, 'tcdl', 8192, args.rows))]
    for name, indices in cases:
        model.train(); model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model.losses([rows[i] for i in indices], [encs[i] for i in indices]).sum()/64
        loss.backward(); model.clear(); torch.cuda.synchronize()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        measurements.append({'case': name, 'rows': len(indices), 'seconds': time.time()-start,
                             'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
    model.zero_grad(set_to_none=True); model.eval()
    indices = [next(i for i, r in enumerate(rows) if r['essay_set'] == p) for p in panel.RANGES]
    predictions = []
    for start in range(0, len(indices), args.rows):
        chunk = indices[start:start + args.rows]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictions.extend(model.predict([rows[i] for i in chunk], [encs[i] for i in chunk]))
    results = [panel.prediction_row(rows[i], p) for i, p in zip(indices, predictions)]
    assert panel.metrics(results)['task/score']['rows'] == 8
    assert panel.score_rule_report(results, 'preflight')['rows'] == 8
    atomic(out / 'complete.json', {'status': 'passed', 'variant': 'numeric_cdf',
        'code_sha256': code_hashes, 'checkpoint_policy': 'half', 'rows_per_microbatch': args.rows,
        'token_budget': 8192,
        'optimizer_updates': 0, 'data': str(data), 'cache_audit_sha256': sha(data / 'cache/audit.json'),
        'measurements': measurements, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'evaluation_all_levels': sorted({len(r['support']) for r in results})})
    print(json.dumps(measurements), flush=True)


if __name__ == '__main__':
    main()
