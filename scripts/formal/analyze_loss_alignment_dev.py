"""CPU-only loss/decision diagnostics from frozen complete development predictions."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from score_decoding import score_decoding_diagnostic
from typed_losses import ordinal_log_loss

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    'A4': 'formal_v6_ablations_20261103_v1/a4_ce_only',
    'L1_pair': 'formal_choice_losses_20261103_v1/l1_choice_pair',
    'L2_focal': 'formal_choice_losses_20261103_v1/l2_choice_focal',
    'L3_brier': 'formal_typed_losses_20261103_v1/l3_noul_brier',
    'L4_ordinal': 'formal_typed_losses_20261103_v1/l4_score_ordinal',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return dict(zip(['min', 'p10', 'median', 'p90', 'max'],
                    np.quantile(values, [0, .1, .5, .9, 1]).tolist()))


def main():
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'numpy': np.__version__, 'torch': torch.__version__,
              'script_sha256': digest(Path(__file__)),
              'scope': 'Saved full-native dev predictions only; no model or test changes',
              'gradient_scope': 'Output-logit derivatives, not shared-parameter gradients',
              'models': {}}
    reference_ids = None
    for name, relative in ARMS.items():
        folder = ROOT / 'runs' / relative / 'evaluations'
        files = sorted(folder.glob('epoch_1.dev.rank*.jsonl'))
        rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
        ids = {r['id'] for r in rows}
        assert len(rows) == len(ids) == 6000
        if reference_ids is None:
            reference_ids = ids
        assert ids == reference_ids
        metric_path = folder / 'epoch_1.dev.json'
        metrics = json.loads(metric_path.read_text())
        assert metrics['native_full_candidates'] and metrics['rows'] == 6000
        decoding = score_decoding_diagnostic(rows)
        assert abs(decoding['mae']['argmax'] - metrics['task/score']['mae']) < 1e-12
        cosine, ratios, combined = [], [], []
        checked = 0
        for row in rows:
            if row['task'] != 'score':
                continue
            p = np.asarray(row['probabilities'], dtype=np.float64)
            assert np.all(p > 0), 'This diagnostic requires finite saved log probabilities'
            p /= p.sum()
            y = row['label_index']
            values = np.asarray(row['support'], dtype=np.float64)
            ce = p.copy()
            ce[y] -= 1
            weights = np.diff(values) / (values[-1] - values[0])
            ordinal = np.zeros_like(p)
            for k, weight in enumerate(weights):
                event = np.arange(len(p)) <= k if y <= k else np.arange(len(p)) > k
                derivative = p.copy()
                derivative[event] -= p[event] / p[event].sum()
                ordinal += weight * derivative
            if checked < 16:
                z = torch.tensor(np.log(p), requires_grad=True)
                derivative, = torch.autograd.grad(ordinal_log_loss(z, y, values.tolist()), z)
                np.testing.assert_allclose(derivative.numpy(), ordinal, atol=1e-12, rtol=1e-10)
                checked += 1
            nc, no = np.linalg.norm(ce), np.linalg.norm(ordinal)
            if nc == 0 or no == 0:
                raise ValueError('Zero gradient requires explicit diagnostic handling')
            cosine.append(float(ce @ ordinal / (nc * no)))
            ratios.append(float(no / nc))
            mixed = ce + .1 * ordinal
            combined.append(float(ce @ mixed / (nc * np.linalg.norm(mixed))))
        report['models'][name] = {
            'inputs_sha256': {str(p): digest(p) for p in [*files, metric_path]},
            'metrics': {task: metrics['task/' + task] for task in ['choice', 'noul', 'score']},
            'score_decoding': decoding,
            'score_output_gradient': {
                'rows': len(cosine), 'autograd_checked_rows': checked,
                'negative_ce_ordinal_cosine_rows': sum(v < 0 for v in cosine),
                'ce_ordinal_cosine': quantiles(cosine),
                'ordinal_over_ce_norm': quantiles(ratios),
                'ce_vs_ce_plus_point1_ordinal_cosine': quantiles(combined),
            },
        }
    out = ROOT / 'runs/loss_postmortem_20261103_v1/diagnostics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(out)
    for name, result in report['models'].items():
        print(name, json.dumps({'score_decoding': result['score_decoding']['mae'],
                               'score_output_gradient': result['score_output_gradient']}))


if __name__ == '__main__':
    main()
