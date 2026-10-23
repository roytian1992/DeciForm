"""Accuracy, calibration and selective-risk metrics."""
from collections import Counter
import math

def summarize(predictions):
    if not predictions:
        raise ValueError('empty evaluation panel')
    n = len(predictions)
    labels = sorted({k for p in predictions for k in p['probabilities']})
    correct = [p['prediction'] == p['label'] for p in predictions]
    for p in predictions:
        probs = p['probabilities']
        if (p['label'] not in probs or any(not math.isfinite(v) or v < 0 or v > 1 for v in probs.values())
                or abs(sum(probs.values()) - 1) > 1e-5):
            raise ValueError('invalid probabilities or label')
    brier = sum(sum((v - float(k == p['label'])) ** 2 for k, v in p['probabilities'].items())
                for p in predictions) / n
    nll = -sum(math.log(max(p['probabilities'][p['label']], 1e-12)) for p in predictions) / n
    bins = []
    for i in range(10):
        indices = [j for j, p in enumerate(predictions) if min(int(p['confidence'] * 10), 9) == i]
        bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'count': len(indices),
                     'accuracy': sum(correct[j] for j in indices) / len(indices) if indices else None,
                     'confidence': sum(predictions[j]['confidence'] for j in indices) / len(indices) if indices else None})
    ece = sum(b['count'] * abs(b['accuracy'] - b['confidence']) for b in bins if b['count']) / n
    f1 = []
    for label in labels:
        tp = sum(p['label'] == label and p['prediction'] == label for p in predictions)
        denom = sum(p['label'] == label for p in predictions) + sum(p['prediction'] == label for p in predictions)
        f1.append(2 * tp / denom if denom else 0.0)
    # Fixed thresholds are descriptive; no threshold is fitted on these rows.
    risks = {}
    for threshold in (0.5, 0.8, 0.9, 0.95):
        kept = [p for p in predictions if p['confidence'] >= threshold]
        risks[str(threshold)] = {'count': len(kept), 'coverage': len(kept) / n,
                                'risk': sum(p['prediction'] != p['label'] for p in kept) / len(kept) if kept else None}
    acc = sum(correct) / n
    z = 1.959963984540054
    center = (acc + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {'record_count': n, 'accuracy': acc, 'macro_f1': sum(f1)/len(f1),
            'brier': brier, 'ece': ece, 'nll': nll, 'label_counts': dict(Counter(p['label'] for p in predictions)),
            'accuracy_wilson_95': [center-radius, center+radius], 'reliability_bins': bins,
            'fixed_threshold_risk': risks, 'fitted_temperature': False,
            'macro_f1_label_policy': 'all offered classes, zero for absent classes'}
