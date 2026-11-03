"""Dev-only paired native-score decoding diagnostic; never changes predictions."""
import math


def score_decoding_diagnostic(rows):
    score_rows = [r for r in rows if r['task'] == 'score']
    if not score_rows or len({r['id'] for r in score_rows}) != len(score_rows):
        raise ValueError('Expected nonempty unique score records')
    errors = {'argmax': 0., 'median': 0.}
    changed = 0
    for row in score_rows:
        values, probs = row['support'], row['probabilities']
        if not row['valid'] or len(values) != len(probs) or len(values) != row['k']:
            raise ValueError('Incomplete native prediction')
        if len(values) < 2 or not all(math.isfinite(v) for v in values) or any(
                a >= b for a, b in zip(values, values[1:])):
            raise ValueError('Native score values must be finite and increasing')
        if any(not math.isfinite(p) or p < 0 for p in probs) or abs(sum(probs)-1) > 1e-5:
            raise ValueError('Invalid probability distribution')
        y = row['label_index']
        if values[y] != row['label']:
            raise ValueError('Native label mismatch')
        mode = max(range(len(probs)), key=probs.__getitem__)
        if mode != row['prediction_index']:
            raise ValueError('Original prediction is not argmax')
        threshold = .5*sum(probs)
        cumulative = 0.
        median = len(probs)-1
        for j, p in enumerate(probs):
            cumulative += p
            if cumulative >= threshold:
                median = j
                break
        errors['argmax'] += abs(values[mode]-values[y])
        errors['median'] += abs(values[median]-values[y])
        changed += mode != median
    return {'split': 'dev', 'rows': len(score_rows), 'native_full_candidates': True,
            'mae': {k: v/len(score_rows) for k, v in errors.items()},
            'median_changed_predictions': changed,
            'selection_use': False, 'original_predictions_modified': False,
            'scope': 'paired CPU decoding diagnostic; original argmax remains headline protocol'}
