"""Semantic-preserving answer-order interventions and paired audit statistics."""
import copy
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from common import encode, json_answer, json_prompt, support
from score_distribution import decisions


PANELS = ('native', 'instruction_list')


def orders(k):
    """A balanced dihedral design: 10 orders for K=5, both orders for K=2."""
    original = list(range(k))
    reverse = original[::-1]
    proposals = [('original', original), ('reversed', reverse)]
    for prefix, sequence in [('rotate', original), ('reverse_rotate', reverse)]:
        proposals.extend((f'{prefix}_{i}', sequence[i:] + sequence[:i]) for i in range(1, k))
    seen, result = set(), []
    for name, order in proposals:
        if tuple(order) not in seen:
            result.append((name, order))
            seen.add(tuple(order))
    return result


def applicable(arm, panel, task):
    return panel == 'instruction_list' or arm in ('kev', 'tcdl') or (arm == 'json_sft' and task == 'score')


def view_orders(arm, panel, row):
    result = orders(len(support(row)))
    return result if applicable(arm, panel, row['task']) else result[:1]


def permute_pointer(enc, order):
    """Move token spans; gather readouts by semantic ID for BOTH Kev and D2.

    Only readout indices are canonicalized. The backbone sees the actual
    permuted tokens at their new causal positions. State/instruction stay fixed.
    """
    if enc['option_isolation'] or len(enc['opt_idx']) != 1:
        raise ValueError('Expected one ordinary causal question')
    k = len(enc['opt_idx'][0])
    if sorted(order) != list(range(k)):
        raise ValueError('A complete permutation is required')
    result = copy.deepcopy(enc)
    spans = []
    for i in range(k):
        positions = [j for j, value in enumerate(enc['opt']) if value == i]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError('Non-contiguous native option span')
        if positions[-1] != enc['opt_idx'][0][i]:
            raise ValueError('Semantic readout does not match option end')
        spans.append((positions[0], positions[-1] + 1))
    start, end = spans[0][0], spans[-1][1]
    if any(spans[i][1] != spans[i + 1][0] for i in range(k - 1)):
        raise ValueError('Unexpected text between candidate spans')
    ids, semantic_ids, ends = [], [], [None] * k
    for semantic in order:
        lo, hi = spans[semantic]
        ids.extend(enc['ids'][lo:hi])
        semantic_ids.extend([semantic] * (hi - lo))
        ends[semantic] = start + len(ids) - 1
    result['ids'][start:end] = ids
    result['opt'][start:end] = semantic_ids
    result['opt_idx'] = [ends]
    assert len(result['ids']) == len(enc['ids'])
    return result


def encode_view(row, tokenizer, arm, panel, order, original=None):
    native = support(row)
    if sorted(order) != list(range(len(native))):
        raise ValueError('Never remove, duplicate or relabel an answer')
    if panel == 'instruction_list':
        changed = copy.deepcopy(row)
        changed['question']['instructions'] += '\nAllowed answers: ' + json.dumps([native[i] for i in order]) + '.'
        # Native output interfaces stay fixed here: this panel isolates a
        # redundant, explicit list in the same instruction field for all arms.
        return encode(changed, arm, tokenizer)
    if panel != 'native':
        raise ValueError('Unknown audit panel')
    original = original if original is not None else encode(row, arm, tokenizer)
    if arm in ('kev', 'tcdl'):
        return permute_pointer(original, order)
    if arm == 'json_sft' and row['task'] == 'score':
        tail = '. Return only one JSON object with the field "score" containing one allowed integer.\nAnswer: '
        ending = 'Allowed scores: ' + json.dumps(native) + tail
        prompt = json_prompt(row)
        assert prompt.endswith(ending)
        prompt = prompt[:-len(ending)] + 'Allowed scores: ' + json.dumps([native[i] for i in order]) + tail
        return {'prompt': tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids,
                'answer': tokenizer(json_answer(row), add_special_tokens=False, truncation=False).input_ids + [tokenizer.eos_token_id]}
    return copy.deepcopy(original)


def input_hash(enc):
    actual = enc.get('ids', enc.get('prompt', enc.get('ids_list')))
    return hashlib.sha256(json.dumps(actual, separators=(',', ':')).encode()).hexdigest()


def normalize_prediction(row, prediction):
    prediction = dict(prediction)
    if 'probabilities' in prediction and row['task'] == 'score':
        out = decisions(support(row), prediction['probabilities'])
        prediction.update(prediction_index=out['indices']['median'],
                          score_decision_rule='median',
                          argmax_prediction_index=out['indices']['argmax'],
                          score_mean=out['values']['mean'],
                          mean_nearest_prediction_index=out['indices']['mean_nearest'])
    return prediction


def point_error(record):
    if not record['valid']:
        return float(record['support'][-1] - record['support'][0]) if record['task'] == 'score' else 1.
    pred = record['prediction_index']
    if type(pred) is not int or not 0 <= pred < len(record['support']):
        raise ValueError('Invalid semantic prediction index')
    if record['task'] == 'score':
        return abs(record['support'][pred] - record['label'])
    return float(pred != record['label_index'])


def changed(a, b):
    return a['valid'] != b['valid'] or (a['valid'] and a['prediction_index'] != b['prediction_index'])


def cluster_interval(values, groups, seed=20261104, repeats=2000):
    clusters = defaultdict(list)
    for value, group in zip(values, groups):
        clusters[group].append(value)
    sums = np.array([sum(v) for v in clusters.values()], dtype=float)
    counts = np.array([len(v) for v in clusters.values()], dtype=float)
    if not len(sums):
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        indices = rng.integers(0, len(sums), len(sums))
        estimates.append(float(sums[indices].sum() / counts[indices].sum()))
    return {'low': float(np.quantile(estimates, .025)), 'high': float(np.quantile(estimates, .975)),
            'clusters': len(sums), 'repeats': repeats, 'scope': 'paired within this fixed model; not training-seed uncertainty'}


def summarize_records(records, with_interval=True):
    by_id = defaultdict(dict)
    for item in records:
        if item['view'] in by_id[item['id']]:
            raise ValueError('Duplicate prediction')
        by_id[item['id']][item['view']] = item
    if not by_id:
        raise ValueError('Empty audit population')
    views = list(next(iter(by_id.values())))
    if 'original' not in views or any(set(v) != set(views) for v in by_id.values()):
        raise ValueError('Incomplete paired candidate-order views')
    task = records[0]['task']
    can_test = records[0]['order_test_applicable']
    if any(r['task'] != task or r['order_test_applicable'] != can_test for r in records):
        raise ValueError('Mixed task or applicability in one report')
    n = len(by_id)
    stats = {}
    for view in views:
        items = [r[view] for r in by_id.values()]
        mean_error = sum(point_error(r) for r in items) / n
        stats[view] = {'rows': n, 'validity': sum(r['valid'] for r in items) / n,
                       'accuracy': sum(r['valid'] and r['prediction_index'] == r['label_index'] for r in items) / n,
                       'mae': mean_error if task == 'score' else None}
    result = {'rows': n, 'evaluated_predictions': len(records), 'task': task,
              'applicable': can_test, 'per_view': stats,
              'invalid_score_penalty': 'native score span; failures remain in denominator',
              'native_candidates_complete': True}
    if not can_test:
        result['order_metrics'] = None
        result['reason'] = 'No candidate-list input in this native task interface; no perturbation was performed'
        return result
    reverse_delta, reverse_flip, groups, any_flip, worst, means, tvs, jsds, shifts = [], [], [], [], [], [], [], [], []
    for entries in by_id.values():
        base, reverse = entries['original'], entries['reversed']
        others = [r for view, r in entries.items() if view != 'original']
        reverse_delta.append(point_error(reverse) - point_error(base))
        reverse_flip.append(float(changed(base, reverse)))
        groups.append(str(base['source']) + ':' + str(base['group_id']))
        any_flip.append(any(changed(base, r) for r in others))
        errors = [point_error(r) for r in entries.values()]
        worst.append(max(errors)); means.append(sum(errors) / len(errors))
        for r in others:
            if 'probabilities' in base and 'probabilities' in r:
                p, q = base['probabilities'], r['probabilities']
                tvs.append(.5 * sum(abs(a-b) for a,b in zip(p,q)))
                m = [(a+b)/2 for a,b in zip(p,q)]
                jsds.append(.5 * sum(a*math.log(a/c) if a else 0 for a,c in zip(p,m)) +
                            .5 * sum(b*math.log(b/c) if b else 0 for b,c in zip(q,m)))
            if task == 'score' and base['valid'] and r['valid']:
                shifts.append(abs(base['support'][base['prediction_index']] - r['support'][r['prediction_index']]))
    result['order_metrics'] = {
        'reverse_flip_rate': sum(reverse_flip)/n, 'any_order_flip_rate': sum(any_flip)/n,
        'reverse_error_delta_positive_is_worse': sum(reverse_delta)/n,
        'average_order_error': sum(means)/n, 'worst_order_error': sum(worst)/n,
        'mean_total_variation': sum(tvs)/len(tvs) if tvs else None,
        'mean_jensen_shannon': sum(jsds)/len(jsds) if jsds else None,
        'mean_absolute_score_change_on_valid_pairs': sum(shifts)/len(shifts) if shifts else None,
        'valid_score_pairs': len(shifts),
        'any_invalid_rate': sum(any(not r['valid'] for r in entries.values()) for entries in by_id.values())/n,
    }
    if with_interval:
        result['order_metrics']['reverse_error_delta_95ci'] = cluster_interval(reverse_delta, groups)
        result['order_metrics']['reverse_flip_rate_95ci'] = cluster_interval(reverse_flip, groups)
    return result
