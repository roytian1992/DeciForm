"""Choice-only completion of the classification answer-order audit.

The previous score/noul audit is immutable provenance. Score is excluded from
this classification report. Existing noul predictions are reused separately.
"""
import copy
import hashlib
import random

from common import encode, support
from order_audit import input_hash, permute_pointer, summarize_records as _summarize

PANELS = ('native',)


def applicable(arm, panel, task):
    if arm not in ('lora', 'kev', 'json_sft', 'tcdl') or panel != 'native' or task != 'choice':
        raise ValueError('This completion evaluates native choice only')
    return True


def view_orders(arm, panel, row):
    applicable(arm, panel, row['task'])
    k = len(support(row))
    if k < 3:
        raise ValueError('Frozen choice panel is expected to have at least 3 options')
    original = list(range(k))
    result = [('original', original), ('reversed', original[::-1])]
    seed = int.from_bytes(hashlib.sha256(('choice-order-20261104:' + row['id']).encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    seen = {tuple(order) for _, order in result}
    while len(result) < 5:
        order = original.copy()
        rng.shuffle(order)
        if tuple(order) not in seen:
            seen.add(tuple(order))
            result.append((f'shuffle_{len(result)-1}', order))
    return result


def encode_view(row, tokenizer, arm, panel, order, original=None):
    applicable(arm, panel, row['task'])
    if sorted(order) != list(range(len(support(row)))):
        raise ValueError('Every native choice must appear exactly once')
    original = original if original is not None else encode(row, arm, tokenizer)
    if arm in ('kev', 'tcdl'):
        return permute_pointer(original, order)
    if arm == 'lora':
        result = copy.deepcopy(original)
        result['ids_list'] = [result['ids_list'][i] for i in order]
        return result
    # Keep candidate IDs attached to their original text while changing display
    # order. Parsing uses the original row, so answers stay in semantic space.
    changed = copy.deepcopy(row)
    for key in ('options', 'option_ids', 'option_definitions'):
        if key in changed['question']:
            changed['question'][key] = [row['question'][key][i] for i in order]
    changed['json_keys'] = [row['json_keys'][i] for i in order]
    return encode(changed, arm, tokenizer)


def normalize_prediction(row, prediction, arm, order):
    applicable(arm, 'native', row['task'])
    prediction = dict(prediction)
    if arm == 'lora':
        if prediction['valid']:
            prediction['prediction_index'] = order[prediction['prediction_index']]
        if 'probabilities' in prediction:
            aligned = [None] * len(order)
            for position, semantic in enumerate(order):
                aligned[semantic] = prediction['probabilities'][position]
            prediction['probabilities'] = aligned
    return prediction


def summarize_records(records, with_interval=True):
    if any(record['task'] not in ('choice', 'noul') for record in records):
        raise ValueError('Flip-rate reporting is restricted to choice and noul')
    result = _summarize(records, with_interval=with_interval)
    result.pop('invalid_score_penalty', None)
    for stats in result['per_view'].values():
        stats.pop('mae', None)
    if result.get('order_metrics'):
        for key in ('mean_absolute_score_change_on_valid_pairs', 'valid_score_pairs'):
            result['order_metrics'].pop(key, None)
    return result
