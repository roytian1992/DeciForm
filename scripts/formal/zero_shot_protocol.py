"""Frozen, unadapted LLM JSON references; no fitting or answer repair."""
import copy
import hashlib
import json
import random
from collections import defaultdict

from common import json_prompt as main_prompt, parse_answer, support, target
from classification_order_audit import view_orders as legacy_orders, summarize_records
from score_baselines import json_prompt as score_prompt, parse_score
from score_baselines import prediction_row as score_record, metrics as score_metrics

NEW_ORDERS = ['original', 'reversed', 'rotate_left_1', 'rotate_right_1',
              'shuffle_11', 'shuffle_29', 'shuffle_47']
LEGACY_ORDERS = ['original', 'reversed', 'legacy_shuffle_1', 'legacy_shuffle_2', 'legacy_shuffle_3']
CHOICE_ORDERS = NEW_ORDERS + LEGACY_ORDERS[2:]
PANELS = ['main', 'asap', 'continuous', 'choice_order', 'noul_order']


def order_for(row, name):
    k = len(support(row))
    identity = list(range(k))
    if name == 'original': return identity
    if name == 'reversed': return identity[::-1]
    if name == 'rotate_left_1': return identity[1:] + identity[:1]
    if name == 'rotate_right_1': return identity[-1:] + identity[:-1]
    if name.startswith('legacy_'):
        return dict(legacy_orders('json_sft', 'native', row))[name[7:]]
    if name.startswith('shuffle_'):
        seed = int(name.split('_')[1])
        digest = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
        random.Random(seed ^ digest).shuffle(identity)
        return identity
    raise ValueError(name)


def views_for(panel):
    if panel == 'choice_order': return CHOICE_ORDERS
    if panel == 'noul_order': return ['original', 'reversed']
    if panel in PANELS: return ['original']
    raise ValueError(panel)


def prompt_body(row, panel, view='original'):
    if panel in ('asap', 'continuous'):
        assert row['task'] == 'score' and view == 'original'
        return score_prompt(row)
    changed = copy.deepcopy(row)
    if panel == 'choice_order':
        assert row['task'] == 'choice'
        order = order_for(row, view)
        for key in ('options', 'option_ids', 'option_definitions'):
            if key in row['question']:
                changed['question'][key] = [row['question'][key][i] for i in order]
        changed['json_keys'] = [row['json_keys'][i] for i in order]
    elif panel == 'noul_order':
        assert row['task'] == 'noul'
        values = [support(row)[i] for i in order_for(row, view)]
        changed['question']['instructions'] += '\nAllowed answers: ' + json.dumps(values) + '.'
    else:
        assert panel == 'main' and view == 'original'
    return main_prompt(changed)


def prompt_ids(row, panel, view, tokenizer, mode):
    text = prompt_body(row, panel, view)
    if mode == 'matched_base':
        return tokenizer(text, add_special_tokens=True, truncation=False).input_ids
    if mode == 'official_chat':
        # Transformers 5 defaults to a BatchEncoding mapping. Our batching
        # contract requires a flat token list, never the mapping's two keys.
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise TypeError('Chat template must return a nonempty flat token-ID list')
        return ids
    raise ValueError(mode)


def strip_terminal(text, tokens):
    tokens = sorted({t for t in tokens if t}, key=len, reverse=True)
    while True:
        token = next((t for t in tokens if text.endswith(t)), None)
        if token is None: return text
        text = text[:-len(token)]


def record_prediction(row, panel, view, raw, clean):
    common = {'id': row['id'], 'source': row['source_family'], 'group_id': row['group_id'],
              'task': row['task'], 'panel': panel, 'view': view, 'raw_output': raw}
    if panel in ('asap', 'continuous'):
        value, error = parse_score(clean, row)
        return {**score_record(row, {'valid': error is None, 'prediction_value': value,
                                    'parse_error': error, 'raw_output': raw}), **common}
    index, error = parse_answer(clean, row)  # Parse against ORIGINAL semantic IDs.
    values = support(row)
    valid = error is None
    result = {**common, 'valid': valid, 'prediction_index': index, 'parse_error': error,
              'label': row['label'], 'label_index': target(row), 'support': values,
              'correct': bool(valid and index == target(row)), 'k': len(values)}
    if row['task'] == 'score':
        result['mae'] = abs(values[index] - row['label']) if valid else values[-1] - values[0]
    if panel.endswith('_order'):
        result.update(order=order_for(row, view), order_test_applicable=True)
    return result


def classification_summary(rows):
    report = summarize_records(rows, with_interval=False)
    paired = defaultdict(dict)
    for row in rows: paired[row['id']][row['view']] = row
    valid = [(v['original'], v['reversed']) for v in paired.values()
             if v['original']['valid'] and v['reversed']['valid']]
    report['order_metrics']['both_valid_reverse_pairs'] = len(valid)
    report['order_metrics']['reverse_flip_rate_on_valid_pairs'] = (
        sum(a['prediction_index'] != b['prediction_index'] for a, b in valid) / len(valid)
        if valid else None)
    report['invalid_caveat'] = 'Two invalid answers count as unchanged, not as a successful invariant decision; inspect validity and both-valid flip rate.'
    return report


def summarize(panel, rows):
    if not rows: raise ValueError('Empty panel')
    if panel in ('asap', 'continuous'): return score_metrics(rows)
    if panel == 'choice_order':
        return {'new_d2_seven_orders': classification_summary([r for r in rows if r['view'] in NEW_ORDERS]),
                'legacy_baseline_five_orders': classification_summary([r for r in rows if r['view'] in LEGACY_ORDERS])}
    if panel == 'noul_order': return classification_summary(rows)
    groups = defaultdict(list)
    for row in rows:
        for key in ('all', 'task/' + row['task'], 'source/' + row['source']): groups[key].append(row)
    report = {}
    for key, group in groups.items():
        n = len(group)
        report[key] = {'rows': n, 'accuracy': sum(r['correct'] for r in group) / n,
                       'validity': sum(r['valid'] for r in group) / n,
                       'invalid_rows': sum(not r['valid'] for r in group)}
        scores = [r for r in group if r['task'] == 'score']
        if scores:
            report[key].update(mae=sum(r['mae'] for r in scores) / len(scores), mae_rows=len(scores))
    return report
