"""Matched Score baseline interfaces; raw labels and full allowed ranges."""
import collections
import json
import math

from score_specialist import support, decisions, prediction_row as distribution_row
from score_specialist import metrics as distribution_metrics

PROTOCOL = 'score_baselines_20261105_v1'


def json_prompt(row):
    values = support(row)
    text = f'State: {row["state"]["text"]}\nQuestion: {row["question"]["instructions"]}\n'
    if row['label_kind'] == 'native_integer':
        text += 'Allowed scores: ' + json.dumps(values) + '. '
        text += 'Return only one JSON object with the field "score" containing one allowed integer.'
    else:
        text += f'Allowed score range: [{values[0]:g}, {values[-1]:g}]. Fractional scores are valid. '
        text += 'Return only one JSON object with the field "score" containing a finite JSON number.'
    return text + '\nAnswer: '


def json_answer(row):
    return json.dumps({'score': row['label']}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def encode_baseline(row, arm, tokenizer):
    if arm == 'lora':
        from deciform.typed_tasks import render_scalar_prompt
        ids = tokenizer(render_scalar_prompt(row), add_special_tokens=True, truncation=False).input_ids
        return {'ids_list': [ids]}
    if arm != 'json_sft':
        raise ValueError(arm)
    return {'prompt': tokenizer(json_prompt(row), add_special_tokens=True, truncation=False).input_ids,
            'answer': tokenizer(json_answer(row), add_special_tokens=False, truncation=False).input_ids + [tokenizer.eos_token_id]}


def parse_score(text, row):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number')
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
        if not isinstance(obj, dict) or set(obj) != {'score'}:
            raise ValueError('Expected exactly one score field')
        value = obj['score']
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Score must be a finite number')
        values = support(row)
        if not values[0] <= value <= values[-1]:
            raise ValueError('Score outside declared range')
        if row['label_kind'] == 'native_integer' and (type(value) is not int or value not in values):
            raise ValueError('Native ASAP score must be an allowed integer')
        return value, None
    except (ValueError, TypeError, OverflowError) as exc:
        return None, str(exc)


def prediction_row(row, prediction):
    if 'probabilities' in prediction:
        return distribution_row(row, prediction)
    values = support(row)
    width = values[-1] - values[0]
    valid = prediction['valid']
    y = float(row['label'])
    value = prediction.get('prediction_value')
    if valid and (value is None or not values[0] <= value <= values[-1]):
        raise ValueError('Invalid numeric prediction marked valid')
    return {**prediction, 'id': row['id'], 'source': row['source_family'],
            'group_id': row['group_id'], 'task': 'score', 'label': y,
            'support': values, 'k': len(values), 'score_decision_rule': 'generated_json_number',
            'mae': abs(value-y) if valid else width,
            'invalid_penalty': None if valid else 'full allowed score range',
            'quantization_floor': 0. if row['label_kind'] == 'continuous' else min(abs(v-y) for v in values),
            'original_label_preserved': True}


def metrics(rows):
    if all('probabilities' in row for row in rows):
        result = distribution_metrics(rows)
        for key, value in result.items():
            if isinstance(value, dict):
                value.update(validity=1., invalid_rows=0)
        return result
    if any('probabilities' in row for row in rows):
        raise ValueError('Do not mix text answers and probability predictions')
    groups = collections.defaultdict(list)
    for row in rows:
        for key in ('all', 'task/score', 'source/'+row['source']):
            groups[key].append(row)
    result = {}
    for key, items in groups.items():
        valid = [r for r in items if r['valid']]
        result[key] = {'rows': len(items), 'valid_rows': len(valid),
            'invalid_rows': len(items)-len(valid), 'validity': len(valid)/len(items),
            'mae': sum(r['mae'] for r in items)/len(items),
            'nmae': sum(r['mae']/(r['support'][-1]-r['support'][0]) for r in items)/len(items),
            'mae_on_valid_only': sum(r['mae'] for r in valid)/len(valid) if valid else None,
            'invalid_policy': 'full-range error included in headline denominator'}
    result['macro_source_nmae'] = sum(v['nmae'] for k,v in result.items() if k.startswith('source/'))/sum(k.startswith('source/') for k in result)
    return result


def score_rule_report(rows, split):
    report = metrics(rows)
    probabilistic = all('probabilities' in row for row in rows)
    return {'evaluation_protocol': PROTOCOL, 'split': split, 'rows': len(rows),
        'primary_rule': 'median' if probabilistic else 'generated_json_number',
        'primary_metric': 'native MAE against unchanged original labels',
        'primary_mae': report['task/score']['mae'],
        'macro_source_nmae': report['macro_source_nmae'],
        'by_source': {k[7:]:v for k,v in report.items() if k.startswith('source/')},
        'rules': report['task/score'].get('rules'), 'probability_metrics_applicable': probabilistic,
        'no_candidate_sampling': True, 'no_target_rounding': True}
