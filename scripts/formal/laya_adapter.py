"""Strict, complete-text data adapter for the pinned upstream Laya release.

Upstream architecture and RLCD are imported unchanged. This adapter removes the
SDK's text budgets, preserves our already frozen candidate views, and maps all
permuted outputs back to stable native semantic identities.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from common import support, target
from score_distribution import decisions


def question(row):
    task, q = row['task'], row['question']
    if task == 'choice':
        values = q.get('option_definitions', q['options'])
        criteria = dict(zip(row['json_keys'], values))
    elif task == 'noul':
        criteria = {'false': 'The statement is false', 'true': 'The statement is true'}
    else:
        criteria = [str(v) for v in support(row)]
    return {'t': task, 'ins': q['instructions'], 'crit': criteria}


def make_parts(row, tok, state_ids=None, token_cache=None):
    from laya.common import render_options, QTYPES
    q = question(row)
    def tokens(text):
        text = text.replace(tok.mask_token, ' ')
        if token_cache is not None and text in token_cache:
            return token_cache[text]
        ids = tok(text, add_special_tokens=False, truncation=False)['input_ids']
        if token_cache is not None:
            token_cache[text] = ids
        return ids
    prefix = [tok.cls_token_id] + tokens(f"{q['t']} question: {q['ins']}") + [tok.sep_token_id]
    options = [[tok.mask_token_id] + tokens(' ' + text) for text in render_options(q)]
    if state_ids is None:
        state_ids = tok(row['state']['text'].replace(tok.mask_token, ' '),
                        add_special_tokens=False, truncation=False)['input_ids']
    y = target(row)
    assert len(options) == len(support(row)) and 0 <= y < len(options)
    return {'prefix': prefix, 'options': options, 'suffix': [tok.sep_token_id] + state_ids + [tok.sep_token_id],
            'qtype': QTYPES[row['task']], 'label_index': y,
            'length': len(prefix) + sum(map(len, options)) + len(state_ids) + 2}


def encode_parts(parts, layout='sequential', order=None):
    from laya.common import parallel_layout
    k = len(parts['options'])
    order = list(range(k)) if order is None else order
    assert sorted(order) == list(range(k))
    ids, markers = list(parts['prefix']), []
    for i in order:
        markers.append(len(ids))
        ids.extend(parts['options'][i])
    head_len = len(ids) + 1
    ids.extend(parts['suffix'])
    assert len(ids) == parts['length']
    item = {'ids': ids, 'markers': markers, 'qtype': parts['qtype'],
            'target': [float(i == parts['label_index']) for i in order]}
    if layout == 'parallel':
        item['layout'] = parallel_layout(markers, head_len, len(ids))
    elif layout != 'sequential':
        raise ValueError(layout)
    return item


def order_for(name, k, row_id):
    order = list(range(k))
    if name == 'original':
        return order
    if name == 'reverse':
        return order[::-1]
    if name == 'rotate_left_1':
        return order[1:] + order[:1]
    if name == 'rotate_right_1':
        return order[-1:] + order[:-1]
    if name.startswith('shuffle_'):
        seed = int(name.split('_')[1])
        digest = int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16)
        random.Random(seed ^ digest).shuffle(order)
        return order
    raise ValueError(name)


def canonical_probabilities(logits, order):
    probs = logits[:len(order)].float().softmax(-1).double().cpu().numpy()
    canonical = np.empty_like(probs)
    canonical[np.asarray(order)] = probs
    return canonical.tolist()


def predict_record(row, probabilities):
    ss, y = support(row), target(row)
    if len(probabilities) != len(ss) or abs(math.fsum(probabilities) - 1) > 1e-5:
        raise ValueError('Incomplete probability distribution')
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError('Invalid probabilities')
    if row['task'] == 'score':
        pred = decisions(ss, probabilities)['indices']['median']
    else:
        # Exact ties are resolved by the stable semantic ID, never display slot.
        pred = min(range(len(ss)), key=lambda i: (-probabilities[i], json.dumps(ss[i], ensure_ascii=False)))
    record = {'id': row['id'], 'source': row['source_family'], 'task': row['task'],
              'group_id': row['group_id'], 'label': row['label'], 'label_index': y,
              'prediction_index': pred, 'probabilities': probabilities, 'support': ss,
              'k': len(ss), 'valid': True, 'correct': pred == y,
              'nll': -math.log(max(probabilities[y], 1e-30)),
              'brier': sum((p-float(i == y))**2 for i,p in enumerate(probabilities))}
    if row['task'] == 'score':
        cdf = np.cumsum(probabilities)
        record.update(score_decision_rule='median', mae=abs(ss[pred]-row['label']),
                      expected_mae=abs(sum(v*p for v,p in zip(ss, probabilities))-row['label']),
                      rps=float(np.mean((cdf[:-1]-(np.arange(len(ss)-1) >= y))**2)))
    return record


def metrics(records):
    groups = {}
    for r in records:
        for key in ('all', 'task/'+r['task'], 'source/'+r['source']):
            groups.setdefault(key, []).append(r)
    out = {}
    for key, rows in groups.items():
        d = {'rows': len(rows), 'accuracy': sum(r['correct'] for r in rows)/len(rows),
             'validity': sum(r['valid'] for r in rows)/len(rows)}
        for name in ('nll', 'brier', 'mae', 'expected_mae', 'rps'):
            values = [r[name] for r in rows if name in r]
            if values:
                d[name], d[name+'_rows'] = sum(values)/len(values), len(values)
        out[key] = d
    return out


def microbatches(indices, parts, max_rows, token_budget):
    indices = sorted(indices, key=lambda i: parts[i]['length'], reverse=True)
    batch, longest = [], 0
    for i in indices:
        n = parts[i]['length']
        if batch and (len(batch) >= max_rows or (len(batch)+1)*max(longest,n) > token_budget):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch

