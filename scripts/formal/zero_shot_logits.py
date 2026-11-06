"""Exact frozen-LM continuation scoring; no fitted readout or tokenizer shortcuts."""
import copy
import hashlib
import json
import math
from collections import defaultdict

import torch
from common import support as main_support, target
from zero_shot_protocol import prompt_ids, order_for, summarize as json_summarize
from score_specialist import decisions, prediction_row as score_record, metrics as score_metrics

RULES = ('sequence_likelihood', 'constrained_trie')
PANELS = ('main', 'asap', 'continuous', 'choice_order', 'noul_order')


def values_for(row):
    return list(row['score_support']) if 'score_support' in row else main_support(row)


def answer_strings(row):
    values = values_for(row)
    if row['task'] == 'choice':
        return [json.dumps({'choice_id': k}, separators=(',', ':')) for k in row['json_keys']]
    if row['task'] == 'noul':
        return [json.dumps({'answer': v}, separators=(',', ':')) for v in values]
    # One canonical representation per legal numeric value, preserving decimals.
    return ['{"score":' + format(v, '.12g') + '}' for v in values]


def split_common(sequences):
    if len(sequences) < 2 or len({tuple(s) for s in sequences}) != len(sequences):
        raise ValueError('Need distinct complete answers')
    n = 0
    while all(len(s) > n and s[n] == sequences[0][n] for s in sequences): n += 1
    tails = [s[n:] for s in sequences]
    if any(not s for s in tails): raise ValueError('Answers must be prefix-free')
    return sequences[0][:n], tails


def make_trie(sequences):
    root = {'children': {}, 'leaf': None}
    for i, seq in enumerate(sequences):
        node = root
        for token in seq:
            if node['leaf'] is not None: raise ValueError('Non-prefix-free answers')
            node = node['children'].setdefault(token, {'children': {}, 'leaf': None})
        if node['children'] or node['leaf'] is not None: raise ValueError('Duplicate/prefix answer')
        node['leaf'] = i
    return root


def encode(row, panel, view, tokenizer):
    answers = answer_strings(row)
    seqs = [tokenizer(s, add_special_tokens=False, truncation=False).input_ids for s in answers]
    assert all(tokenizer.decode(s, skip_special_tokens=False) == a for s, a in zip(seqs, answers))
    common, tails = split_common(seqs)
    ids = prompt_ids(row, panel, view, tokenizer, 'official_chat') + common
    return {'ids': ids, 'tails': tails, 'common_tokens': len(common), 'answers': answers,
            'input_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()}


def batches(items, budget=100000, max_rows=8, max_nodes=256):
    batch, longest, nodes = [], 0, 0
    for item in sorted(items, key=lambda b: (-b['length'], b['row']['id'])):
        k = len(item['enc']['tails'])
        length = item['length']
        if batch and ((nodes+k)*max(longest, length) > budget or len(batch) >= max_rows or nodes+k > max_nodes):
            yield batch
            batch, longest, nodes = [], 0, 0
        batch.append(item); longest = max(longest, length); nodes += k
    if batch: yield batch


def pad(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq, device=device)
        mask[i, -len(seq):] = 1
    return ids, mask


@torch.inference_mode()
def score_batch(model, encodings, pad_id):
    """Breadth-first exact enumeration with hybrid-cache reorder (including DeltaNet states).

    Each active node is one distinct suffix prefix. No beam pruning; all leaves
    are evaluated. Full vocabulary normalizers are retained for sequence likelihood.
    Shared answer-prefix probabilities cancel across candidates in the same request.
    """
    device = next(model.parameters()).device
    ids, mask = pad([e['ids'] for e in encodings], pad_id, device)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                use_cache=True, logits_to_keep=1)
    cache, logits = out.past_key_values, out.logits[:, -1].float()
    del out
    nodes = [(i, make_trie(e['tails'])) for i, e in enumerate(encodings)]
    raw = [0.] * len(nodes)
    restricted = [0.] * len(nodes)
    result = [{'sequence_likelihood': [None]*len(e['tails']),
               'constrained_trie': [None]*len(e['tails'])} for e in encodings]
    while nodes:
        lp = logits.log_softmax(-1)
        edges = [(j, token) for j, (_, node) in enumerate(nodes) for token in node['children']]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device)
        edge_logp = lp[edge_index[:, 0], edge_index[:, 1]].cpu().tolist()
        branches, parents, tokens, next_raw, next_restricted = [], [], [], [], []
        offset = 0
        for j, (owner, node) in enumerate(nodes):
            allowed = list(node['children'])
            local = edge_logp[offset:offset+len(allowed)]
            offset += len(allowed)
            m = max(local)
            norm = m + math.log(sum(math.exp(v-m) for v in local))
            for token, token_lp in zip(allowed, local):
                child = node['children'][token]
                a, b = raw[j]+token_lp, restricted[j]+token_lp-norm
                if child['leaf'] is not None:
                    k = child['leaf']
                    result[owner]['sequence_likelihood'][k] = a
                    result[owner]['constrained_trie'][k] = b
                else:
                    parents.append(j); tokens.append(token)
                    branches.append((owner, child)); next_raw.append(a); next_restricted.append(b)
        if not branches: break
        index = torch.tensor(parents, dtype=torch.long, device=device)
        # reorder_cache supports Qwen's full attention AND recurrent/conv caches.
        # index_select copies duplicated parents before in-place recurrent updates.
        cache.reorder_cache(index)
        mask = torch.cat((mask.index_select(0, index), torch.ones((len(index), 1), device=device, dtype=mask.dtype)), -1)
        positions = mask.sum(-1, keepdim=True)-1
        out = model(input_ids=torch.tensor(tokens, device=device)[:, None], attention_mask=mask,
                    position_ids=positions, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
        del out, lp
        nodes, raw, restricted = branches, next_raw, next_restricted
    for item in result:
        for rule in RULES:
            assert all(v is not None and math.isfinite(v) for v in item[rule])
        assert abs(sum(math.exp(v) for v in item['constrained_trie'])-1) < 1e-4
    return result


@torch.inference_mode()
def dense_scores(model, enc, indices):
    """Slow independent teacher-forced verification, never used for production scoring."""
    device = next(model.parameters()).device
    scores = []
    for i in indices:
        suffix = enc['tails'][i]
        seq = torch.tensor([enc['ids']+suffix], device=device)
        out = model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False,
                    logits_to_keep=len(suffix)+1)
        z = out.logits[0, :-1].float().log_softmax(-1)
        scores.append(float(z[torch.arange(len(suffix), device=device), torch.tensor(suffix, device=device)].sum()))
    return scores


def prediction(row, panel, view, enc, scores, rule):
    z = torch.tensor(scores[rule], dtype=torch.float64)
    probs = z.softmax(0).tolist()
    values = values_for(row)
    index = int(z.argmax()) if row['task'] != 'score' else decisions(values, probs)['indices']['median']
    result = {'id': row['id'], 'source': row['source_family'], 'group_id': row['group_id'],
              'task': row['task'], 'panel': panel, 'view': view, 'valid': True,
              'prediction_index': index, 'probabilities': probs, 'label': row['label'],
              'support': values, 'k': len(values), 'scoring_rule': rule,
              'log_scores': scores[rule], 'canonical_answers': enc['answers'],
              'input_tokens': len(enc['ids']), 'input_sha256': enc['input_sha256'],
              'common_answer_prefix_tokens': enc['common_tokens'],
              'parse_error': None, 'optimizer_updates': 0}
    if panel in ('asap', 'continuous'):
        return score_record(row, result)
    result.update(label_index=target(row), correct=index == target(row))
    if row['task'] == 'score':
        ds = decisions(values, probs)
        result.update(prediction_value=values[index], score_decisions=ds['values'],
                      mae=abs(values[index]-row['label']), score_decision_rule='median')
    if panel.endswith('_order'):
        result.update(order=order_for(row, view), order_test_applicable=True)
    return result


def summarize(panel, records):
    if panel in ('asap', 'continuous'): return score_metrics(records)
    result = json_summarize(panel, records)
    if panel == 'main':
        rows = [r for r in records if r['task'] == 'score']
        result['task/score']['rules'] = {rule: {'mae': sum(abs(r['score_decisions'][rule]-r['label']) for r in rows)/len(rows)}
                                             for rule in ('median', 'mean', 'argmax', 'mean_nearest')}
    return result
