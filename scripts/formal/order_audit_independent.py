"""Full-native position sensitivity audit for the frozen independent D2 model."""
import argparse
import datetime
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from common import BASE, PROJECT, SEED, TASKS, atomic, sha, target
from independent_candidates import cost, permute_candidates, split_candidates
from models_independent import FormalModel


def batches(indices, encs, budget, max_rows=8):
    indices = sorted(indices, key=lambda i: cost(encs[i], 'tcdl')[1], reverse=True)
    cur, tokens = [], 0
    longest = 0
    for i in indices:
        _, length = cost(encs[i], 'tcdl')
        if cur and ((tokens + length) * max(longest, length) > budget or len(cur) >= max_rows):
            yield cur
            cur, tokens, longest = [], 0, 0
        cur.append(i)
        tokens += length
        longest = max(longest, length)
    if cur:
        yield cur


def order_for(name, enc, row):
    k = len(enc['branches'])
    identity = list(range(k))
    if name == 'original':
        return identity
    if name == 'reverse':
        return list(reversed(identity))
    if name == 'rotate_left_1':
        return identity[1:] + identity[:1] if k > 1 else identity
    if name == 'rotate_right_1':
        return identity[-1:] + identity[:-1] if k > 1 else identity
    if name.startswith('shuffle_'):
        seed = int(name.split('_')[1])
        digest = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
        value = identity[:]
        random.Random(seed ^ digest).shuffle(value)
        return value
    raise ValueError(name)


def load_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(state['trainable']):
        raise ValueError('Frozen D2 checkpoint trainable inventory differs from audit model')
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state['trainable'][name])
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.getenv('RANK', 0))
    world = int(os.getenv('WORLD_SIZE', 1))
    local = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=90))
    try:
        assert cfg['evaluation_candidates'] == 'complete_native'
        assert cfg['tasks'] == ['choice', 'noul']
        assert sha(Path(cfg['checkpoint'])) == cfg['checkpoint_sha256']
        data = Path(cfg['data'])
        audit_path = data / 'cache/audit.json'
        assert sha(audit_path) == cfg['cache_audit_sha256']
        audit = json.loads(audit_path.read_text())
        desc = audit['splits']['test_in_domain']
        assert desc['native_evaluation'] and desc['rows'] == 15000
        assert sha(desc['views_path']) == desc['views_sha256']
        entry = desc['arms']['kev']
        assert sha(entry['path']) == entry['sha256']
        rows = [json.loads(line) for line in open(desc['views_path'])]
        with open(entry['path'], 'rb') as handle:
            encs = pickle.load(handle)
        assert len(rows) == len(encs) == 15000
        encs = [split_candidates(e) for e in encs]
        native = [i for i, row in enumerate(rows) if row['task'] in cfg['tasks']]
        assert len(native) > 0 and all(rows[i]['candidate_mode'] == 'complete' for i in native)

        objective = cfg['distribution_objective']
        model = FormalModel('tcdl', BASE, ablation=cfg['ablation'], distribution_objective=objective)
        model.set_checkpoint_policy('half')
        load_checkpoint(model, cfg['checkpoint'])
        model.eval()
        max_new_tokens = audit['splits']['test_in_domain']['max_json_response_tokens'] + 8
        mine = native[rank::world]
        names = cfg['orders']
        local_original = {}
        started = time.time()

        for order_name in names:
            records = []
            for batch_ids in batches(mine, encs, cfg['eval_token_budget']):
                ordered = [permute_candidates(encs[i], order_for(order_name, encs[i], rows[i]))
                           for i in batch_ids]
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model.predict([rows[i] for i in batch_ids], ordered, max_new_tokens)
                if order_name == 'original':
                    local_original.update({rows[i]['id']: p for i, p in zip(batch_ids, predictions)})
                for i, prediction in zip(batch_ids, predictions):
                    original = local_original.get(rows[i]['id'])
                    if original is None:
                        raise AssertionError('Original prediction missing before comparison')
                    delta = max(abs(a - b) for a, b in zip(original['probabilities'], prediction['probabilities']))
                    records.append({
                        'id': rows[i]['id'], 'task': rows[i]['task'], 'native_k': len(encs[i]['branches']),
                        'target_index': target(rows[i]), 'original_index': original['prediction_index'],
                        'permuted_index': prediction['prediction_index'],
                        'flip': original['prediction_index'] != prediction['prediction_index'],
                        'max_probability_delta': delta,
                    })
                if rank == 0:
                    atomic(out / 'progress.json', {'phase': 'order_audit', 'order': order_name,
                        'local_completed': len(records), 'local_total': len(mine), 'total': len(native),
                        'updated_unix': time.time(), 'elapsed_seconds': time.time() - started})
            parts = [None] * world
            if world > 1:
                dist.all_gather_object(parts, records)
                dist.barrier()
            else:
                parts = [records]
            if rank == 0:
                all_records = [record for part in parts for record in part]
                assert len(all_records) == len(native)
                by_task = {}
                for task in cfg['tasks']:
                    subset = [r for r in all_records if r['task'] == task]
                    by_task[task] = {
                        'rows': len(subset),
                        'flips': sum(r['flip'] for r in subset),
                        'flip_rate': sum(r['flip'] for r in subset) / len(subset),
                        'original_accuracy': sum(r['original_index'] == r['target_index'] for r in subset) / len(subset),
                        'permuted_accuracy': sum(r['permuted_index'] == r['target_index'] for r in subset) / len(subset),
                        'max_probability_delta': max(r['max_probability_delta'] for r in subset),
                        'mean_probability_delta': sum(r['max_probability_delta'] for r in subset) / len(subset),
                    }
                report = {'order': order_name, 'tasks': by_task, 'rows': len(all_records),
                          'native_full_candidates': True, 'evaluation_candidates': 'complete_native',
                          'semantic_alignment': True, 'seconds': time.time() - started}
                atomic(out / f'{order_name}.json', report)
                atomic(out / f'{order_name}.records.json', {'rows': all_records, 'sha256': sha(out / f'{order_name}.json')})

        if rank == 0:
            reports = {name: json.loads((out / f'{name}.json').read_text()) for name in names}
            atomic(out / 'position_sensitivity.json', {
                'status': 'complete', 'model': 'new D2 independent-candidate final',
                'checkpoint': cfg['checkpoint'], 'checkpoint_sha256': cfg['checkpoint_sha256'],
                'orders': names, 'tasks': cfg['tasks'], 'native_full_candidates': True,
                'reports': reports, 'elapsed_seconds': time.time() - started,
                'scope': 'Full native in-domain test; candidate display order permuted while semantic IDs remain fixed; Score excluded by protocol.',
            })
            atomic(out / 'complete.json', {'status': 'complete', 'orders': names,
                'rows': len(native), 'tasks': cfg['tasks'], 'native_full_candidates': True,
                'elapsed_seconds': time.time() - started})
            atomic(out / 'progress.json', {'phase': 'complete', 'orders': names,
                                            'updated_unix': time.time()})
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
    except BaseException as exc:
        atomic(out / f'failure.rank{rank}.json', {'error': str(exc), 'traceback': traceback.format_exc(),
                                                  'time': time.time()})
        raise


if __name__ == '__main__':
    main()
