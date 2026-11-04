"""Frozen-checkpoint, full-candidate score/noul answer-order evaluation."""
import argparse
import collections
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import torch

from common import PROJECT, BASE, SEED, atomic, sha, support, target, cost
from models import FormalModel as BaselineModel
from models_score_distribution import FormalModel as FullModel
from order_audit import (PANELS, applicable, encode_view, input_hash,
                         normalize_prediction, summarize_records, view_orders)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    spec = manifest['models'][args.model]
    arm = spec['arm']
    out = root / args.model
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('This model audit already completed')
    journal_path = out / 'predictions.jsonl'
    if journal_path.exists() and not args.resume:
        raise FileExistsError('Use --resume for a partial prediction journal')
    for path, digest in manifest['code_sha256'].items():
        assert sha(PROJECT / path) == digest, path
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(spec['config']) == spec['config_sha256']
    assert sha(root / 'panel.json') == manifest['panel_sha256']
    assert sha(manifest['views_path']) == manifest['views_sha256']
    assert sha(root / 'protocol.json') == manifest['protocol_sha256']
    protocol = json.loads((root / 'protocol.json').read_text())
    panel = json.loads((root / 'panel.json').read_text())
    all_rows = [json.loads(line) for line in open(manifest['views_path'])]
    by_id = {row['id']: i for i, row in enumerate(all_rows)}
    with open(manifest['caches'][arm]['path'], 'rb') as handle:
        assert sha(manifest['caches'][arm]['path']) == manifest['caches'][arm]['sha256']
        cached = pickle.load(handle)
    assert len(cached) == len(all_rows)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    done = set()
    if args.resume:
        for line in journal_path.open():
            record = json.loads(line)
            key = (record['panel'], record['id'], record['view'])
            if key in done:
                raise ValueError('Duplicate saved prediction')
            done.add(key)

    def progress(phase, **extra):
        atomic(out / 'progress.json', {'phase': phase, 'model': args.model,
            'updated_unix': time.time(), 'completed_predictions': len(done),
            'expected_predictions': spec['expected_predictions'],
            'seconds': time.time()-started, **extra})

    progress('loading_model')
    config = json.loads(Path(spec['config']).read_text())
    if arm == 'tcdl':
        model = FullModel(arm, BASE, checkpointing=False, ablation=config['ablation'],
                          distribution_objective=config['distribution_objective'])
    else:
        model = BaselineModel(arm, BASE, checkpointing=False)
    state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    weights = state['trainable']
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(named) != set(weights):
        raise ValueError('Checkpoint and inference parameter inventory differ')
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(weights[name])
            if not torch.equal(parameter.detach().cpu(), weights[name]):
                raise ValueError(f'Checkpoint tensor did not restore exactly: {name}')
    atomic(out / 'checkpoint_loaded.json', {'checkpoint': spec['checkpoint'],
        'checkpoint_sha256': spec['checkpoint_sha256'], 'exact_tensor_restore': True,
        'trainable_parameter_count': sum(p.numel() for p in named.values()),
        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'training_updates': 0})
    del state, weights, named
    model.eval().requires_grad_(False)
    expected = set()
    controls = []
    with journal_path.open('a' if args.resume else 'x', buffering=1) as journal:
        for panel_name in PANELS:
            for task in ('noul', 'score'):
                rows = [all_rows[by_id[identifier]] for identifier in panel[panel_name]
                        if all_rows[by_id[identifier]]['task'] == task]
                bundles = []
                progress('encoding', panel=panel_name, task=task)
                for index, row in enumerate(rows):
                    original = cached[by_id[row['id']]]
                    views = view_orders(arm, panel_name, row)
                    encs = {name: encode_view(row, model.tok, arm, panel_name, order,
                                              original=original) for name, order in views}
                    if panel_name == 'native':
                        assert encs['original'] == original, 'Identity changed the native model input'
                    hashes = {name: input_hash(e) for name, e in encs.items()}
                    if applicable(arm, panel_name, task):
                        assert all(h != hashes['original'] for name,h in hashes.items() if name != 'original'), 'An intended order perturbation was a no-op'
                    length = max(cost(e, arm)[1] for e in encs.values())
                    bundles.append((row, views, encs, hashes, length))
                    expected.update((panel_name, row['id'], name) for name,_ in views)
                    if index % 100 == 0:
                        progress('encoding', panel=panel_name, task=task, encoded_rows=index+1,
                                 total_rows=len(rows))
                bundles.sort(key=lambda item: (-item[4], item[0]['id']))
                chunks, current, longest = [], [], 0
                for bundle in bundles:
                    if current and ((len(current)+1)*max(longest, bundle[4]) > protocol['token_budget'] or
                                    len(current) >= protocol['max_batch_rows']):
                        chunks.append(current); current=[]; longest=0
                    current.append(bundle); longest=max(longest,bundle[4])
                if current:
                    chunks.append(current)
                for chunk_index, chunk in enumerate(chunks):
                    for view, order in chunk[0][1]:
                        batch = [item for item in chunk if (panel_name,item[0]['id'],view) not in done]
                        if not batch:
                            continue
                        rs = [item[0] for item in batch]
                        es = [item[2][view] for item in batch]
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            predictions = model.predict(rs, es, protocol['max_new_tokens'])
                            if chunk_index == 0 and view == 'original':
                                repeated = model.predict(rs, es, protocol['max_new_tokens'])
                                controls.append({'panel': panel_name, 'task': task,
                                    'rows': len(rs), 'all_outputs_identical': predictions == repeated})
                        assert len(predictions) == len(rs)
                        for item, prediction in zip(batch, predictions):
                            row = item[0]
                            prediction = normalize_prediction(row, prediction)
                            record = {**prediction, 'id': row['id'], 'source': row['source_family'],
                                'group_id': row['group_id'], 'task': task, 'label': row['label'],
                                'label_index': target(row), 'support': support(row), 'panel': panel_name,
                                'view': view, 'order': order,
                                'order_test_applicable': applicable(arm, panel_name, task),
                                'input_sha256': item[3][view],
                                'original_input_sha256': item[3]['original']}
                            journal.write(json.dumps(record, ensure_ascii=False)+'\n')
                            done.add((panel_name,row['id'],view))
                    progress('evaluating', panel=panel_name, task=task,
                             completed_batches=chunk_index+1, total_batches=len(chunks),
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                del bundles, chunks
    assert done == expected and len(done) == spec['expected_predictions']
    atomic(out / 'repeat_controls.json', controls)
    progress('summarizing')
    groups = collections.defaultdict(list)
    source_groups = collections.defaultdict(list)
    for line in journal_path.open():
        record = json.loads(line)
        groups[(record['panel'],record['task'])].append(record)
        source_groups[(record['panel'],record['task'],record['source'])].append(record)
    report = {'model': args.model, 'checkpoint_sha256': spec['checkpoint_sha256'],
              'protocol_sha256': manifest['protocol_sha256'], 'evaluated_predictions': len(done),
              'groups': {f'{p}/{t}': summarize_records(records) for (p,t),records in groups.items()},
              'by_source': {f'{p}/{t}/{s}': summarize_records(records, with_interval=False)
                            for (p,t,s),records in source_groups.items()},
              'repeat_controls': controls, 'predictions_sha256': sha(journal_path),
              'seconds': time.time()-started,
              'peak_memory_gib': torch.cuda.max_memory_allocated()/2**30}
    atomic(out / 'report.json', report)
    atomic(out / 'complete.json', {'status': 'complete', 'evaluated_predictions': len(done),
          'seconds': time.time()-started, 'report_sha256': sha(out / 'report.json')})
    progress('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import sys
        try:
            root = Path(sys.argv[sys.argv.index('--root')+1])
            name = sys.argv[sys.argv.index('--model')+1]
            atomic(root / name / 'failure.json', {'error': str(exc),
                'traceback': traceback.format_exc(), 'updated_unix': time.time()})
        except Exception:
            pass
        raise
