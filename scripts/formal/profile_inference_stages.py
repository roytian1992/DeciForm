"""Read-only per-stage diagnosis; never replaces official latency measurements."""
import argparse
import collections
from contextlib import contextmanager
import json
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch

from benchmark_inference import ROOT as TIMING_ROOT, load_model, request
from common import PROJECT, atomic, json_prompt, sha, support

ROOT = PROJECT/'runs/formal_inference_profile_20261106_v1'


class Timer:
    def __init__(self):
        self.events = []

    @contextmanager
    def stage(self, name):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        began = time.perf_counter()
        try:
            yield
        finally:
            cpu_ms = (time.perf_counter()-began)*1000
            end.record()
            self.events.append((name, start, end, cpu_ms))

    def collect(self):
        torch.cuda.synchronize()
        groups = collections.defaultdict(list)
        for name, start, end, cpu in self.events:
            groups[name].append((start.elapsed_time(end), cpu))
        self.events = []
        return {k: {'calls': len(v), 'cuda_timeline_ms': sum(x[0] for x in v),
                    'cpu_enqueue_ms': sum(x[1] for x in v)} for k, v in groups.items()}


def instrument(model, arm, timer):
    # Wrapping records events only: all original functions/tensors are retained.
    backbone = model.lm.get_base_model()
    for index, layer in enumerate(backbone.layers):
        original = layer.forward
        counter = [0]
        def wrapper(*args, _original=original, _counter=counter, **kwargs):
            name = ('prefix_layers' if _counter[0] % 2 == 0 else 'candidate_layers') if arm == 'd2' else 'decoder_layers'
            _counter[0] += 1
            with timer.stage(name):
                return _original(*args, **kwargs)
        layer.forward = wrapper

    if arm == 'd2':
        import independent_candidates as independent
        original_routing = independent.routing
        @contextmanager
        def routing(*args, **kwargs):
            context = original_routing(*args, **kwargs)
            with timer.stage('routing_prepare'):
                context.__enter__()
            try:
                yield
            finally:
                with timer.stage('routing_clear'):
                    context.__exit__(None, None, None)
        independent.routing = routing
        original_branch = independent.Prefix.branch
        def branch(prefix, owner):
            with timer.stage('prefix_cache_expand'):
                return original_branch(prefix, owner)
        independent.Prefix.branch = branch
        original_head = model.net.head.padded
        def head(*args, **kwargs):
            with timer.stage('head'):
                return original_head(*args, **kwargs)
        model.net.head.padded = head
    else:
        original_head = model.net.head.forward
        def head(*args, **kwargs):
            with timer.stage('head'):
                return original_head(*args, **kwargs)
        model.net.head.forward = head


def worker(root, arm):
    cfg = json.loads((TIMING_ROOT/'config.json').read_text())
    rows = [json.loads(s) for s in (root/'sample.jsonl').read_text().splitlines()]
    out = root/arm
    out.mkdir()
    try:
        torch.set_num_threads(cfg['cpu_threads'])
        os.sched_setaffinity(0, cfg['cpu_affinity'])
        torch.cuda.set_device(0)
        assert sha(cfg['checkpoints'][arm]['path']) == cfg['checkpoints'][arm]['sha256']
        atomic(out/'status.json', {'status': 'loading'})
        model, tok = load_model(arm, cfg)
        plain, measured = [], []
        reference = {}
        with torch.inference_mode():
            for row in rows:
                request(model, tok, arm, row, cfg['max_new_tokens'])
            for repeat in range(3):
                for row in rows:
                    torch.cuda.synchronize()
                    prediction, tokens, timing = request(model, tok, arm, row, cfg['max_new_tokens'])
                    reference[row['id']] = prediction
                    plain.append({'id': row['id'], 'task': row['task'], 'k': len(support(row)),
                                  'repeat': repeat, **tokens, **timing})
            timer = Timer()
            instrument(model, arm, timer)
            for row in rows:
                request(model, tok, arm, row, cfg['max_new_tokens'])
                timer.collect()
            for repeat in range(3):
                for row in rows:
                    torch.cuda.synchronize()
                    prediction, tokens, timing = request(model, tok, arm, row, cfg['max_new_tokens'])
                    stages = timer.collect()
                    expected = reference[row['id']]
                    delta = max(abs(a-b) for a,b in zip(prediction['probabilities'], expected['probabilities']))
                    assert delta == 0 and prediction['prediction_index'] == expected['prediction_index']
                    measured.append({'id': row['id'], 'task': row['task'], 'k': len(support(row)),
                                     'repeat': repeat, **tokens, **timing, 'stages': stages,
                                     'probability_delta_from_uninstrumented': delta})
        names = sorted({s for r in measured for s in r['stages']})
        stages = {s: {k: statistics.mean(r['stages'].get(s, {}).get(k, 0) for r in measured)
                      for k in ('calls', 'cuda_timeline_ms', 'cpu_enqueue_ms')} for s in names}
        atomic(out/'results.json', {'arm': arm, 'plain': plain, 'instrumented': measured,
                                  'plain_mean_ms': statistics.mean(r['e2e_ms'] for r in plain),
                                  'instrumented_mean_ms': statistics.mean(r['e2e_ms'] for r in measured),
                                  'stages_mean': stages,
                                  'notes': 'CUDA event intervals include launch/idle gaps on the stream. CPU enqueue overlaps GPU execution; do not add CPU and GPU times. Instrumentation changes timing, never use this as the official latency table.'})
        atomic(out/'status.json', {'status': 'complete', 'plain_requests': len(plain), 'instrumented_requests': len(measured)})
    except Exception:
        atomic(out/'status.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


def main(root):
    root.mkdir()
    cfg = json.loads((TIMING_ROOT/'config.json').read_text())
    population = [json.loads(s) for s in Path(cfg['sample_path']).read_text().splitlines()]
    selected = []
    for source in ['ag_news', 'cfpb', 'banking77', 'boolq', 'helpsteer3', 'prometheus']:
        xs = sorted([r for r in population if r['source_family'] == source], key=lambda r: len(json_prompt(r)))
        selected.append(xs[len(xs)//2])
    (root/'sample.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    atomic(root/'manifest.json', {'purpose': 'Explain D2 latency without changing weights, model functions, or official timings',
                                'selection': 'median prompt-character length within six fixed sources from the existing frozen timing sample',
                                'rows': len(selected), 'script_sha256': sha(__file__),
                                'parent_config': str(TIMING_ROOT/'config.json'), 'parent_config_sha256': sha(TIMING_ROOT/'config.json'),
                                'sample_sha256': sha(root/'sample.jsonl'), 'python': sys.executable,
                                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    for arm in ('kev', 'd2'):
        atomic(root/'status.json', {'status': 'running', 'arm': arm})
        with open(root/f'{arm}.log', 'x') as handle:
            subprocess.run([sys.executable, '-u', __file__, '--root', str(root), '--arm', arm],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic(root/'status.json', {'status': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=('kev', 'd2'))
    args = parser.parse_args()
    worker(args.root, args.arm) if args.arm else main(args.root)
