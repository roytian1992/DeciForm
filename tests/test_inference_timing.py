import collections
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/formal'))
import benchmark_inference as bench


def test_length_stratified_sample_preserves_sources_and_native_options():
    rows = []
    for task in bench.TASKS:
        for source, n in [('large', 80), ('small', 20)]:
            for i in range(n):
                q = {'instructions': 'decide', 'options': ['a', 'b', 'c'], 'scale': {'min': 0, 'max': 4}}
                rows.append({'id': f'{task}/{source}/{i}', 'task': task, 'source_family': source,
                             'candidate_mode': 'complete', 'state': {'text': 'x' * (i+1)},
                             'json_keys': ['c0', 'c1', 'c2'], 'question': q})
    sample = bench.select_rows(rows, per_task=10)
    assert sample == bench.select_rows(rows, per_task=10)
    assert len({r['id'] for r in sample}) == 30
    assert collections.Counter((r['task'], r['source_family']) for r in sample) == {
        (task, source): count for task in bench.TASKS for source, count in [('large', 8), ('small', 2)]}
    assert all(r in rows for r in sample)
    assert all(len(r['question']['options']) == 3 for r in sample)


def test_serial_throughput_uses_reciprocal_of_mean_latency():
    stats = bench.describe([100, 300])
    assert stats['mean_ms'] == 200
    assert stats['serial_requests_s'] == 5
    assert stats['p95_ms'] == 300


def test_repetitions_are_not_unique_examples():
    records = [dict(id='one', task='choice', source='banking77', k=77,
                    repeat=i, e2e_ms=100+i, encode_ms=1, inference_ms=99+i,
                    prediction={'valid': True}) for i in range(3)]
    stats = bench.summarize(records)
    assert stats['all']['count'] == 3
    assert stats['task_k/choice/77']['mean_ms'] == 101
    assert stats['repeat/0']['count'] == 1
