"""Isolated RQ-Score driver; matched retained sources and complete score supports."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import atomic
import score_specialist as panel
import worker_score_specialist as driver
from models_score_research import BatchedD2, NumericD2


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    model_type = {'d2_batched': BatchedD2, 'numeric_cdf': NumericD2}[cfg['score_architecture']]
    assert cfg['allowed_sources'] == ['stsb', 'wmt_en_zh']
    assert cfg['distribution_objective']['kind'] == 'soft_cdf'
    assert cfg['parent_checkpoint'] is None
    data = json.loads((Path(cfg['data']) / 'manifest.json').read_text())
    assert data['sources'] == cfg['allowed_sources']
    assert data['excluded_source'] == 'wmt_en_de'
    captured = {}
    preflight = sys.argv[sys.argv.index('--mode') + 1] == 'preflight'

    class ValidatedModel(model_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if hasattr(self, 'initial_shared_head_sha256'):
                captured['shared_head_sha256_before_replacement'] = self.initial_shared_head_sha256

        def scores(self, rows, encodings):
            assert all(r['source_family'] in cfg['allowed_sources'] for r in rows)
            return super().scores(rows, encodings)

        def predict(self, rows, encodings, max_new_tokens=0):
            outputs = super().predict(rows, encodings, max_new_tokens)
            if preflight:
                predictions = [panel.prediction_row(r, p) for r, p in zip(rows, outputs)]
                assert panel.metrics(predictions)['task/score']['rows'] == len(rows)
                assert panel.score_rule_report(predictions, 'preflight')['rows'] == len(rows)
            return outputs

    batches = driver.batches

    def bounded_batches(indices, encs, arm, budget, max_rows=16):
        yield from batches(indices, encs, arm, budget, min(max_rows, cfg['max_microbatch_rows']))

    def report(rows, split):
        return {**panel.score_rule_report(rows, split),
                'evaluation_protocol': cfg['evaluation_protocol'],
                'allowed_sources': cfg['allowed_sources']}

    driver.FormalModel = ValidatedModel
    driver.batches = bounded_batches
    driver.PROTOCOL = cfg['evaluation_protocol']
    driver.score_rule_report = report
    driver.main()
    if int(os.getenv('RANK', '0')) == 0:
        out = Path(cfg['output'])
        path = out / ('preflight.json' if preflight else 'complete.json')
        evidence = json.loads(path.read_text())
        evidence.update(score_architecture=cfg['score_architecture'],
                        allowed_sources=cfg['allowed_sources'], grid_sizes=[51, 101],
                        max_microbatch_rows=cfg['max_microbatch_rows'], **captured)
        atomic(path, evidence)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
        atomic(Path(cfg['output']) / f"failure.rank{os.getenv('RANK', '0')}.json",
               {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
