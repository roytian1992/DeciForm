"""Apply the frozen numeric CDF research module to unchanged native ASAP rows."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import atomic
import asap_score as panel
import worker_score_specialist as driver
from models_score_research import NumericD2


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    assert cfg['score_architecture'] == 'numeric_cdf'
    assert cfg['parent_checkpoint'] is None
    assert cfg['evaluation_protocol'] == panel.PROTOCOL
    assert cfg['distribution_objective']['kind'] == 'soft_cdf'
    captured = {}
    preflight = sys.argv[sys.argv.index('--mode') + 1] == 'preflight'

    class ValidatedNumeric(NumericD2):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured['shared_head_sha256_before_replacement'] = self.initial_shared_head_sha256

        def scores(self, rows, encodings):
            for row in rows:
                lo, hi = panel.RANGES[row['essay_set']]
                assert type(row['label']) is int
                assert panel.support(row) == list(range(lo, hi + 1))
            return super().scores(rows, encodings)

        def predict(self, rows, encodings, max_new_tokens=0):
            outputs = super().predict(rows, encodings, max_new_tokens)
            if preflight:
                predictions = [panel.prediction_row(r, p) for r, p in zip(rows, outputs)]
                assert panel.metrics(predictions)['task/score']['rows'] == len(rows)
                assert panel.score_rule_report(predictions, 'preflight')['rows'] == len(rows)
            return outputs

    batches = driver.batches

    def bounded(indices, encodings, arm, budget, max_rows=16):
        yield from batches(indices, encodings, arm, budget, min(max_rows, cfg['max_microbatch_rows']))

    driver.FormalModel = ValidatedNumeric
    driver.batches = bounded
    driver.PROTOCOL = panel.PROTOCOL
    driver.prediction_row = panel.prediction_row
    driver.metrics = panel.metrics
    driver.score_rule_report = panel.score_rule_report
    driver.main()
    if int(os.getenv('RANK', '0')) == 0:
        out = Path(cfg['output'])
        path = out / ('preflight.json' if preflight else 'complete.json')
        result = json.loads(path.read_text())
        result.update(score_architecture='numeric_cdf', native_integer_labels=True,
                      grid_sizes=[4, 5, 6, 11, 31, 61],
                      max_microbatch_rows=cfg['max_microbatch_rows'], **captured)
        atomic(path, result)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
        atomic(Path(cfg['output']) / f"failure.rank{os.getenv('RANK', '0')}.json",
               {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
