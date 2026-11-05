"""Explicit panel/model wiring for independent-D2 Score comparisons."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import asap_score
import score_specialist
import worker_score_specialist as driver
from common import atomic
from models_independent_score_specialist import BRANCH_CHUNK_SIZE, FormalModel


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    panel = {'asap': asap_score, 'continuous': score_specialist}[cfg['score_panel']]
    assert cfg['score_branch_chunk_size'] == BRANCH_CHUNK_SIZE
    assert cfg['evaluation_protocol'] == panel.PROTOCOL
    preflight = '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] == 'preflight'
    checked_sources = set()

    class ValidatedModel(FormalModel):
        def predict(self, rs, es, max_new_tokens=0):
            predictions = super().predict(rs, es, max_new_tokens)
            if preflight:
                # Exercise the actual downstream reporting functions before training.
                rows = [panel.prediction_row(r, p) for r, p in zip(rs, predictions)]
                report = panel.metrics(rows)
                rules = panel.score_rule_report(rows, 'preflight')
                assert report['task/score']['rows'] == rules['rows'] == len(rs)
                assert abs(report['task/score']['mae'] - rules['primary_mae']) < 1e-10
                checked_sources.update(r['source_family'] for r in rs)
            return predictions

    driver.FormalModel = ValidatedModel
    for name in ('PROTOCOL', 'TASKS', 'prediction_row', 'metrics', 'score_rule_report'):
        setattr(driver, name, getattr(panel, name))
    driver.main()
    if int(os.getenv('RANK', '0')) == 0:
        out = Path(cfg['output'])
        if preflight:
            assert checked_sources, 'No real prediction/reporting path exercised'
            atomic(out / 'preflight.reporting.json', {
                'status': 'passed', 'panel': cfg['score_panel'],
                'sources_exercised': sorted(checked_sources),
                'functions': ['prediction_row', 'metrics', 'score_rule_report']})
        else:
            path = out / 'complete.json'
            complete = json.loads(path.read_text())
            complete.update(score_panel=cfg['score_panel'], grid_sizes=cfg['grid_sizes'],
                            continuous_labels_preserved=cfg['score_panel'] == 'continuous',
                            native_integer_labels=cfg['score_panel'] == 'asap',
                            candidate_encoder=cfg['candidate_encoder'])
            atomic(path, complete)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        try:
            cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
            atomic(Path(cfg['output']) / f"failure.rank{os.getenv('RANK', '0')}.json",
                   {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc(),
                    'mode': sys.argv})
        except Exception:
            pass
        raise
