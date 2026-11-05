"""Frozen independent D2 on the existing, matched Kev learning-curve subsets."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import worker_score_curve as driver
from common import atomic
from independent_candidates import split_candidates
from models_independent import FormalModel as IndependentModel


class FormalModel(IndependentModel):
    def scores(self, rows, encodings):
        return super().scores(rows, [split_candidates(e) for e in encodings])


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    assert cfg['candidate_encoder'] == 'shared_prefix_independent_branches'
    assert cfg['parent_checkpoint'] is None
    driver.FormalModel = FormalModel
    driver.main()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
        atomic(Path(cfg['output']) / f"failure.rank{os.getenv('RANK', '0')}.json",
               {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
