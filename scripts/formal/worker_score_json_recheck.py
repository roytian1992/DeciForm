"""Fresh JSON-LoRA rerun with target-free generation inputs by construction."""
import json
import os
from pathlib import Path
import sys
import traceback
import time

from common import atomic
import worker_score_baselines as driver
from models_score_baselines import FormalModel as Baseline


class TargetFreeGeneration(Baseline):
    def predict(self, rows, encodings, max_new_tokens):
        if self.arm != 'json_sft':
            raise ValueError('This replay is the generative baseline only')
        safe_rows = [{k: v for k, v in row.items() if k != 'label'} for row in rows]
        safe_encodings = [{'prompt': encoding['prompt']} for encoding in encodings]
        return super().predict(safe_rows, safe_encodings, max_new_tokens)


if __name__ == '__main__':
    try:
        driver.FormalModel = TargetFreeGeneration
        driver.main()
        if int(os.getenv('RANK', '0')) == 0:
            cfg = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
            path = Path(cfg['output']) / ('preflight.json' if 'preflight' == sys.argv[sys.argv.index('--mode')+1] else 'complete.json')
            proof = json.loads(path.read_text())
            proof['generation_receives_labels_or_answer_tokens'] = False
            atomic(path, proof)
    except BaseException as exc:
        cfg = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
        atomic(Path(cfg['output']) / f"failure.rank{os.getenv('RANK', '0')}.json",
               {'time': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
