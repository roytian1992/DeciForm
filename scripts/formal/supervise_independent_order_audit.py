"""Launch and monitor the final-D2 full-native position audit."""
import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from common import PROJECT, atomic, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    try:
        config = Path(manifest['config'])
        assert sha(config) == manifest['config_sha256']
        for path, digest in manifest['code_sha256'].items():
            assert sha(PROJECT / path) == digest, f'Code changed since audit freeze: {path}'
        if (root / 'complete.json').exists():
            return
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': manifest['gpus'],
               'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false', 'FLA_DISABLE_TENSOR_CACHE': '1'}
        cmd = manifest['launch_command']
        with (root / 'audit.log').open('a', buffering=1) as log:
            child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            start = time.time()
            while True:
                code = child.poll()
                progress = json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').exists() else {}
                atomic(root / 'status.json', {'phase': 'running' if code is None else 'process_complete',
                    'gpus': manifest['gpus'], 'pid': child.pid, 'command': cmd,
                    'updated_unix': time.time(), 'progress': progress, 'exit_code': code,
                    'elapsed_seconds': time.time() - start, 'log': str(root / 'audit.log')})
                if code is not None:
                    if code:
                        raise RuntimeError(f'Position audit exited {code}; see {root / "audit.log"}')
                    break
                time.sleep(5)
        complete = json.loads((root / 'complete.json').read_text())
        assert complete['status'] == 'complete' and complete['native_full_candidates']
        atomic(root / 'status.json', {'phase': 'complete', 'gpus': manifest['gpus'],
                                      'updated_unix': time.time(), 'exit_code': 0, 'complete': complete})
    except Exception as exc:
        atomic(root / 'alerts.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'time': time.time()})
        atomic(root / 'status.json', {'phase': 'failed', 'error': str(exc), 'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
