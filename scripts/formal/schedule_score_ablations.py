"""Run one or more isolated D2 ablations in sequence on a GPU group."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from common import PROJECT, atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()
    experiment = json.loads((args.root / 'experiment_manifest.json').read_text())
    queues = experiment['gpu_queues']
    matched = [group for group, arms in queues.items() if arms == args.arms]
    if len(matched) != 1:
        raise ValueError('Requested arms do not match one frozen GPU queue')
    gpu_group = matched[0]
    queue_state = args.root / ('queue_' + gpu_group.replace(',', '') + '.json')
    for index, name in enumerate(args.arms):
        manifest = json.loads((args.root / name / 'experiment_manifest.json').read_text())
        if manifest['gpus'] != gpu_group:
            raise ValueError('GPU queue and arm manifest disagree')
        atomic(queue_state, {'phase': 'running', 'gpus': gpu_group, 'arm': name,
                             'index': index, 'total': len(args.arms),
                             'updated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
        with (args.root / name / 'supervisor.log').open('a', buffering=1) as log:
            code = subprocess.run(manifest['launch_command'], cwd=PROJECT,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            atomic(queue_state, {'phase': 'failed', 'gpus': gpu_group, 'arm': name,
                                 'exit_code': code})
            raise SystemExit(code)
    atomic(queue_state, {'phase': 'complete', 'gpus': gpu_group,
                         'completed_arms': args.arms})


if __name__ == '__main__':
    main()
