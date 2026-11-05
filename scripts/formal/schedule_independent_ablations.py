"""Launch A1-A3 concurrently, one two-GPU supervisor per arm."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from common import atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root_manifest = json.loads((args.root / 'experiment_manifest.json').read_text())
    arms = root_manifest['arms']
    children = []
    for name, item in arms.items():
        manifest = json.loads(Path(item['manifest']).read_text())
        log = (args.root / name / 'supervisor.log').open('a', buffering=1)
        env = {**os.environ, **manifest['environment_variables']}
        cmd = manifest['launch_command']
        child = subprocess.Popen(cmd, cwd=args.root.parent.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        children.append((name, child, log, cmd))
    atomic(args.root / 'scheduler.json', {'phase': 'running', 'started_unix': time.time(),
                                          'arms': {name: {'pid': child.pid, 'command': cmd}
                                                   for name, child, _, cmd in children}})
    while children:
        live = []
        states = {}
        for name, child, log, cmd in children:
            code = child.poll()
            status_path = args.root / name / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            states[name] = {'pid': child.pid, 'exit_code': code, 'status': status}
            if code is None:
                live.append((name, child, log, cmd))
            elif code:
                atomic(args.root / 'scheduler.json', {'phase': 'failed', 'failed_arm': name,
                                                       'exit_code': code, 'states': states})
                raise SystemExit(code)
            else:
                log.close()
        children = live
        atomic(args.root / 'scheduler.json', {'phase': 'running' if children else 'complete',
                                              'updated_unix': time.time(), 'states': states})
        if children:
            time.sleep(10)


if __name__ == '__main__':
    main()
