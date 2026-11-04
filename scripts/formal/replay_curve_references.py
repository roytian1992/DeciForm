"""Replay existing Kev size points with the same frozen score median rule as D2."""
import argparse
import json
from pathlib import Path

from common import PROJECT, atomic, sha
from score_distribution import PROTOCOL, score_rule_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'experiment_manifest.json').read_text())
    assert manifest['evaluation_protocol'] == PROTOCOL
    assert sha(args.root / 'evaluation_protocol.json') == manifest['protocol_file_sha256']
    references = {f'Kev_{size}k': PROJECT / f'runs/formal_next_20261102/kev_{size}k'
                  for size in (25, 50, 100)}
    references['Kev_200k'] = PROJECT / 'runs/formal_main_20261101_v2_one_epoch/kev'
    references['D2_full_200k'] = Path(manifest['reference_200k'])
    for split, expected in manifest['evaluation_rows'].items():
        destination = args.root / 'reference_score_rules' / f'{split}.json'
        if destination.exists():
            raise FileExistsError(destination)
        report = {'protocol': PROTOCOL, 'split': split, 'rows': expected,
                  'scope': 'CPU replay of unchanged saved full-native predictions', 'models': {}}
        signature = None
        for name, run in references.items():
            paths = sorted((run / 'evaluations').glob(f'epoch_1.{split}.rank*.jsonl'))
            rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
            metric_path = run / 'evaluations' / f'epoch_1.{split}.json'
            metrics = json.loads(metric_path.read_text())
            assert metrics['native_full_candidates']
            assert len(rows) == len({row['id'] for row in rows}) == metrics['rows'] == expected
            current = sorted((row['id'], row['task'], row['label'], tuple(row['support'])) for row in rows)
            if signature is None:
                signature = current
            assert current == signature, f'Panel mismatch: {name}'
            score = score_rule_report(rows, split)
            original_rule = 'median' if name == 'D2_full_200k' else 'argmax'
            assert abs(score['rules'][original_rule]['mae'] - metrics['task/score']['mae']) < 1e-10
            report['models'][name] = {'primary': {
                'choice_accuracy': metrics['task/choice']['accuracy'],
                'noul_accuracy': metrics['task/noul']['accuracy'],
                'score_mae': score['primary_mae'], 'score_rule': 'median'},
                'score_rules': score,
                'source_sha256': {str(path): sha(path) for path in [*paths, metric_path]}}
        atomic(destination, report)
        print(destination)


if __name__ == '__main__':
    main()
