"""Reuse exactly matching retained JSON encodings for the no-German Score panel."""
import copy
import json
from pathlib import Path
import pickle

from common import PROJECT, atomic, sha, cost


def main():
    target = PROJECT / 'data/score_specialist_stsb_enzh_20261105_v1'
    parent_path = PROJECT / 'data/score_baseline_encodings_20261105_v1/continuous/audit.json'
    parent = json.loads(parent_path.read_text())
    audit = json.loads((target / 'cache/audit.json').read_text())
    assert audit['status'] == parent['status'] == 'passed'
    dest = PROJECT / 'data/score_baseline_encodings_stsb_enzh_20261106_v1'
    dest.mkdir(exist_ok=False)
    for split, entry in audit['splits'].items():
        old = parent['splits'][split]
        assert sha(old['views_path']) == old['views_sha256']
        assert sha(entry['views_path']) == entry['views_sha256']
        old_rows = [json.loads(line) for line in open(old['views_path'])]
        rows = [json.loads(line) for line in open(entry['views_path'])]
        ids = {row['id']: i for i, row in enumerate(old_rows)}
        assert len(ids) == len(old_rows)
        indices = [ids[r['id']] for r in rows]
        assert all(old_rows[i] == row for i, row in zip(indices, rows))
        assert {r['source_family'] for r in rows} == {'stsb', 'wmt_en_zh'}
        enc = old['arms']['json_sft']
        assert sha(enc['path']) == enc['sha256']
        with open(enc['path'], 'rb') as f:
            parent_encodings = pickle.load(f)
        encodings = [parent_encodings[i] for i in indices]
        path = dest / (split+'.json_sft.pkl')
        with path.open('xb') as f:
            pickle.dump(encodings, f, protocol=5)
        lengths = [cost(e, 'json_sft')[1] for e in encodings]
        entry['arms']['json_sft'] = {'path': str(path), 'sha256': sha(path),
            'max_tokens': max(lengths), 'forward_tokens': sum(lengths),
            'longest_indices': sorted(range(len(rows)), key=lambda i: lengths[i], reverse=True)[:10]}
        entry['max_json_response_tokens'] = max(len(e['answer']) for e in encodings)
    audit.update(status='passed', data=str(target), original_audit=str(target/'cache/audit.json'),
                 original_audit_sha256=sha(target/'cache/audit.json'),
                 parent_json_audit_sha256=sha(parent_path), code_sha256=sha(Path(__file__)),
                 encoder_sha256=sha(PROJECT/'scripts/formal/score_baselines.py'),
                 cache_policy='Exact retained parent JSON encodings; same original text and fractional targets')
    atomic(dest/'audit.json', audit)
    print(dest/'audit.json')


if __name__ == '__main__':
    main()
