"""Freeze exact formal views and full-text Laya tokenizations; no dataset edits."""
import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

from common import PROJECT, atomic, sha
from laya_adapter import make_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'vendor/laya_release'))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root/'base', local_files_only=True)
    data = PROJECT/'data/formal_200k_20261101_v1'
    old = json.loads((data/'cache/audit.json').read_text())
    audit = {'source_audit_sha256': sha(data/'cache/audit.json'), 'splits': {},
             'tokenizer': str(root/'base'), 'truncated': 0, 'dropped': 0,
             'format': 'upstream Laya template, without option/instruction/state token clipping'}
    cache = root/'cache'
    cache.mkdir(exist_ok=True)
    token_cache = {}
    for split in ('train', 'dev', 'test_in_domain'):
        desc = old['splits'][split]
        assert sha(desc['views_path']) == desc['views_sha256']
        rows = [json.loads(line) for line in open(desc['views_path'])]
        assert len(rows) == desc['rows']
        if split != 'train':
            assert all(r['candidate_mode'] == 'complete' for r in rows)
        allparts = []
        for start in range(0, len(rows), 256):
            chunk = rows[start:start+256]
            states = [r['state']['text'].replace(tok.mask_token, ' ') for r in chunk]
            tokens = tok(states, add_special_tokens=False, truncation=False)['input_ids']
            allparts.extend(make_parts(r, tok, ids, token_cache) for r,ids in zip(chunk,tokens))
            if start % 4096 == 0:
                atomic(root/'prepare_progress.json', {'split': split, 'completed':len(allparts),
                                                      'total':len(rows), 'updated_unix':time.time()})
        path = cache/f'{split}.pkl'
        with path.open('wb') as f:
            pickle.dump(allparts, f, protocol=5)
        lengths = sorted(p['length'] for p in allparts)
        longest = sorted(range(len(rows)), key=lambda i:allparts[i]['length'], reverse=True)[:10]
        audit['splits'][split] = {'rows':len(rows), 'views_path':desc['views_path'],
            'views_sha256':desc['views_sha256'], 'path':str(path), 'sha256':sha(path),
            'task_counts':dict(collections.Counter(r['task'] for r in rows)),
            'source_counts':dict(collections.Counter(r['source_family'] for r in rows)),
            'max_tokens':max(lengths), 'p99_tokens':lengths[int(len(lengths)*.99)],
            'tokens':sum(lengths), 'over_8192':sum(n>8192 for n in lengths),
            'longest_indices':longest, 'longest_ids':[rows[i]['id'] for i in longest],
            'native_evaluation':split!='train'}
        atomic(cache/'audit.json', audit)
        print(split, json.dumps(audit['splits'][split]), flush=True)
    audit['status'] = 'passed'
    atomic(cache/'audit.json', audit)


if __name__ == '__main__':
    main()
