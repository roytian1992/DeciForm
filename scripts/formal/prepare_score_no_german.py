"""Versioned source removal; reuse exact retained rows and full token encodings."""
from collections import Counter
import json
from pathlib import Path
import pickle

from common import PROJECT, atomic, sha


def main():
    source = PROJECT / 'data/score_specialist_continuous_20261104_v1'
    dest = PROJECT / 'data/score_specialist_stsb_enzh_20261105_v1'
    prior = json.loads((source / 'cache/audit.json').read_text())
    assert prior['status'] == 'passed'
    assert prior['data_manifest_sha256'] == sha(source / 'manifest.json')
    dest.mkdir(exist_ok=False)
    (dest / 'cache').mkdir()
    manifest = {'status': 'passed', 'purpose': 'RQ-Score; user removes WMT en-de from future runs',
                'parent': str(source), 'parent_manifest_sha256': sha(source / 'manifest.json'),
                'parent_cache_audit_sha256': sha(source / 'cache/audit.json'),
                'sources': ['stsb', 'wmt_en_zh'], 'excluded_source': 'wmt_en_de',
                'splits': {}, 'seed': 20261101, 'rounded_labels': 0, 'truncated_rows': 0,
                'policy': 'Filter only source identity, preserve retained IDs/order/labels/text/full supports/splits. Parent overlap exclusions are retained, not relaxed.',
                'code_sha256': sha(Path(__file__))}
    audit = {'status': 'passed', 'splits': {}, 'truncated': 0, 'rounded_labels': 0,
             'candidate_sampling': False, 'cache_policy': 'exact retained parent encodings'}
    sets = {}
    for split, entry in prior['splits'].items():
        assert sha(entry['views_path']) == entry['views_sha256']
        rows = [json.loads(line) for line in open(entry['views_path'])]
        cache = entry['arms']['kev']
        assert sha(cache['path']) == cache['sha256']
        with open(cache['path'], 'rb') as f:
            encodings = pickle.load(f)
        assert len(rows) == len(encodings) == entry['rows']
        indices = [i for i, r in enumerate(rows) if r['source_family'] != 'wmt_en_de']
        kept, encs = [rows[i] for i in indices], [encodings[i] for i in indices]
        assert {r['source_family'] for r in kept} == {'stsb', 'wmt_en_zh'}
        assert all(e['labels'] == [None] and not e['state_truncated'] and
                   len(e['opt_idx'][0]) == len(r['score_support']) for r, e in zip(kept, encs))
        sets[split] = {r['group_id'] for r in kept}
        path = dest / f'{split}.jsonl'
        with path.open('x') as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        cachepath = dest / 'cache' / f'{split}.kev.pkl'
        with cachepath.open('xb') as f:
            pickle.dump(encs, f, protocol=5)
        lengths = [len(e['ids']) for e in encs]
        desc = {'path': str(path), 'sha256': sha(path), 'rows': len(kept),
                'by_source': dict(Counter(r['source_family'] for r in kept)),
                'removed_en_de_rows': len(rows) - len(kept),
                'parent_indices': indices}
        manifest['splits'][split] = desc
        audit['splits'][split] = {'rows': len(kept), 'views_path': str(path), 'views_sha256': sha(path),
            'native_evaluation': split != 'train', 'complete_output_grid': True,
            'max_json_response_tokens': 0, 'arms': {'kev': {'path': str(cachepath),
                'sha256': sha(cachepath), 'max_tokens': max(lengths),
                'p99_tokens': sorted(lengths)[int(.99*(len(lengths)-1))],
                'forward_tokens': sum(lengths),
                'longest_indices': sorted(range(len(kept)), key=lambda i: lengths[i], reverse=True)[:10]}}}
    assert not sets['train'] & (sets['dev'] | sets['test_in_domain'])
    manifest['train_heldout_group_overlap'] = 0
    manifest['dev_test_shared_groups'] = len(sets['dev'] & sets['test_in_domain'])
    atomic(dest / 'manifest.json', manifest)
    audit['data_manifest_sha256'] = sha(dest / 'manifest.json')
    atomic(dest / 'cache/audit.json', audit)
    print(json.dumps({'data': str(dest), 'counts': {s: e['rows'] for s, e in audit['splits'].items()}}))


if __name__ == '__main__':
    main()
