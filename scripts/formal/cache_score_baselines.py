"""Add baseline encodings separately, never mutate frozen Score datasets/caches."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import pickle
from pathlib import Path

from common import BASE, PROJECT, atomic, sha, cost
from score_baselines import encode_baseline

TOKENIZER = None


def initialize():
    global TOKENIZER
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    TOKENIZER=AutoTokenizer.from_pretrained(BASE,local_files_only=True)


def chunk(job):
    arm,rows=job
    return [encode_baseline(r,arm,TOKENIZER) for r in rows]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    root=PROJECT/'data/score_baseline_encodings_20261105_v1'
    root.mkdir(exist_ok=False)
    panels={'asap':'score_specialist_asap_fold0_20261104_v1',
            'continuous':'score_specialist_continuous_20261104_v1'}
    with concurrent.futures.ProcessPoolExecutor(args.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for panel,name in panels.items():
            data=PROJECT/'data'/name
            old=json.loads((data/'cache/audit.json').read_text())
            assert old['status']=='passed' and sha(data/'manifest.json')==old['data_manifest_sha256']
            out=root/panel;out.mkdir()
            audit=copy.deepcopy(old)
            audit.update(status='encoding',data=str(data),original_audit_sha256=sha(data/'cache/audit.json'),
                original_audit=str(data/'cache/audit.json'),code_sha256=sha(Path(__file__)),
                encoder_sha256=sha(PROJECT/'scripts/formal/score_baselines.py'))
            atomic(out/'audit.json',audit)
            for split,entry in audit['splits'].items():
                assert sha(entry['views_path'])==entry['views_sha256']
                rows=[json.loads(line) for line in open(entry['views_path'])]
                assert len(rows)==entry['rows']
                for arm in ('lora','json_sft'):
                    encs=[]
                    for batch in pool.map(chunk,[(arm,rows[i:i+64]) for i in range(0,len(rows),64)]):
                        encs.extend(batch)
                        atomic(root/'progress.json',{'panel':panel,'split':split,'arm':arm,'rows':len(encs),'total':len(rows)})
                    lengths=[cost(e,arm)[1] for e in encs]
                    assert len(encs)==len(rows) and max(lengths)<262144
                    path=out/f'{split}.{arm}.pkl'
                    with path.open('xb') as handle:pickle.dump(encs,handle,protocol=5)
                    entry['arms'][arm]={'path':str(path),'sha256':sha(path),'max_tokens':max(lengths),
                        'forward_tokens':sum(lengths),'longest_indices':sorted(range(len(rows)),key=lambda i:lengths[i],reverse=True)[:10]}
                    if arm=='json_sft':entry['max_json_response_tokens']=max(len(e['answer']) for e in encs)
                    atomic(out/'audit.json',audit)
                    print(panel,split,arm,len(rows),'max_tokens',max(lengths),flush=True)
            audit.update(status='passed',truncated=0,rounded_labels=0,candidate_sampling=False)
            atomic(out/'audit.json',audit)
    atomic(root/'complete.json',{'status':'passed','panels':panels,'truncated':0,'rounded_labels':0})


if __name__=='__main__':main()
