"""Tokenize complete continuous score grids on CPU without truncation."""
import argparse
import concurrent.futures
import json
import multiprocessing
import pickle
from pathlib import Path

from common import PROJECT, BASE, atomic, sha
from score_specialist import encode_score

TOKENIZER=None


class CachedTokenizer:
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.cache={}
    def convert_tokens_to_ids(self,text):return self.tokenizer.convert_tokens_to_ids(text)
    def __call__(self,text,**kwargs):
        if text.startswith('level_') or text.startswith('Rate '):
            if text not in self.cache:self.cache[text]=self.tokenizer(text,**kwargs)
            return self.cache[text]
        return self.tokenizer(text,**kwargs)


def initialize():
    global TOKENIZER
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    TOKENIZER=CachedTokenizer(AutoTokenizer.from_pretrained(BASE,local_files_only=True))


def chunk(rows):return [encode_score(row,TOKENIZER) for row in rows]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    data=args.data.resolve();manifest=json.loads((data/'manifest.json').read_text())
    out=data/'cache';out.mkdir(exist_ok=False)
    report={'status':'encoding','data_manifest_sha256':sha(data/'manifest.json'),'splits':{},
            'code_sha256':sha(Path(__file__)),'truncated':0,'candidate_sampling':False,'rounded_labels':0}
    atomic(out/'audit.json',report)
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for split,desc in manifest['splits'].items():
            assert sha(desc['path'])==desc['sha256']
            rows=[json.loads(line) for line in open(desc['path'])];encs=[]
            jobs=[rows[i:i+128] for i in range(0,len(rows),128)]
            for batch in pool.map(chunk,jobs):
                encs.extend(batch)
                atomic(out/'progress.json',{'split':split,'rows':len(encs),'total':len(rows)})
            lengths=[len(e['ids']) for e in encs]
            assert len(encs)==len(rows) and max(lengths)<=262144
            assert all(e['labels']==[None] and len(e['opt_idx'][0])==len(r['score_support']) for r,e in zip(rows,encs))
            path=out/f'{split}.kev.pkl'
            with path.open('xb') as handle:pickle.dump(encs,handle,protocol=5)
            entry={'path':str(path),'sha256':sha(path),'max_tokens':max(lengths),
                'p99_tokens':sorted(lengths)[int(.99*(len(lengths)-1))],'forward_tokens':sum(lengths),
                'longest_indices':sorted(range(len(rows)),key=lambda i:lengths[i],reverse=True)[:10]}
            report['splits'][split]={'rows':len(rows),'views_path':desc['path'],'views_sha256':desc['sha256'],
                'native_evaluation':split!='train','complete_output_grid':True,'arms':{'kev':entry},
                'max_json_response_tokens':0}
            atomic(out/'audit.json',report)
            print(split,len(rows),'max',max(lengths),'p99',entry['p99_tokens'],flush=True)
    report['status']='passed';atomic(out/'audit.json',report)


if __name__=='__main__':main()
