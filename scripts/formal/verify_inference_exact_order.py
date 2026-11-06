"""Fail-fast exact-output and zero semantic flip gate for frozen D2 acceleration."""
import argparse
import collections
import contextlib
import json
import os
from pathlib import Path
import random
import traceback

import torch

from benchmark_inference import ROOT as TIMING_ROOT, load_model
from common import PROJECT, atomic, encode, sha, support
from independent_candidates import split_candidates, permute_candidates
from inference_route_cache import FrozenRouteCache
from inference_factor_trials import PrecastFactors


def main(rank):
    root=PROJECT/'runs/formal_inference_factor_trials_20261106_v1/exact_order'
    root.mkdir(exist_ok=True)
    out=root/f'rank{rank}'
    out.mkdir(exist_ok=False)
    status=lambda **kw:atomic(out/'status.json',kw)
    try:
        cfg=json.loads((TIMING_ROOT/'config.json').read_text())
        assert sha(cfg['sample_path'])==cfg['sample_sha256']
        assert sha(cfg['checkpoints']['d2']['path'])==cfg['checkpoints']['d2']['sha256']
        rows=[json.loads(x) for x in Path(cfg['sample_path']).read_text().splitlines()]
        rows=[r for r in rows if r['task'] in ('choice','noul')][rank::2]
        cpus=sorted(os.sched_getaffinity(0))[-8+rank*4:][:4]
        os.sched_setaffinity(0,cpus);torch.set_num_threads(4);torch.cuda.set_device(0)
        atomic(out/'manifest.json',{'script_sha256':sha(__file__),'rows':len(rows),'rank':rank,
              'sample_sha256':cfg['sample_sha256'],'checkpoint':cfg['checkpoints']['d2'],
              'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'orders':['original','reverse','shuffle'],
              'scope':'Choice/Noul only; complete native candidates; reference original model versus cache+precast; abort on any probability change or semantic decision flip'})
        status(status='loading')
        model,tok=load_model('d2',cfg)
        cache=FrozenRouteCache(model);precast=PrecastFactors(model)
        encs={r['id']:split_candidates(encode(r,'kev',tok)) for r in rows}
        references={};records=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),open(out/'requests.jsonl','x',buffering=1) as log:
            for variant in ('original','cache_precast'):
                with contextlib.ExitStack() as stack:
                    if variant=='cache_precast':
                        stack.enter_context(cache);stack.enter_context(precast)
                    for row in rows:
                        k=len(support(row))
                        identity=list(range(k));shuffle=identity.copy()
                        random.Random('20261106/'+row['id']).shuffle(shuffle)
                        for name,order in [('original',identity),('reverse',identity[::-1]),('shuffle',shuffle)]:
                            encoded=permute_candidates(encs[row['id']],order)
                            p=model.predict([row],[encoded],cfg['max_new_tokens'])[0]
                            if variant=='original':references[row['id'],name]=p
                            expected=references[row['id'],name]
                            exact=p==expected
                            delta=max(abs(a-b) for a,b in zip(p['probabilities'],expected['probabilities']))
                            changed=p['prediction_index']!=references[row['id'],'original']['prediction_index']
                            record={'id':row['id'],'task':row['task'],'k':k,'variant':variant,'order':name,
                                    'exact_output':exact,'max_probability_delta':delta,'semantic_flip':changed,'prediction':p}
                            log.write(json.dumps(record,ensure_ascii=False)+'\n');records.append(record)
                            if not exact or changed:
                                raise AssertionError(f'Hard parity gate failed: {row["id"]}/{variant}/{name}: exact={exact}, flip={changed}, delta={delta}')
                        if len(records)%30==0:status(status='verifying',records=len(records),total=len(rows)*6)
        atomic(out/'summary.json',{'status':'passed','unique_rows':len(rows),'records':len(records),
                'counts_by_task':dict(collections.Counter(r['task'] for r in rows)),
                'exact_output':all(r['exact_output'] for r in records),'max_probability_delta':max(r['max_probability_delta'] for r in records),
                'semantic_flips':sum(r['semantic_flip'] for r in records)})
        status(status='complete')
    except Exception:
        status(status='failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=[0,1],required=True)
    main(parser.parse_args().rank)
