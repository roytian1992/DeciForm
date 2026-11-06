"""Paired full-native timing and exact-output check of a frozen route cache."""
import contextlib
import datetime
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback

import torch

from benchmark_inference import ROOT as TIMING_ROOT, load_model, request, summarize
from common import PROJECT, atomic, sha, support
from inference_route_cache import FrozenRouteCache


def main():
    root = PROJECT/'runs/formal_inference_route_cache_20261106_v1'
    root.mkdir()
    cfg = json.loads((TIMING_ROOT/'config.json').read_text())
    rows = [json.loads(line) for line in Path(cfg['sample_path']).read_text().splitlines()]
    atomic(root/'manifest.json', {'parent_config':str(TIMING_ROOT/'config.json'),
           'parent_config_sha256':sha(TIMING_ROOT/'config.json'), 'created_at':datetime.datetime.now().astimezone().isoformat(),
           'script_sha256':sha(__file__), 'optimization_sha256':sha(Path(__file__).with_name('inference_route_cache.py')),
           'candidate_mode':'complete_native', 'rows':len(rows), 'repeats':3,
           'blocks':['original','cached','cached','original','original','cached'],
           'scope':'Same loaded frozen checkpoint. Route constants only. No LoRA factor folding or matrix reassociation.'})
    try:
        torch.set_num_threads(cfg['cpu_threads'])
        os.sched_setaffinity(0,cfg['cpu_affinity'])
        torch.cuda.set_device(0)
        assert sha(cfg['sample_path']) == cfg['sample_sha256']
        assert sha(cfg['checkpoints']['d2']['path']) == cfg['checkpoints']['d2']['sha256']
        atomic(root/'status.json', {'status':'loading'})
        model,tok=load_model('d2',cfg)
        cache=FrozenRouteCache(model)
        reference={}
        records={'original':[],'cached':[]}
        blocks=['original','cached','cached','original','original','cached']
        with torch.inference_mode(), open(root/'requests.jsonl','x',buffering=1) as handle:
            for block,variant in enumerate(blocks):
                repeat=block//2
                context=cache if variant=='cached' else contextlib.nullcontext()
                with context:
                    # Each distinct task/K shape is warmed for both paths.
                    for row in rows:
                        if row['id'] in cfg['warmup_ids']:
                            request(model,tok,'d2',row,cfg['max_new_tokens'])
                    order=list(range(len(rows)))
                    random.Random(cfg['seed']+repeat).shuffle(order)
                    for number,i in enumerate(order):
                        row=rows[i]
                        torch.cuda.synchronize()
                        p,tokens,timing=request(model,tok,'d2',row,cfg['max_new_tokens'])
                        if row['id'] not in reference:
                            assert variant=='original'
                            reference[row['id']]=p
                        base=reference[row['id']]
                        delta=max(abs(a-b) for a,b in zip(p['probabilities'],base['probabilities']))
                        exact=p==base
                        record={'id':row['id'],'task':row['task'],'source':row['source_family'],'k':len(support(row)),
                                'repeat':repeat,'variant':variant,'block':block,'order':number,
                                'prediction':p,'exact_output':exact,'max_probability_delta':delta,**tokens,**timing}
                        records[variant].append(record)
                        handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                        if not exact:
                            raise AssertionError(f'Output changed for {row["id"]}: delta={delta}')
                        if (number+1)%25==0:
                            atomic(root/'status.json',{'status':'measuring','block':block,'variant':variant,
                                   'completed_in_block':number+1,'total_blocks':6,'rows_per_block':len(rows)})
                            print(variant,block,number+1,flush=True)
        summary={v:summarize(rs) for v,rs in records.items()}
        atomic(root/'summary.json',{'status':'complete','unique_rows':len(rows),'requests_per_variant':{v:len(rs) for v,rs in records.items()},
                'all_outputs_exact':all(r['exact_output'] for rs in records.values() for r in rs),
                'max_probability_delta':max(r['max_probability_delta'] for rs in records.values() for r in rs),
                'metrics':summary,'speedup':summary['original']['all']['mean_ms']/summary['cached']['all']['mean_ms'],
                'cache_context_hits':cache.hits,'cache_context_misses':cache.misses,'cache_contexts':len(cache.contexts)})
        atomic(root/'status.json',{'status':'complete'})
    except Exception:
        atomic(root/'status.json',{'status':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
