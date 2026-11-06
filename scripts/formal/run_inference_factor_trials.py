"""Minimal, paired D2 deployment trials on two user-allocated GPUs."""
import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

from benchmark_inference import ROOT as TIMING_ROOT, load_model, request, summarize, command_output
from common import PROJECT, atomic, sha, support, target
from inference_route_cache import FrozenRouteCache
from inference_factor_trials import PrecastFactors, FoldedFactors

ROOT = PROJECT/'runs/formal_inference_factor_trials_20261106_v1'


def model_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if 'lora_' in name or 'task_adapter.' in name or '.head.' in name:
            h.update(name.encode());h.update(str(p.dtype).encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def quality(rows, predictions):
    out = {}
    for task in ('choice','noul','score'):
        selected=[r for r in rows if r['task']==task]
        if task=='score':
            value=statistics.mean(abs(support(r)[predictions[r['id']]['prediction_index']]-r['label']) for r in selected)
            out[task]={'rows':len(selected),'mae':value}
        else:
            value=statistics.mean(predictions[r['id']]['prediction_index']==target(r) for r in selected)
            out[task]={'rows':len(selected),'accuracy':value}
    return out


def main(arm, root):
    out=root/arm
    out.mkdir(parents=True,exist_ok=False)
    status=lambda **kw:atomic(out/'status.json',{'at':datetime.datetime.now().astimezone().isoformat(),**kw})
    try:
        cfg=json.loads((TIMING_ROOT/'config.json').read_text())
        assert sha(cfg['sample_path'])==cfg['sample_sha256']
        assert sha(cfg['checkpoints']['d2']['path'])==cfg['checkpoints']['d2']['sha256']
        rows=[json.loads(x) for x in Path(cfg['sample_path']).read_text().splitlines()]
        affinity=sorted(os.sched_getaffinity(0))
        chosen=affinity[-8:-4] if arm=='fold' else affinity[-4:]
        os.sched_setaffinity(0,chosen)
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        variants=['cached','fold_a','fold_b'] if arm=='fold' else ['cached','precast']
        manifest={'at':datetime.datetime.now().astimezone().isoformat(),'arm':arm,'variants':variants,
            'python':sys.executable,'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),
            'gpu':command_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv']),
            'torch':torch.__version__,'cpu_affinity':chosen,'cpu_threads':4,'parent_config':str(TIMING_ROOT/'config.json'),
            'parent_config_sha256':sha(TIMING_ROOT/'config.json'),'sample_sha256':cfg['sample_sha256'],
            'checkpoint':cfg['checkpoints']['d2'],'rows':len(rows),'repeats':3,
            'full_native':True,'batch_size':1,'no_training':True,
            'protocol':'one full original pass for same-device reference; rotate variant order across three repeats; full original input; same warmed task/K contexts; end-to-end raw input to CPU answer',
            'cold_setup_excluded':True,'code':{str(Path(__file__)):sha(__file__),
               str(Path(__file__).with_name('inference_factor_trials.py')):sha(Path(__file__).with_name('inference_factor_trials.py')),
               str(Path(__file__).with_name('inference_route_cache.py')):sha(Path(__file__).with_name('inference_route_cache.py'))}}
        atomic(out/'manifest.json',manifest)
        status(status='loading')
        model,tok=load_model('d2',cfg)
        digest=model_digest(model)
        started=time.perf_counter()
        cache=FrozenRouteCache(model)
        variants_impl={'cached':None}
        if arm=='fold':
            variants_impl.update(fold_a=FoldedFactors(model,cache,'a'),fold_b=FoldedFactors(model,cache,'b'))
        else:
            variants_impl['precast']=PrecastFactors(model)
        torch.cuda.synchronize()
        setup_seconds=time.perf_counter()-started
        references={}
        records={v:[] for v in ['original',*variants]}
        schedule=[(-1,'original')]+[(repeat,v) for repeat in range(3)
                      for v in variants[repeat%len(variants):]+variants[:repeat%len(variants)]]
        with torch.inference_mode(),open(out/'requests.jsonl','x',buffering=1) as handle:
            for block,(repeat,variant) in enumerate(schedule):
                status(status='warmup',block=block,variant=variant,total_blocks=len(schedule))
                with contextlib.ExitStack() as stack:
                    if variant!='original':
                        stack.enter_context(cache)
                        if variants_impl[variant] is not None:
                            stack.enter_context(variants_impl[variant])
                    for row in rows:
                        if row['id'] in cfg['warmup_ids']:
                            request(model,tok,'d2',row,cfg['max_new_tokens'])
                    torch.cuda.reset_peak_memory_stats()
                    order=list(range(len(rows)))
                    random.Random(cfg['seed']+max(repeat,0)).shuffle(order)
                    for number,i in enumerate(order):
                        row=rows[i]
                        torch.cuda.synchronize()
                        pred,tokens,timing=request(model,tok,'d2',row,cfg['max_new_tokens'])
                        if variant=='original':references[row['id']]=pred
                        base=references[row['id']]
                        probs,expected=pred['probabilities'],base['probabilities']
                        assert len(probs)==len(expected)==len(support(row))
                        exact=pred==base
                        delta=max(abs(a-b) for a,b in zip(probs,expected))
                        flip=pred['prediction_index']!=base['prediction_index']
                        record={'id':row['id'],'task':row['task'],'source':row['source_family'],
                            'k':len(probs),'variant':variant,'repeat':repeat,'block':block,'order':number,
                            'prediction':pred,'exact_output':exact,'max_probability_delta':delta,
                            'decision_changed':flip,'probability_l1':sum(abs(a-b) for a,b in zip(probs,expected)),**tokens,**timing}
                        records[variant].append(record)
                        handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                        # Even nominally exact variants are fully measured if parity fails;
                        # all differences are reported, never hidden or called lossless.
                        if (number+1)%25==0:
                            status(status='measuring',block=block,variant=variant,completed=number+1,total=150,total_blocks=len(schedule))
                            print(arm,variant,repeat,number+1,flush=True)
                atomic(out/f'block_{block}.json',{'variant':variant,'repeat':repeat,
                       'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        after=model_digest(model)
        assert after==digest,'Inference context failed to restore original weights'
        result={}
        for variant,rs in records.items():
            predictions={r['id']:r['prediction'] for r in rs}
            result[variant]={'requests':len(rs),'unique_rows':len(predictions),'timing':summarize(rs),
                'exact_requests':sum(r['exact_output'] for r in rs),
                'max_probability_delta':max(r['max_probability_delta'] for r in rs),
                'mean_probability_l1':statistics.mean(r['probability_l1'] for r in rs),
                'decision_changes':sum(r['decision_changed'] for r in rs),
                'changed_unique_ids':sorted({r['id'] for r in rs if r['decision_changed']}),
                'sample_quality_diagnostic_only':quality(rows,predictions),
                'by_task_parity':{t:{'changed_requests':sum(r['decision_changed'] for r in rs if r['task']==t),
                       'max_probability_delta':max(r['max_probability_delta'] for r in rs if r['task']==t)} for t in ('choice','noul','score')}}
        summary={'status':'complete','arm':arm,'setup_seconds':setup_seconds,'weights_restored_exact':after==digest,
                 'original_weight_digest':digest,'variants':result,
                 'warning':'150-row numerical/latency screen, not a full quality evaluation; no accuracy generalization or lossless claim from unmeasured inputs.'}
        atomic(out/'summary.json',summary)
        status(status='complete')
    except Exception:
        status(status='failed',traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['fold','precast'],required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    main(args.arm,args.root)
