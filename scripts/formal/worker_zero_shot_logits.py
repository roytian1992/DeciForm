"""Resumable, one-GPU frozen LM likelihood evaluation with full native inventories."""
import argparse,fcntl,json,os,time,traceback
from pathlib import Path
os.environ.setdefault('FLA_DISABLE_TENSOR_CACHE','1')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import PROJECT,atomic,sha
from zero_shot_logits import RULES,PANELS,encode,batches,score_batch,prediction,summarize
from zero_shot_protocol import views_for


def run(root,shard):
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    out=root/f'shard_{shard}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    for path,digest in manifest['code_sha256'].items():assert sha(PROJECT/path)==digest,path
    assert json.loads((root/'preflight.json').read_text())['status']=='passed'
    started=time.time()
    def progress(phase,**extra):
        atomic(out/'progress.json',{'phase':phase,'shard':shard,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'updated_unix':time.time(),'seconds':time.time()-started,**extra})
    progress('loading_model')
    torch.set_num_threads(4);torch.manual_seed(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    path=manifest['model']['path']
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
                                            attn_implementation='sdpa',device_map={'':0})
    model.requires_grad_(False).eval()
    assert not any(p.requires_grad for p in model.parameters())
    assert not any('lora_' in n or 'adapter' in n for n,_ in model.named_parameters())
    atomic(out/'model_loaded.json',{'path':path,'optimizer_updates':0,'trainable_parameters':0,
            'config_class':type(model).__name__,'files_sha256':manifest['model']['files_sha256']})
    for panel in PANELS:
        desc=manifest['panels'][panel]
        assert sha(desc['views_path'])==desc['views_sha256']
        rows=[json.loads(line) for line in open(desc['views_path'])]
        if desc.get('task'):rows=[r for r in rows if r['task']==desc['task']]
        if desc.get('sources'):rows=[r for r in rows if r['source_family'] in desc['sources']]
        assert len(rows)==desc['rows'] and len({r['id'] for r in rows})==len(rows)
        assert all(r['candidate_mode']=='complete' for r in rows)
        mine=rows[shard::manifest['shards']]
        journal=out/f'{panel}.predictions.jsonl'
        records=[json.loads(line) for line in journal.open()] if journal.exists() else []
        seen={(r['id'],r['view'],r['scoring_rule']) for r in records}
        expected={(r['id'],v,rule) for r in mine for v in views_for(panel) for rule in RULES}
        assert len(seen)==len(records) and seen.issubset(expected)
        progress('encoding',panel=panel,completed=len(seen),expected=len(expected))
        bundles=[]
        for ri,row in enumerate(mine):
            encs={v:encode(row,panel,v,tokenizer) for v in views_for(panel)}
            length=max(len(e['ids']) for e in encs.values())
            assert length+max(len(t) for e in encs.values() for t in e['tails'])<=manifest['max_context_tokens']
            bundles.append({'row':row,'encs':encs,'enc':encs['original'],'length':length})
            if (ri+1)%250==0:progress('encoding',panel=panel,encoded=ri+1,total=len(mine),completed=len(seen),expected=len(expected))
        fixed_chunks=list(batches(bundles,**manifest['batching']))
        atomic(out/f'{panel}.batch_groups.json',[[b['row']['id'] for b in chunk] for chunk in fixed_chunks])
        for view in views_for(panel):
            # Identical row groups in all order views; only candidate order changes.
            chunks=[[{**b,'enc':b['encs'][view]} for b in chunk] for chunk in fixed_chunks]
            if not (out/f'{panel}.{view}.preflight.json').exists():
                first=chunks[0]
                start=time.time()
                scores=score_batch(model,[b['enc'] for b in first],tokenizer.pad_token_id)
                assert len(scores)==len(first)
                atomic(out/f'{panel}.{view}.preflight.json',{'status':'passed','longest_tokens':max(b['length'] for b in bundles),
                    'longest_batch_rows':len(first),'seconds':time.time()-start,'no_truncation':True,
                    'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})
            with journal.open('a',buffering=1) as f:
                for bi,batch in enumerate(chunks):
                    if all((b['row']['id'],view,rule) in seen for b in batch for rule in RULES):continue
                    scores=score_batch(model,[b['enc'] for b in batch],tokenizer.pad_token_id)
                    for bundle,s in zip(batch,scores):
                        for rule in RULES:
                            key=(bundle['row']['id'],view,rule)
                            if key in seen:continue
                            r=prediction(bundle['row'],panel,view,bundle['enc'],s,rule)
                            f.write(json.dumps(r,ensure_ascii=False)+'\n');records.append(r);seen.add(key)
                    progress('evaluating',panel=panel,view=view,batch=bi+1,batches=len(chunks),
                        completed=len(seen),expected=len(expected),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        assert seen==expected
        atomic(out/f'{panel}.json',{'status':'complete','rows':len(mine),'records':len(records),
            'reports':{rule:summarize(panel,[r for r in records if r['scoring_rule']==rule]) for rule in RULES},
            'predictions_sha256':sha(journal)})
    atomic(out/'complete.json',{'status':'complete','optimizer_updates':0,'seconds':time.time()-started,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})
    progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,required=True)
    args=p.parse_args()
    try:run(args.root.resolve(),args.shard)
    except BaseException as exc:
        atomic(args.root/f'shard_{args.shard}'/'failure.json',{'updated_unix':time.time(),'error':str(exc),'traceback':traceback.format_exc()})
        raise
