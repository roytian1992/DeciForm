import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('FLA_DISABLE_TENSOR_CACHE','1')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import PROJECT,atomic
from zero_shot_logits import encode,score_batch,dense_scores,values_for

def run(out):
    torch.set_num_threads(4)
    torch.manual_seed(20261101)
    model_path=PROJECT.parent/'models/Qwen3.5-9B'
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    model.requires_grad_(False).eval()
    manifest=json.loads((PROJECT/'runs/formal_zero_shot_instruct_20261105_v3/experiment_manifest.json').read_text())
    rows=[json.loads(l) for l in open(manifest['panels']['main']['views_path'])]
    samples=[]
    for task in ('choice','noul','score'):
        row=next(r for r in rows if r['task']==task and (task!='choice' or len(values_for(r))==77))
        samples.append((row,'main'))
    for panel in ('asap','continuous'):
        rs=[json.loads(l) for l in open(manifest['panels'][panel]['views_path'])]
        ks=sorted({len(values_for(r)) for r in rs})
        for k in ks:
            row=next(r for r in rs if len(values_for(r))==k)
            samples.append((row,panel))
    checks=[]
    for row,panel in samples:
        e=encode(row,panel,'original',tok)
        start=time.time();scores=score_batch(model,[e],tok.pad_token_id)[0]
        indexes=sorted({0,len(e['tails'])//2,len(e['tails'])-1})
        dense=dense_scores(model,e,indexes)
        diffs=[abs(scores['sequence_likelihood'][i]-d) for i,d in zip(indexes,dense)]
        probs=torch.tensor(scores['sequence_likelihood']).softmax(0)
        rev={**e,'tails':e['tails'][::-1]}
        reversed_scores=score_batch(model,[rev],tok.pad_token_id)[0]
        delta=float((probs-torch.tensor(reversed_scores['sequence_likelihood'][::-1]).softmax(0)).abs().max())
        check={'panel':panel,'id':row['id'],'k':len(e['tails']),'tokens':len(e['ids']),
               'teacher_forced_logp_max_difference':max(diffs),'suffix_enumeration_reversal_probability_delta':delta,
               'seconds':time.time()-start,'selected_dense_scores':dense,'selected_cached_scores':[scores['sequence_likelihood'][i] for i in indexes]}
        print(json.dumps(check),flush=True);checks.append(check)
        assert max(diffs)<.25,check
        assert delta<1e-5,check
    # Mixed padded batch vs separate scoring, covering boolean and variable decimal lengths.
    es=[encode(r,p,'original',tok) for r,p in [samples[1],samples[-1]]]
    batched=score_batch(model,es,tok.pad_token_id)
    batch_diff=[]
    for e,b in zip(es,batched):
        a=score_batch(model,[e],tok.pad_token_id)[0]
        batch_diff.append(float((torch.tensor(a['sequence_likelihood']).softmax(0)-torch.tensor(b['sequence_likelihood']).softmax(0)).abs().max()))
    assert max(batch_diff)<.02,batch_diff
    atomic(out,{'status':'passed','checks':checks,'mixed_batch_probability_differences':batch_diff,
                'optimizer_updates':0,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'no_truncation':True,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                'numerical_note':'BF16 cached recurrent decoding versus fused full teacher forcing; logp tolerance .25, probability padded-batch tolerance .02. Enumeration reverse <1e-5.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
