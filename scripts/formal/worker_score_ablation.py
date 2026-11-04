"""D2 architectural ablations with frozen native decision rules.

Evaluation-only amendment (2026-11-02): external-source panels deferred.
All arms share record order, global batches, one configured OneCycle schedule,
and a sole epoch-1 endpoint (no metric-based checkpoint selection).
Choice/noul report accuracy; score reports native MAE. No candidate resampling.
"""
import argparse,collections,datetime,hashlib,json,math,os,pickle,random,shutil,sys,time,traceback
from pathlib import Path
from common import *
from models_score_ablation import *
from score_distribution import PROTOCOL, score_rule_report
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors,_unflatten_dense_tensors

def digest_tensors(items):
 h=hashlib.sha256()
 for name,p in items:
  h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def batches(indices,encs,arm,budget,max_rows=16):
 indices=sorted(indices,key=lambda i:cost(encs[i],arm)[1],reverse=True)
 cur=[];n=0;longest=0
 for i in indices:
  nr,l=cost(encs[i],arm)
  if cur and ((n+nr)*max(longest,l)>budget or len(cur)>=max_rows):yield cur;cur=[];n=0;longest=0
  cur.append(i);n+=nr;longest=max(longest,l)
 if cur:yield cur

def synchronize(params,world):
 if world==1:return
 present=torch.tensor([p.grad is not None for p in params],device='cuda',dtype=torch.int32);dist.all_reduce(present,op=dist.ReduceOp.MAX)
 active=[]
 for yes,p in zip(present.tolist(),params):
  if yes:
   if p.grad is None:p.grad=torch.zeros_like(p)
   active.append(p.grad)
 for start in range(0,len(active),32):
  chunk=active[start:start+32];flat=_flatten_dense_tensors(chunk);dist.all_reduce(flat,op=dist.ReduceOp.SUM)
  for grad,reduced in zip(chunk,_unflatten_dense_tensors(flat,chunk)):grad.copy_(reduced)

def trainable_state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def restore(model,state):
 named={n:p for n,p in model.named_parameters() if p.requires_grad}
 if set(named)!=set(state):raise ValueError('Trainable inventory mismatch')
 with torch.no_grad():
  for n,p in named.items():p.copy_(state[n])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--mode',choices=['preflight','train'],default='train');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 cfg=json.loads(a.config.read_text());arm=cfg['arm'];out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
 assert cfg['epochs']==1 and cfg['train_rows']==200000 and cfg['global_batch']==64
 assert cfg['seed']==SEED and cfg['checkpoint_policy']=='half'
 assert cfg['ablation']==ABLATIONS[cfg['method_variant']]
 assert cfg['distribution_objective']['kind']=='ordinal_main'
 assert cfg['task_losses']['choice']=='categorical CE'
 assert cfg['task_losses']['noul']=='Bernoulli NLL'
 assert cfg['evaluation_protocol']==PROTOCOL and cfg['score_decision_rule']=='median'
 assert cfg['evaluation_splits']==['dev','test_in_domain']
 assert set(cfg['ablation']).issubset({'task_routing','quadratic','geometry','auxiliary_losses'})
 rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0));torch.cuda.set_device(local)
 if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 torch.manual_seed(SEED);random.seed(SEED)
 if rank:sys.stdout=open(os.devnull,'w')
 started=time.time()
 def progress(phase,**kwargs):
  if rank==0:atomic(out/('preflight.progress.json' if a.mode=='preflight' else 'progress.json'),{'phase':phase,'arm':cfg.get('label',arm),'updated_unix':time.time(),'world_size':world,**kwargs})
 progress('loading_model')
 audit_path=Path(cfg['data'])/'cache/audit.json';audit=json.loads(audit_path.read_text())
 assert audit['status']=='passed' and sha(audit_path)==cfg['cache_audit_sha256']
 data_manifest=Path(cfg['data'])/'manifest.json'
 assert sha(data_manifest)==audit['data_manifest_sha256']
 assert cfg['base']==str(BASE.resolve()) and cfg['initialization']=='fresh_base_only'
 model=FormalModel(arm,BASE,ablation=cfg['ablation'],distribution_objective=cfg['distribution_objective']);groups=model.groups();params=[p for p in model.parameters() if p.requires_grad]
 policy_inventory=model.set_checkpoint_policy(cfg.get('checkpoint_policy','all'))
 if rank==0:
  atomic(out/('preflight.checkpoint_policy.json' if a.mode=='preflight' else 'checkpoint_policy.json'),policy_inventory)
  atomic(out/('preflight.ablation.json' if a.mode=='preflight' else 'ablation.json'),model.ablation_inventory)
 assert len({id(p) for p in params})==sum(len(g['params']) for g in groups)
 # Require pristine zero-B LoRA at every initial model construction, including resume.
 assert all(torch.count_nonzero(p).item()==0 for n,p in model.lm.named_parameters() if 'lora_B' in n)
 initial={'mode':'fresh_base_only','base':str(BASE.resolve()),'base_config_sha256':sha(BASE/'config.json'),'inherited_adapter':None,'inherited_head':None,'trainable_parameters':sum(p.numel() for p in params),'lora_sha256':digest_tensors((n,p) for n,p in model.lm.named_parameters() if 'lora_' in n),'arm':arm,'head_sha256':digest_tensors(model.net.head.named_parameters())}
 if arm in ('kev','tcdl'):initial['pointer_sha256']=digest_tensors((n,p) for n,p in model.net.head.named_parameters() if n.startswith(('q.','k.')))
 allinit=[None]*world
 if world>1:dist.all_gather_object(allinit,initial)
 else:allinit=[initial]
 assert all(x==initial for x in allinit),'Ranks initialize differently'
 if rank==0:
  atomic(out/('preflight.initialization.json' if a.mode=='preflight' else 'initialization.json'),initial)
  atomic(out/('preflight.loss_objective.json' if a.mode=='preflight' else 'loss_objective.json'), model.distribution_objective.inventory())
  atomic(out/'optimizer_groups.json',[{'name':g['name'],'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in groups])
 cache={}
 def population(split):
  if split not in cache:
   desc=audit['splits'][split];keyarm='kev' if arm=='tcdl' else arm;entry=desc['arms'][keyarm]
   assert sha(desc['views_path'])==desc['views_sha256'] and sha(entry['path'])==entry['sha256']
   rs=[json.loads(l) for l in open(desc['views_path'])]
   with open(entry['path'],'rb') as f:es=pickle.load(f)
   assert len(rs)==len(es)==desc['rows']
   if split!='train':assert all(r['candidate_mode']=='complete' for r in rs)
   cache[split]=(rs,es)
  return cache[split]
 rs,es=population('train')
 subset_path=cfg.get('train_subset_indices')
 if subset_path:
  subset=json.loads(Path(subset_path).read_text());assert subset['source_rows']==len(rs)
  ids=subset['indices'];assert len(ids)==cfg['train_rows'] and ids==sorted(set(ids))
  rs=[rs[i] for i in ids];es=[es[i] for i in ids]
 n=len(rs);global_batch=cfg['global_batch'];epochs=cfg['epochs'];per_epoch=math.ceil(n/global_batch);total_steps=per_epoch*epochs
 assert n==200000 and total_steps==3125
 optimizer=torch.optim.AdamW(groups,weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[g['lr'] for g in groups],total_steps=total_steps,pct_start=float(cfg.get('warmup_fraction',.1)))
 max_new_tokens=max(d['max_json_response_tokens'] for d in audit['splits'].values())+8
 if a.mode=='preflight':
  model.train();checks=[]
  model.collect_loss_diagnostics=True
  # A normal mixed step checks all task pathways and distributed normalization.
  chosen=[]
  for task in TASKS:chosen.extend([i for i,r in enumerate(rs) if r['task']==task][:world*2])
  for ids in batches(chosen[rank::world],es,arm,cfg['token_budget']):
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.losses([rs[i] for i in ids],[es[i] for i in ids]).sum()/len(chosen)
   assert torch.isfinite(loss);loss.backward();model.clear()
  synchronize(params,world)
  assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
  checks.append('mixed-task backward and distributed gradient sum')
  diagnostic=model.take_loss_diagnostics()
  if world>1:dist.all_reduce(diagnostic,op=dist.ReduceOp.SUM)
  assert int(diagnostic[:,0].sum())==len(chosen) and torch.isfinite(diagnostic).all()
  model.collect_loss_diagnostics=False
  checks.append('distributed detached loss/logit-gradient diagnostics')
  optimizer.zero_grad(set_to_none=True)
  # Full longest input backward, per task and total, on separate GPUs.
  longest=[]
  for task in TASKS:longest.append(max((i for i,r in enumerate(rs) if r['task']==task),key=lambda i:cost(es[i],arm)[0]*cost(es[i],arm)[1]))
  for i in longest:
   progress('preflight_longest_backward',id=rs[i]['id'],tokens=cost(es[i],arm)[1])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.losses([rs[i]],[es[i]]).sum()
   assert torch.isfinite(loss);loss.backward();model.clear();optimizer.zero_grad(set_to_none=True)
  checks.append('longest complete training request per task backward')
  model.eval();vr,ve=population('dev');vi=[next(i for i,r in enumerate(vr) if r['task']==t) for t in TASKS]
  for i in vi:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model.predict([vr[i]],[ve[i]],max_new_tokens)
   if arm!='json_sft':assert pred[0]['valid']
  checks.append('native support inference for all task types')
  for split in ('dev','test_in_domain'):
   vr,ve=population(split)
   i=max(range(len(vr)),key=lambda i:cost(ve[i],arm)[0]*cost(ve[i],arm)[1])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model.predict([vr[i]],[ve[i]],max_new_tokens)
  checks.append('largest complete evaluation request in every panel')
  # Keep the frozen v6 half-layer policy; validate the two-GPU packing
  # without selecting a different speed policy for either ablation.
  model.train()
  dense=min(range(n),key=lambda i:abs(cost(es[i],arm)[1]-cfg['token_budget']//16))
  first_order=list(range(n));random.Random(SEED).shuffle(first_order)
  for case,indices in [('dense',[dense]*global_batch),('first_global_batch',first_order[:global_batch])]:
   progress('preflight_fixed_policy',case=case,checkpoint_policy=cfg['checkpoint_policy'])
   for ids in batches(indices[rank::world],es,arm,cfg['token_budget']):
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.losses([rs[i] for i in ids],[es[i] for i in ids]).sum()/len(indices)
    assert torch.isfinite(loss);loss.backward();model.clear()
   synchronize(params,world)
   assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
   optimizer.zero_grad(set_to_none=True)
  checks.append('fixed half-layer checkpoint policy; dense/first full global batches; zero optimizer updates')
  peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device='cuda')
  if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
  assert float(peak)<=cfg['preflight_memory_limit_gib'],'Fixed-policy preflight exceeds memory budget'
  if rank==0:atomic(out/'preflight.json',{'status':'passed','checks':checks,'checkpoint_policy':policy_inventory,'ablation':model.ablation_inventory,'distribution_objective':model.distribution_objective.inventory(),'initialization':initial,'peak_memory_gib':float(peak),'optimizer_updates':0,'evaluation_protocol':PROTOCOL,'seconds':time.time()-started})
  if world>1:dist.barrier();dist.destroy_process_group()
  return
 checkpoint=out/'resume.pt';step=0;start_epoch=0;cursor=0;best=-1.;best_epoch=None;train_seconds=0.;evaluation_seconds=0.
 if a.resume:
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert state['config_sha256']==sha(a.config) and state['world_size']==world
  restore(model,state['trainable']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
  step,start_epoch,cursor=state['step'],state['epoch'],state['cursor'];best=state['best'];best_epoch=state['best_epoch'];train_seconds=state['train_seconds'];evaluation_seconds=state.get('evaluation_seconds',0.)
  assert scheduler.total_steps==total_steps and scheduler.last_epoch==step,'Resume schedule does not match configured training horizon'
  torch.set_rng_state(state['rng'][rank]['cpu']);torch.cuda.set_rng_state(state['rng'][rank]['cuda'])
 else:
  if checkpoint.exists() or (out/'epoch_1.pt').exists():raise FileExistsError('Use --resume for an existing training run')
  torch.manual_seed(SEED+rank)
 def save(ep,cur,epoch_file=None):
  local_rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()};rngs=[None]*world
  if world>1:dist.all_gather_object(rngs,local_rng)
  else:rngs=[local_rng]
  if rank==0:
   state={'trainable':trainable_state(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rngs,'world_size':world,'config_sha256':sha(a.config),'step':step,'epoch':ep,'cursor':cur,'best':best,'best_epoch':best_epoch,'train_seconds':train_seconds,'evaluation_seconds':evaluation_seconds}
   tmp=checkpoint.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(checkpoint)
   if epoch_file:torch.save({'trainable':state['trainable'],'config':cfg,'step':step,'epoch':ep},out/epoch_file)
  if world>1:dist.barrier()
 def evaluate(split,label):
  nonlocal evaluation_seconds
  t0=time.time();vr,ve=population(split);model.eval();result=[];my_indices=list(range(rank,len(vr),world));progress('evaluating',split=split,checkpoint=label,completed=0,total=len(vr))
  tmpdir=out/'evaluations';tmpdir.mkdir(exist_ok=True);journal=tmpdir/f'{label}.{split}.rank{rank}.jsonl'
  with journal.open('w',buffering=1) as f:
   for indices in batches(my_indices,ve,arm,cfg['eval_token_budget'],max_rows=8):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model.predict([vr[i] for i in indices],[ve[i] for i in indices],max_new_tokens)
    for i,prediction in zip(indices,pred):
     row=prediction_row(vr[i],prediction);result.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if len(result)%100<len(indices):progress('evaluating',split=split,checkpoint=label,local_completed=len(result),local_total=len(my_indices),total=len(vr))
  parts=[None]*world
  if world>1:dist.all_gather_object(parts,result)
  else:parts=[result]
  allrows=[r for part in parts for r in part];assert len(allrows)==len(vr) and len({r['id'] for r in allrows})==len(vr)
  report=metrics(allrows);report['selection_macro_task_accuracy']=sum(report['task/'+t]['accuracy'] for t in TASKS)/3 if split=='dev' else None
  elapsed=time.time()-t0;evaluation_seconds+=elapsed
  report.update(seconds=elapsed,checkpoint=label,split=split,native_full_candidates=True,rows=len(allrows),evaluation_protocol=PROTOCOL,score_decision_rule='median',peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
  if rank==0:
   atomic(tmpdir/f'{label}.{split}.json',report)
   decisions_report=score_rule_report(allrows,split)
   assert decisions_report['rows']==report['task/score']['rows']
   assert abs(decisions_report['primary_mae']-report['task/score']['mae'])<1e-10
   decisions_report['checkpoint']=label
   decisions_report['prediction_files_sha256']={str(tmpdir/f'{label}.{split}.rank{r}.jsonl'):sha(tmpdir/f'{label}.{split}.rank{r}.jsonl') for r in range(world)}
   atomic(tmpdir/f'{label}.{split}.score_rules.json',decisions_report)
  return report
 model.train()
 step_times=[]
 for epoch in range(start_epoch,epochs):
  order=list(range(n));random.Random(SEED+epoch).shuffle(order)
  for offset in range(cursor if epoch==start_epoch else 0,n,global_batch):
   t0=time.time();indices=order[offset:offset+global_batch];mine=indices[rank::world];optimizer.zero_grad(set_to_none=True);loss_value=0.
   model.collect_loss_diagnostics = step == 0 or (step+1)%cfg['loss_diagnostics_every']==0
   task_counts=collections.Counter(rs[i]['task'] for i in indices)
   balance_weights=None
   if cfg.get('task_balanced'):
    balance_weights={t:len(indices)/(len(TASKS)*task_counts[t]) for t in TASKS if task_counts[t]}
   for ids in batches(mine,es,arm,cfg['token_budget']):
    with torch.autocast('cuda',dtype=torch.bfloat16):
     losses=model.losses([rs[i] for i in ids],[es[i] for i in ids])
     if balance_weights:
      weights=torch.tensor([balance_weights[rs[i]['task']] for i in ids],device=losses.device,dtype=losses.dtype)
      loss=(losses*weights).sum()/len(indices)
     else:
      loss=losses.sum()/len(indices)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();loss_value+=loss.item();model.clear()
   synchronize(params,world);grad_norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();scheduler.step();step+=1
   if model.collect_loss_diagnostics:
    diagnostic=model.take_loss_diagnostics()
    if world>1:dist.all_reduce(diagnostic,op=dist.ReduceOp.SUM)
    assert int(diagnostic[:,0].sum())==len(indices)
    if rank==0:
     fields=('data_loss','reference_ce','logit_gradient_l2','reference_ce_logit_gradient_l2','true_probability','candidate_count')
     detail={t:{'rows':int(row[0]),**{name:float(value/row[0]) for name,value in zip(fields,row[1:])}} for t,row in zip(TASKS,diagnostic.cpu().tolist()) if row[0]}
     with (out/'loss_diagnostics.jsonl').open('a',buffering=1) as journal:
      journal.write(json.dumps({'step':step,'tasks':detail,'scope':'per-row output-logit gradients before global-batch normalization; not backbone gradients; excludes structural penalty'})+'\n')
   model.collect_loss_diagnostics=False
   elapsed=time.time()-t0;train_seconds+=elapsed;step_times.append(elapsed)
   if step%10==0 or step==1:
    lv=torch.tensor(loss_value,device='cuda')
    if world>1:dist.all_reduce(lv)
    recent=sum(step_times[-100:])/len(step_times[-100:]);status={'epoch':epoch+1,'cursor':offset+len(indices),'step':step,'total_steps':total_steps,'visits':epoch*n+offset+len(indices),'loss':lv.item(),'gradient_norm':float(grad_norm),'recent_step_seconds':recent,'remaining_training_hours':recent*(total_steps-step)/3600,'train_seconds':train_seconds,'evaluation_seconds':evaluation_seconds,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    progress('training',**status)
    if rank==0:print(json.dumps(status),flush=True)
   if step%cfg['save_every']==0:save(epoch,offset+len(indices))
  # Store both the epoch model and resume position before dev inference.
  save(epoch,n,f'epoch_{epoch+1}.pt')
  report=evaluate('dev',f'epoch_{epoch+1}');score=report['selection_macro_task_accuracy']
  if score>best:
   best=score;best_epoch=epoch+1
   if rank==0:atomic(out/'selected_checkpoint.json',{'epoch':best_epoch,'path':str(out/f'epoch_{best_epoch}.pt'),'dev_macro_task_accuracy':best,'criterion':'sole frozen epoch-1 endpoint; no metric selection','primary_metrics':{'choice':'accuracy','noul':'accuracy','score':'mae'},'dev_score_mae':report['task/score']['mae']})
  cursor=0;save(epoch+1,0);model.train()
 assert step==total_steps
 selected=torch.load(out/f'epoch_{best_epoch}.pt',map_location='cpu',weights_only=False);restore(model,selected['trainable'])
 for split in ('test_in_domain',):evaluate(split,f'epoch_{best_epoch}')
 progress('complete',step=step,total_steps=total_steps,visits=epochs*n,selected_epoch=best_epoch,dev_macro_task_accuracy=best,train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,total_session_seconds=time.time()-started)
 if rank==0:atomic(out/'complete.json',{'selected_epoch':best_epoch,'dev_macro_task_accuracy':best,'dev_score_mae':report['task/score']['mae'],'primary_metrics':{'choice':'accuracy','noul':'accuracy','score':'mae'},'train_seconds':train_seconds,'evaluation_seconds':evaluation_seconds,'steps':step,'sample_visits':epochs*n,'test_selection':False,'evaluation_protocol':PROTOCOL,'score_decision_rule':'median','evaluated_splits':['dev','test_in_domain'],'deferred_splits':['test_source_holdout'],'deferral_reason':'User requested external-source preparation and evaluation only after the other experiments'})
 if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  try:
   ix=sys.argv.index('--config');cfg=json.loads(Path(sys.argv[ix+1]).read_text());out=Path(cfg['output']);rank=int(os.getenv('RANK',0))
   atomic(out/f'failure.rank{rank}.json',{'time':time.time(),'error':str(exc),'traceback':traceback.format_exc(),'mode':sys.argv})
  except Exception:pass
  raise
