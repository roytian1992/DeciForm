"""Fresh-base Laya on frozen Score panels; unchanged upstream RLCD and full text."""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from common import atomic, sha
from laya_adapter import canonical_probabilities, microbatches, order_for
from laya_score import encode as encode_parts, metrics, record as predict_record
from score_specialist import score_rule_report


def digest(items):
    result = hashlib.sha256()
    for name, tensor in items:
        result.update(name.encode())
        result.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def synchronize(params, world):
    if world == 1:
        return
    present = torch.tensor([p.grad is not None for p in params], device='cuda', dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    active = []
    for yes, p in zip(present.tolist(), params):
        if yes:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            active.append(p.grad)
    chunk, size = [], 0
    def reduce(tensors):
        flat = _flatten_dense_tensors(tensors)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for old, new in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            old.copy_(new)
    for grad in active:
        if chunk and size + grad.numel() > 32_000_000:
            reduce(chunk)
            chunk, size = [], 0
        chunk.append(grad)
        size += grad.numel()
    if chunk:
        reduce(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight','train'), default='train')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out, root = Path(cfg['output']), Path(cfg['root'])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, cfg['vendor'])
    from laya.common import build_model, collate_items
    from laya.train import rlcd_loss, save_checkpoint
    from transformers import AutoTokenizer

    rank, world = int(os.getenv('RANK',0)), int(os.getenv('WORLD_SIZE',1))
    local = int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    def gather(obj):
        values = [None]*world
        if world > 1:
            dist.all_gather_object(values, obj)
        else:
            values = [obj]
        return values
    def barrier():
        if world > 1:
            dist.barrier()
    def progress(phase, **kw):
        name = 'preflight.progress.json' if args.mode == 'preflight' else 'progress.json'
        atomic(out/f'heartbeat.rank{rank}.json', {'phase':phase, 'updated_unix':time.time(), **kw})
        if rank == 0:
            atomic(out/name, {'phase':phase, 'updated_unix':time.time(), 'layout':cfg['layout'], **kw})
    assert cfg['parent_checkpoint'] is None and cfg['epochs'] == 1
    assert cfg['global_batch'] == 64 and cfg['total_steps'] == math.ceil(cfg['train_rows']/64)
    assert cfg['world_size'] == world
    assert sha(cfg['cache_audit']) == cfg['cache_audit_sha256']
    audit = json.loads(Path(cfg['cache_audit']).read_text())
    assert audit['status'] == 'passed' and audit['truncated'] == audit['dropped'] == 0
    cache = {}
    def population(split):
        if split not in cache:
            desc = audit['splits'][split]
            assert sha(desc['views_path']) == desc['views_sha256']
            assert sha(desc['path']) == desc['sha256']
            rows = [json.loads(line) for line in open(desc['views_path'])]
            with open(desc['path'],'rb') as handle:
                parts = pickle.load(handle)
            assert len(rows) == len(parts) == desc['rows']
            if split != 'train':
                assert all(r['candidate_mode'] == 'complete' for r in rows)
            cache[split] = rows, parts
        return cache[split]

    progress('loading_pretrained_encoder')
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    tok = AutoTokenizer.from_pretrained(cfg['base'], local_files_only=True)
    model_cfg = {'encoder':cfg['base'], 'head_layers':2, 'act_costs':{'escalate':1},
                 'option_layout':cfg['layout'], 'max_len':cfg['max_len'], 'head_max_len':cfg['max_len']}
    model = build_model(model_cfg, pretrained=True)
    # The optional escalation head has no labels in our three-task protocol.
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.head_checkpointing = True
    model.to('cuda')
    initial = {'initialization':'pretrained_encoder_random_head', 'parent_checkpoint':None,
        'encoder_revision':cfg['encoder_revision'], 'base_weights_sha256':cfg['base_weights_sha256'],
        'encoder_sha256':digest(model.encoder.named_parameters()),
        'head_sha256':digest((n,p) for n,p in model.named_parameters() if not n.startswith('encoder.')),
        'inherited_decision_head':False, 'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    assert all(x == initial for x in gather(initial)), 'Rank initialization differs'
    if rank == 0:
        atomic(out/('preflight.initialization.json' if args.mode=='preflight' else 'initialization.json'),initial)
    if args.mode == 'train':
        assert initial == json.loads((out/'preflight.initialization.json').read_text())
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {'params':[p for n,p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad], 'lr':cfg['encoder_lr']},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad], 'lr':cfg['head_lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['total_steps'], eta_min=1e-6)

    def forward(items):
        batch = collate_items([items], tok.pad_token_id)
        fields = ('input_ids','attention_mask','marker_pos','marker_mask','qtype','target')
        batch = {k:v.cuda() for k,v in batch.items() if k in fields or k in ('position_ids','option_ids')}
        extra = {k:batch[k] for k in ('position_ids','option_ids') if k in batch}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits,_ = model(*(batch[k] for k in fields[:5]), **extra)
        return logits.float(),batch

    rows, parts = population('train')
    assert len(rows) == cfg['train_rows']
    def backward_ids(indices):
        total = 0.
        for ids in microbatches(indices[rank::world], parts, cfg['micro_batch'], cfg['token_budget']):
            logits,b = forward([encode_parts(parts[i], cfg['layout']) for i in ids])
            loss = rlcd_loss(logits,b['target'],b['marker_mask'],b['qtype'],
                             sigma=.4,samples=4,w_sph=.75,w_rps=1.) * (len(ids)/len(indices))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite RLCD loss')
            loss.backward()
            total += loss.item()
        synchronize(params,world)
        return total

    if args.mode == 'preflight':
        checks = []
        model.train()
        chosen = []
        for task in ('score',):
            chosen.extend([i for i,r in enumerate(rows) if r['task']==task][:world*2])
        backward_ids(chosen)
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        optimizer.zero_grad(set_to_none=True)
        checks.append('all tasks backward, finite summed distributed gradients')
        for task in ('score',):
            i = max((i for i,r in enumerate(rows) if r['task']==task),key=lambda i:parts[i]['length'])
            progress('longest_backward', task=task, id=rows[i]['id'], tokens=parts[i]['length'])
            backward_ids([i]*world)
            assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            optimizer.zero_grad(set_to_none=True)
        i=max(range(len(rows)),key=lambda i:len(parts[i]['options']))
        progress('largest_k_backward',id=rows[i]['id'],k=len(parts[i]['options']))
        backward_ids([i]*world)
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        optimizer.zero_grad(set_to_none=True)
        checks.append('longest full training input and largest native score support backward')
        order = list(range(len(rows)))
        random.Random(cfg['seed']).shuffle(order)
        progress('global_batch_preflight')
        t0 = time.time()
        backward_ids(order[:64])
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        optimizer.zero_grad(set_to_none=True)
        checks.append('first full global batch; no optimizer updates')
        batch_seconds = time.time()-t0
        model.eval()
        for split in ('dev','test_in_domain'):
            vr,vp = population(split)
            chosen = {max(range(len(vr)), key=lambda i:vp[i]['length']),
                      max(range(len(vr)), key=lambda i:len(vp[i]['options']))}
            chosen.update(next(i for i,r in enumerate(vr) if r['task']==t) for t in ('score',))
            for i in sorted(chosen):
                progress('full_evaluation_preflight', split=split, id=vr[i]['id'], tokens=vp[i]['length'])
                k = len(vp[i]['options'])
                with torch.inference_mode():
                    z,_ = forward([encode_parts(vp[i], cfg['layout'])])
                    p = canonical_probabilities(z[0],list(range(k)))
                    predict_record(vr[i],p)
                    order = list(reversed(range(k)))
                    z2,_ = forward([encode_parts(vp[i], cfg['layout'],order)])
                    p2 = canonical_probabilities(z2[0],order)
                if cfg['layout']=='parallel':
                    # bf16 kernels are approximately equivariant; report actual drift.
                    delta = max(abs(a-b) for a,b in zip(p,p2))
                    if delta > .02:
                        raise AssertionError(f'Parallel probability drift {delta} on {vr[i]["id"]}')
        checks.append('all tasks/native largest K/longest eval and semantic permutation checks')
        peak = max(gather(torch.cuda.max_memory_allocated()/2**30))
        if rank == 0:
            atomic(out/'preflight.json', {'status':'passed','optimizer_updates':0,'checks':checks,
                'initialization':initial,'peak_memory_gib':peak,'first_batch_backward_seconds':batch_seconds})
        progress('passed')
        barrier()
        if world > 1:
            dist.destroy_process_group()
        return

    state_path = out/'resume.pt'
    step, train_seconds = 0,0.
    if args.resume:
        state = torch.load(state_path,map_location='cpu',weights_only=False)
        assert state['config_sha256']==sha(args.config) and state['world_size']==world
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step,train_seconds = state['step'],state['train_seconds']
        torch.set_rng_state(state['rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['rng'][rank]['cuda'])
        del state
    else:
        assert not state_path.exists(), 'Use --resume for an existing run'
        torch.manual_seed(cfg['seed']+rank)
    def save():
        rng = gather({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()})
        if rank == 0:
            state = {'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'step':step,'sample_visits':min(step*64,len(rows)),'train_seconds':train_seconds,
                'config_sha256':sha(args.config),'world_size':world,'rng':rng}
            tmp = state_path.with_suffix('.tmp')
            torch.save(state,tmp)
            tmp.replace(state_path)
            atomic(out/'checkpoint.json',{'step':step,'sample_visits':min(step*64,len(rows)),'updated_unix':time.time()})
        barrier()

    order = list(range(len(rows)))
    random.Random(cfg['seed']).shuffle(order)
    model.train()
    times = []
    for offset in range(step*64,len(rows),64):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss = backward_ids(order[offset:offset+64])
        norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        elapsed = time.time()-t0
        train_seconds += elapsed
        times.append(elapsed)
        if step == 1 or step%10 == 0:
            total_loss = sum(gather(loss))
            recent = sum(times[-100:])/len(times[-100:])
            status = {'step':step,'total_steps':cfg['total_steps'],'sample_visits':min(step*64,len(rows)),
                'loss':total_loss,'gradient_norm':float(norm),'train_seconds':train_seconds,
                'recent_step_seconds':recent,'remaining_training_hours':recent*(cfg['total_steps']-step)/3600,
                'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
            progress('training',**status)
            if rank == 0:
                print(json.dumps(status),flush=True)
        if step%cfg['save_every']==0:
            progress('saving',step=step)
            save()
    assert step == cfg['total_steps']
    progress('saving_epoch_1',step=step)
    save()
    if rank == 0:
        save_checkpoint(model,tok,model_cfg,str(out/'checkpoint'))
        # Preserve the exact evaluated parameters, rather than quantizing the
        # upstream export to fp16 while evaluating the in-memory fp32 master.
        from safetensors.torch import save_file
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},
                  str(out/'checkpoint/model.safetensors'))
        atomic(out/'training_complete.json',{'steps':step,'sample_visits':len(rows),
                                             'train_seconds':train_seconds,'initialization':initial})
    barrier()
    from safetensors.torch import load_file
    saved=load_file(str(out/'checkpoint/model.safetensors'))
    assert all(torch.equal(value.detach().cpu(),saved[key]) for key,value in model.state_dict().items())
    del saved
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    torch.cuda.empty_cache()
    evaluation_started = time.time()
    eval_dir = out/'evaluations'
    eval_dir.mkdir(exist_ok=True)
    def infer(split,name):
        vr,vp = population(split)
        indices = list(range(len(vr)))
        mine = indices[rank::world]
        records = []
        path = eval_dir/f'{split}.{name}.rank{rank}.jsonl'
        started = time.time()
        with path.open('w',buffering=1) as journal:
            for ids in microbatches(mine,vp,cfg['eval_micro_batch'],cfg['eval_token_budget']):
                orders = [order_for(name,len(vp[i]['options']),vr[i]['id']) for i in ids]
                with torch.inference_mode():
                    logits,_ = forward([encode_parts(vp[i],cfg['layout'],o) for i,o in zip(ids,orders)])
                for i,z,o in zip(ids,logits,orders):
                    result = predict_record(vr[i],canonical_probabilities(z,o))
                    records.append(result)
                    journal.write(json.dumps(result,ensure_ascii=False)+'\n')
                if len(records)%100 < len(ids):
                    progress('evaluating',split=split,order=name,local_completed=len(records),local_total=len(mine))
        all_records = [r for chunk in gather(records) for r in chunk]
        assert len(all_records)==len(indices) and len({r['id'] for r in all_records})==len(indices)
        if rank == 0:
            report = metrics(all_records)
            report.update(rows=len(all_records),split=split,order=name,native_full_candidates=True,
                          score_decision_rule='median',seconds=time.time()-started)
            atomic(eval_dir/f'{split}.{name}.json',report)
            atomic(eval_dir/f'{split}.score_rules.json',score_rule_report(all_records,split))
        return all_records

    infer('dev','original')
    infer('test_in_domain','original')
    if rank == 0:
        atomic(out/'complete.json',{'status':'complete','steps':step,'sample_visits':len(rows),
            'train_seconds':train_seconds,'evaluation_seconds':time.time()-evaluation_started,
            'evaluated_splits':['dev','test_in_domain'],
            'dev_rows':len(population('dev')[0]),'test_rows':len(population('test_in_domain')[0]),
            'evaluation_candidates':'complete_native','score_decision_rule':'median',
            'parent_checkpoint':None,'formal_200k_checkpoint_used':False})
    progress('complete',step=step,sample_visits=len(rows))
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        try:
            cfg = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
            atomic(Path(cfg['output'])/f'failure.rank{os.getenv("RANK","0")}.json',
                   {'error':str(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()})
        except Exception:
            pass
        raise
