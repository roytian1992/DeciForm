"""Read-only dev diagnostics; inference interventions are not retrained ablations.

The archived first run also reversed Score; future runs preserve its order.
no_geometry removes only the direct bilinear geometry term, not its QF effect.
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'runs/formal_laya_20261106_v1/vendor/laya_release'))
from laya.common import collate_items
from laya_adapter import encode_parts, canonical_probabilities, predict_record, metrics, microbatches
from laya_mechanisms import load_transfer_checkpoint, bounded_centered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['sequential', 'parallel'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    source = PROJECT/'runs/formal_laya_mechanisms_20261106_v2'/args.arm
    cfg = json.loads((source/'config.json').read_text())
    desc = json.loads((Path(cfg['root'])/'cache/audit.json').read_text())['splits']['dev']
    rows = [json.loads(s) for s in Path(desc['views_path']).read_text().splitlines()]
    with open(desc['path'], 'rb') as f:
        parts = pickle.load(f)
    model = load_transfer_checkpoint(source/'checkpoint', device='cuda').eval()
    model.encoder.config.reference_compile = False
    captured = {}
    def scorer_hook(module, inputs, output):
        captured['base'] = output.squeeze(-1).float()
    def readout_hook(module, inputs, output):
        query, candidates, tasks, valid = inputs
        with torch.autocast('cuda', enabled=False):
            q = module.q(query.float())
            k = module.k(candidates.float())
            raw = (k*q[:, None]).sum(-1)/math.sqrt(module.projection)
            geom = bounded_centered(module.geometry_raw, module.task_weights)[tasks]
            dq = torch.bmm(geom.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)
            delta_geom = (k*dq[:, None]).sum(-1)/math.sqrt(module.projection)
        captured.update(shared=raw, geometry=delta_geom, qf=output-raw-delta_geom)
    model.scorer.register_forward_hook(scorer_hook)
    model.transfer_readout.register_forward_hook(readout_hook)
    modes = ['full', 'no_readout', 'no_geometry', 'no_qf', 'no_encoder_residual']
    records = {view:{mode:[] for mode in modes} for view in ['original', 'reverse']}
    spread = {task:{comp:[] for comp in ['base', 'shared', 'geometry', 'qf']} for task in ['choice','noul','score']}
    started = time.time()
    pad = model.encoder.config.pad_token_id
    batches = list(microbatches(list(range(len(rows))), parts, 16, 16384))
    for batch_number, ids in enumerate(batches):
        for view in ['original', 'reverse']:
            orders = [list(range(len(parts[i]['options']))) for i in ids]
            if view == 'reverse':
                orders = [o[::-1] if rows[i]['task'] in ('choice','noul') else o for i,o in zip(ids,orders)]
            b = collate_items([[encode_parts(parts[i], args.arm, o) for i,o in zip(ids,orders)]], pad)
            b = {k:v.cuda() for k,v in b.items() if isinstance(v,torch.Tensor)}
            inputs = [b[k] for k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']]
            kwargs = {k:b[k] for k in ['position_ids','option_ids'] if k in b}
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                full,_ = model(*inputs, **kwargs)
            zs = {'full':full, 'no_readout':captured['base'],
                  'no_geometry':full-captured['geometry'], 'no_qf':full-captured['qf']}
            if view == 'original':
                for j,i in enumerate(ids):
                    for comp in spread[rows[i]['task']]:
                        v=captured[comp][j,:len(orders[j])]
                        spread[rows[i]['task']][comp].append(float(v.max()-v.min()))
            for mode,z in zs.items():
                for j,i in enumerate(ids):
                    records[view][mode].append(predict_record(rows[i],canonical_probabilities(z[j],orders[j])))
            model.transfer_router.audit_disabled = True
            try:
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                    z,_=model(*inputs, **kwargs)
            finally:
                model.transfer_router.audit_disabled = False
            for j,i in enumerate(ids):
                records[view]['no_encoder_residual'].append(predict_record(rows[i],canonical_probabilities(z[j],orders[j])))
        if batch_number%50 == 0:
            print(json.dumps({'arm':args.arm,'batch':batch_number,'batches':len(batches),'seconds':time.time()-started}),flush=True)
    quality = {view:{mode:metrics(rs) for mode,rs in methods.items()} for view,methods in records.items()}
    flips={}
    for mode in modes:
        a={r['id']:r for r in records['original'][mode]}
        b={r['id']:r for r in records['reverse'][mode]}
        flips[mode]={}
        for task in ['choice','noul']:
            selected=[r for r in a.values() if r['task']==task]
            flips[mode][task]={'rows':len(selected),'reverse_flip_rate':sum(r['prediction_index']!=b[r['id']]['prediction_index'] for r in selected)/len(selected),
                'mean_max_probability_change':sum(max(abs(x-y) for x,y in zip(r['probabilities'],b[r['id']]['probabilities'])) for r in selected)/len(selected)}
    norms={}
    router=model.transfer_router
    with torch.inference_mode():
        maps=bounded_centered(router.raw,router.task_weights).float()
        norms['routing_frobenius_mean_by_task']=torch.linalg.matrix_norm(maps).mean(0).cpu().tolist()
        norms['routing_frobenius_max']=torch.linalg.matrix_norm(maps).max().item()
    def summary(xs):
        v=torch.tensor(xs)
        return {'mean':v.mean().item(),'median':v.median().item(),'p95':v.quantile(.95).item(),'max':v.max().item()}
    output={'status':'complete','diagnostic_only':True,'optimizer_updates':0,
        'caveat':'Removing branches at inference measures dependence of a jointly trained model, not a retrained ablation or causal training benefit.',
        'source':str(source),'config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
        'dev_views_sha256':desc['views_sha256'],'rows':len(rows),'native_complete_candidates':True,
        'identical_batches_across_orders':True,'score_order_preserved':True,
        'no_geometry_definition':'remove direct bilinear geometry term only; keep its QF effect',
        'quality':quality,'flips':flips,
        'logit_range_by_component':{task:{comp:summary(v) for comp,v in cs.items()} for task,cs in spread.items()},
        'parameter_diagnostics':norms,'seconds':time.time()-started}
    (args.output/(args.arm+'.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({'complete':args.arm,'seconds':output['seconds'],'output':str(args.output)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
