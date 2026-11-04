"""Freeze native and common-instruction order panels before GPU evaluation."""
import collections
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha, support
from order_audit import PANELS, view_orders


def main():
    root = PROJECT / 'runs/formal_answer_order_20261104_v1'
    full = PROJECT / 'runs/formal_score_distribution_20261104_v1/d2_ordinal_main'
    main_root = PROJECT / 'runs/formal_main_20261101_v2_one_epoch'
    full_manifest = json.loads((full / 'experiment_manifest.json').read_text())
    baseline_manifest = json.loads((main_root / 'experiment_manifest.json').read_text())
    assert sha(PROJECT / 'scripts/formal/models.py') == baseline_manifest['code_sha256']['scripts/formal/models.py']
    hashes = dict(full_manifest['code_sha256'])
    for path,digest in hashes.items():
        assert sha(PROJECT / path) == digest, path
    additions = ['scripts/formal/models.py', 'scripts/formal/order_audit.py',
                 'scripts/formal/worker_order_audit.py', 'scripts/formal/prepare_order_audit.py',
                 'scripts/formal/supervise_order_audit.py', 'tests/test_order_audit.py']
    hashes.update({path: sha(PROJECT / path) for path in additions})
    audit_path = PROJECT / 'data/formal_200k_20261101_v1/cache/audit.json'
    assert sha(audit_path) == json.loads((full/'config.json').read_text())['cache_audit_sha256']
    audit = json.loads(audit_path.read_text())
    desc = audit['splits']['test_in_domain']
    assert sha(desc['views_path']) == desc['views_sha256']
    rows = [json.loads(line) for line in open(desc['views_path'])]
    native = [r for r in rows if r['task'] in ('score','noul')]
    assert len(native) == 10000 and len({r['id'] for r in native}) == 10000
    assert collections.Counter(r['task'] for r in native) == {'score':5000,'noul':5000}
    assert all(r['candidate_mode']=='complete' and len(support(r))==(5 if r['task']=='score' else 2) for r in native)
    strata = collections.defaultdict(list)
    for row in native:
        strata[(row['task'],row['source_family'])].append(row)
    selected = set()
    for key, items in strata.items():
        assert len(items)%5 == 0
        items.sort(key=lambda r: hashlib.sha256(('order-panel-20261104:'+r['id']).encode()).hexdigest())
        selected.update(r['id'] for r in items[:len(items)//5])
    assert len(selected)==2000
    panel = {'native':[r['id'] for r in native],
             'instruction_list':[r['id'] for r in native if r['id'] in selected],
             'selection':'20% within each source/task using a fixed hash of ID; no label or prediction selection',
             'source_counts':{f'{t}/{s}':len(v) for (t,s),v in strata.items()}}
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    protocol = {'created':created, 'name':'semantic_answer_order_20261104_v1',
        'split':'test_in_domain', 'tasks':['score','noul'],
        'native_rows':10000, 'instruction_list_rows':2000,
        'candidate_policy':'every original answer exactly once; no truncation or candidate removal',
        'orders':'dihedral permutations, ten for five score levels, both for two noul outcomes; balanced candidate positions',
        'readout_alignment':'both pointer models gather option states by stable semantic value AFTER permuted causal encoding; input tokens never sorted back',
        'native_applicability':{'lora':[], 'kev':['score','noul'], 'json_sft':['score'], 'tcdl':['score','noul']},
        'instruction_list':'append identical Allowed answers JSON list to instructions; native output interfaces stay canonical; compare to original-order list, not unaugmented input',
        'score_primary':'predictive median native MAE for probability models; direct parsed numeric answer for generation',
        'noul_primary':'semantic boolean accuracy',
        'secondary':'reverse/any-order flip rate; average and worst-order error; probability TV/JS when available; valid-pair score shift',
        'uncertainty':'2000 paired cluster-bootstrap resamples by source/group_id, 95% intervals; no training-seed claim',
        'parse_failures':'counted incorrect; score error assigned native score span',
        'selection':'all orders and both panels frozen before inference; no retraining or decoder selection',
        'token_budget':16384,'max_batch_rows':8,
        'max_new_tokens':max(d['max_json_response_tokens'] for d in audit['splits'].values())+8,
        'generation':'greedy, no JSON repair or constrained decoding',
        'future_claims':'end-to-end invariance is an empirical question; no guaranteed superiority'}
    root.mkdir(exist_ok=False)
    atomic(root/'panel.json',panel);atomic(root/'protocol.json',protocol)
    for path,digest in hashes.items():
        dst=root/'code'/path;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/path,dst);assert sha(dst)==digest
    references = {'ordinary_lora':('lora',main_root/'lora'),
                  'kev':('kev',main_root/'kev'),
                  'generative_lora':('json_sft',main_root/'json_sft'),
                  'full_d2':('tcdl',full)}
    models = {}
    lookup={r['id']:r for r in native}
    for name,(arm,path) in references.items():
        config=path/'config.json';checkpoint=path/'epoch_1.pt'
        assert (path/'complete.json').exists()
        count=sum(len(view_orders(arm,p,lookup[i])) for p in PANELS for i in panel[p])
        models[name]={'arm':arm,'run':str(path),'config':str(config),'config_sha256':sha(config),
                      'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                      'expected_predictions':count}
        (root/name).mkdir()
    caches={arm:desc['arms']['kev' if arm=='tcdl' else arm] for arm in ('lora','kev','json_sft','tcdl')}
    atomic(root/'experiment_manifest.json',{'created':created,'status':'prepared',
        'models':models,'python':sys.executable,'code_sha256':hashes,
        'python_version':sys.version,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy','triton')},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'data_manifest_sha256':audit['data_manifest_sha256'],'cache_audit_sha256':sha(audit_path),
        'views_path':desc['views_path'],'views_sha256':desc['views_sha256'],'caches':caches,
        'panel_sha256':sha(root/'panel.json'),'protocol_sha256':sha(root/'protocol.json'),
        'prerequisite_status':str(PROJECT/'runs/formal_d2_ablations_20261104_v1/a2_no_quadratic/status.json'),
        'gpu_queues':{'6':['ordinary_lora','generative_lora'],'7':['full_d2','kev']},
        'launch_command':[sys.executable,str(PROJECT/'scripts/formal/supervise_order_audit.py'),'--root',str(root)],
        'scope':'fixed internal score/noul order robustness; external-source work remains deferred'})
    print(root)
    print(json.dumps({name:item['expected_predictions'] for name,item in models.items()}))


if __name__=='__main__':
    main()
