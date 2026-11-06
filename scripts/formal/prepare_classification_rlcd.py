"""Freeze a small Choice/Noul pilot, preserving original splits and negatives."""
import collections
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from common import PROJECT, SEED, atomic, sha, support

ROOT = PROJECT / 'runs/pilot_classification_rlcd_20261106_v1'
DATA = PROJECT / 'data/pilot_classification_rlcd_20261106_v1'
REFERENCE = PROJECT / 'runs/formal_independent_20261105_v1'
PROTOCOL = 'classification_rlcd_pilot_20261106_v1'


def stratified_indices(rows, task, size, split):
    buckets = collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['task'] == task:
            buckets[(r['source'], json.dumps(r['label'],sort_keys=True))].append(i)
    total = sum(map(len,buckets.values()))
    if size > total:
        raise ValueError('Requested more records than the source population')
    quotas = {key:len(ids)*size//total for key,ids in buckets.items()}
    order = sorted(buckets,key=lambda k:(-(len(buckets[k])*size%total),k))
    for key in order[:size-sum(quotas.values())]:
        quotas[key] += 1
    result=[]
    for key,ids in sorted(buckets.items()):
        ids=sorted(ids,key=lambda i:hashlib.sha256(f'{SEED}:{split}:{rows[i]["id"]}'.encode()).digest())
        result.extend(ids[:quotas[key]])
    assert len(result)==size
    return result


def prepare():
    ROOT.mkdir(exist_ok=False)
    DATA.mkdir(exist_ok=False)
    (DATA/'cache').mkdir()
    prior=json.loads((REFERENCE/'experiment_manifest.json').read_text())
    full=json.loads((REFERENCE/'config.json').read_text())
    source_audit_path=Path(full['data'])/'cache/audit.json'
    assert sha(source_audit_path)==full['cache_audit_sha256']
    source_audit=json.loads(source_audit_path.read_text())
    assert sha(Path(full['data'])/'manifest.json')==source_audit['data_manifest_sha256']
    manifest={'protocol':PROTOCOL,'source_data':full['data'],'source_audit_sha256':sha(source_audit_path),
              'tasks':['choice','noul'],'seed':SEED,'selection':'proportional source/label stratification within each task; deterministic ID hash',
              'train_per_task':5000,'dev_per_task':500,'test_per_task':1000,
              'candidate_sampling':'copy original fixed sampled training views exactly; evaluation retains complete native inventory',
              'score_excluded':True,'splits':{}}
    splits={};groups={};identities={}
    for split,size in [('train',5000),('dev',500),('test_in_domain',1000)]:
        desc=source_audit['splits'][split];entry=desc['arms']['kev']
        assert sha(desc['views_path'])==desc['views_sha256'] and sha(entry['path'])==entry['sha256']
        rows=[json.loads(line) for line in open(desc['views_path'])]
        ids=sorted(stratified_indices(rows,'choice',size,split)+stratified_indices(rows,'noul',size,split))
        selected=[rows[i] for i in ids]
        assert len(ids)==2*size and len(set(ids))==len(ids)
        if split!='train':
            assert all(r['candidate_mode']=='complete' for r in selected)
            assert all(support(r)==r['native_candidate_ids'] for r in selected if r['task']=='choice')
        with open(entry['path'],'rb') as f:
            allenc=pickle.load(f)
        assert len(allenc)==len(rows)==desc['rows']
        enc=[allenc[i] for i in ids]
        assert all(e['tasks']==[r['task']] and not e['state_truncated'] and len(e['opt_idx'][0])==len(support(r)) for r,e in zip(selected,enc))
        view_path=DATA/'cache'/f'{split}.views.jsonl';cache_path=DATA/'cache'/f'{split}.kev.pkl'
        with view_path.open('w') as f:
            for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        with cache_path.open('wb') as f:pickle.dump(enc,f,protocol=pickle.HIGHEST_PROTOCOL)
        index_path=DATA/f'{split}.indices.json'
        atomic(index_path,{'source_rows':len(rows),'source_views_sha256':desc['views_sha256'],'indices':ids,'ids':[r['id'] for r in selected]})
        groups[split]=set(r['group_id'] for r in selected);identities[split]=set(r['id'] for r in selected)
        assert len(identities[split])==len(selected)
        lengths=[len(e['ids']) for e in enc]
        splits[split]={'rows':len(selected),'views_path':str(view_path),'views_sha256':sha(view_path),
                      'native_evaluation':split!='train','max_json_response_tokens':desc['max_json_response_tokens'],
                      'arms':{'kev':{'path':str(cache_path),'sha256':sha(cache_path),'max_tokens':max(lengths)}}}
        manifest['splits'][split]={'rows':len(selected),'parent_rows':len(rows),
            'parent_views_sha256':desc['views_sha256'],'parent_cache_sha256':entry['sha256'],
            'indices_path':str(index_path),'indices_sha256':sha(index_path),
            'tasks':dict(collections.Counter(r['task'] for r in selected)),
            'sources':dict(collections.Counter(r['source_family'] for r in selected)),
            'k_distribution':{t:dict(collections.Counter(len(support(r)) for r in selected if r['task']==t)) for t in ['choice','noul']},
            'max_tokens':max(lengths),'raw_rows_modified':0,'candidates_modified':0,'truncated_rows':0}
        del allenc,rows,enc
    for a,b in [('train','dev'),('train','test_in_domain'),('dev','test_in_domain')]:
        assert not groups[a]&groups[b] and not identities[a]&identities[b],(a,b)
    manifest['cross_split_id_group_overlap']=0
    atomic(DATA/'manifest.json',manifest)
    atomic(DATA/'cache/audit.json',{'status':'passed','data_manifest_sha256':sha(DATA/'manifest.json'),'splits':splits})
    additions=['scripts/formal/classification_rlcd.py','scripts/formal/prepare_classification_rlcd.py',
        'scripts/formal/worker_classification_rlcd.py','scripts/formal/supervise_classification_rlcd.py',
        'scripts/formal/report_classification_rlcd.py','tests/test_classification_rlcd.py',
        'scripts/formal/curve_contract.py',
        'runs/formal_laya_20261106_v1/vendor/laya_release/laya/common.py',
        'runs/formal_laya_20261106_v1/vendor/laya_release/laya/train.py']
    names=set(prior['code_sha256'])|set(additions)
    for name in prior['code_sha256']:
        if Path(name).name.startswith(('models_', 'kev_ours', 'independent_candidates', 'score_distribution')):
            assert sha(PROJECT/name)==prior['code_sha256'][name],name
    hashes={name:sha(PROJECT/name) for name in sorted(names)}
    for name in hashes:
        dest=ROOT/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,dest)
    arms={}
    for name,gpus,tasks in [('ce','2,3',[]),('choice_rlcd','4,5',['choice']),('noul_rlcd','6,7',['noul'])]:
        out=ROOT/name;out.mkdir()
        cfg=copy.deepcopy(full)
        cfg.update(data=str(DATA),cache_audit_sha256=sha(DATA/'cache/audit.json'),output=str(out),
            label='D2 classification pilot '+name,experiment_name=name,world_size=2,train_rows=10000,total_steps=157,
            train_views_sha256=splits['train']['views_sha256'],reference_run=str(REFERENCE),rlcd_tasks=tasks,
            worker_variant=str(PROJECT/'scripts/formal/worker_classification_rlcd.py'),
            schedule='OneCycleLR, 157 steps, pct_start=0.3',save_every=25,loss_diagnostics_every=25,
            task_losses={t:'native Laya RLCD' if t in tasks else ('categorical CE' if t=='choice' else 'Bernoulli NLL') for t in ['choice','noul']},
            evaluation_protocol=PROTOCOL,primary_metrics={'choice':'accuracy','noul':'accuracy'},
            secondary_metrics=['nll','brier'],score_rule_reports=[],
            objective_description='Fixed D2 classification pathways, task-specific native RLCD versus original CE; Score absent',
            rlcd_settings={'sigma':.4,'samples':4,'w_sph':.75,'w_rps':0.,'rng':'fork_rng preserves backbone RNG',
                           'advantage_normalization':'all noise draws and selected task rows in actual microbatch'},
            order_audit='architecture and stable semantic-ID tie rule unchanged; no Score rows')
        cfg['method_variant']='classification_rlcd_pilot'
        cfg['score_prediction']='not applicable; no Score records'
        cfg['score_head']='unused frozen-architecture Score pathway; no Score supervision'
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(prior);supervisor=PROJECT/'scripts/formal/supervise_classification_rlcd.py'
        m.pop('changed_reference_config_fields',None)
        m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,
            gpus=gpus,world_size=2,config=str(out/'config.json'),config_sha256=sha(out/'config.json'),
            worker=cfg['worker_variant'],supervisor=str(supervisor),code_sha256=hashes,code_snapshot=str(ROOT/'code'),
            experiment_name=name,rlcd_tasks=tasks,evaluation_protocol=PROTOCOL,
            data_manifest_sha256=sha(DATA/'manifest.json'),cache_audit_sha256=sha(DATA/'cache/audit.json'),reference_run=str(REFERENCE),
            baseline_initialization=str(REFERENCE/'initialization.json'),baseline_initialization_sha256=sha(REFERENCE/'initialization.json'),
            distribution_objective=cfg['distribution_objective'],data_counts={'train':10000,'dev':1000,'test_in_domain':2000},
            environment_variables={'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','FLA_DISABLE_TENSOR_CACHE':'1',
              'USE_TF':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4'},
            tmux_session='deciform_classification_rlcd_'+name+'_20261106',
            launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            comparison='Fresh-base two-task small pilot; CE baseline trained on identical subset; no 200k checkpoint or result comparison',
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'rlcd_tasks':tasks}
    atomic(ROOT/'experiment_manifest.json',{'created':datetime.datetime.now().astimezone().isoformat(),
        'arms':arms,'code_sha256':hashes,'data_manifest':str(DATA/'manifest.json'),'data_manifest_sha256':sha(DATA/'manifest.json'),
        'scope':'Exploratory two-task pilot, 10k train/1k dev/2k test; one seed one epoch; unchanged D2 architecture',
        'parent_checkpoint':None,'evaluation_candidates':'complete native','main_full_unchanged':True})
    print(json.dumps({'root':str(ROOT),'data':manifest,'arms':arms},ensure_ascii=False,indent=2))


if __name__=='__main__':
    prepare()
