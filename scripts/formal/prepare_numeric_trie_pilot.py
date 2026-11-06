"""Matched small N2/T2 replay on the existing frozen proper-score pilot IDs."""
import copy, datetime, json, math, pickle, shutil, subprocess, sys
from pathlib import Path
from common import PROJECT, atomic, sha

ROOT=PROJECT/'runs/pilot_numeric_trie_20261106_v1'
DATA=PROJECT/'data/pilot_numeric_trie_20261106_v1'
SMALL=PROJECT/'data/pilot_score_proper_20261106_v1'
OLD=PROJECT/'runs/formal_numeric_trie_consistency_20261106_v4'


def prepare_data(panel):
    source=SMALL/panel
    small=json.loads((source/'cache/audit.json').read_text())
    full_path=PROJECT/'data/score_numeric_trie_20261106_v1'/panel/'audit.json'
    full=json.loads(full_path.read_text())
    out=DATA/panel;out.mkdir(parents=True)
    splits={};proof={}
    for split,desc in small['splits'].items():
        assert sha(desc['views_path'])==desc['views_sha256']
        rows=[json.loads(x) for x in open(desc['views_path'])]
        d=full['splits'][split];entry=d['arms']['kev']
        assert sha(d['views_path'])==d['views_sha256'] and sha(entry['path'])==entry['sha256']
        original=[json.loads(x) for x in open(d['views_path'])]
        ix=json.loads((source/(split+'.indices.json')).read_text())
        assert ix['source_views_sha256']==d['views_sha256']
        assert rows==[original[i] for i in ix['indices']]
        with open(entry['path'],'rb') as f:allenc=pickle.load(f)
        enc=[allenc[i] for i in ix['indices']]
        assert len(enc)==len(rows)==desc['rows']
        assert all(e['labels']==[None] and e['tasks']==['score'] and not e.get('state_truncated',False) for e in enc)
        cache=out/(split+'.pkl')
        with cache.open('wb') as f:pickle.dump(enc,f,protocol=pickle.HIGHEST_PROTOCOL)
        splits[split]={**desc,'arms':{'kev':{'path':str(cache),'sha256':sha(cache),'max_tokens':max(len(e['ids']) for e in enc)}}}
        proof[split]={'rows':len(rows),'source_cache_sha256':entry['sha256'],'indices_sha256':sha(source/(split+'.indices.json')),
            'identical_records_and_order_to_proper_pilot':True,'source_encodings_reused_exactly':True}
    manifest={'panel':panel,'source_small_manifest':str(source/'manifest.json'),'source_small_manifest_sha256':sha(source/'manifest.json'),
        'source_numeric_audit':str(full_path),'source_numeric_audit_sha256':sha(full_path),'splits':proof,
        'selection':'Same frozen pilot indices, no resampling','input':'Original JSON prompt + constant score prefix; no target digits',
        'truncated':0,'modified_labels':0,'sampled_candidates':0}
    atomic(out/'manifest.json',manifest)
    audit={'status':'passed','data_manifest_sha256':sha(out/'manifest.json'),'splits':splits,
        'original_text_preserved':True,'labels_in_input':False,'full_native_support':True,'matched_small_records':True}
    atomic(out/'audit.json',audit)
    return out,audit


def main():
    if ROOT.exists() or DATA.exists():raise FileExistsError('New version required')
    templates={p:json.loads((OLD/(p+'_token_ce_cdf')/'experiment_manifest.json').read_text()) for p in ['asap','continuous']}
    paths=set().union(*(m['code_sha256'] for m in templates.values()))
    for m in templates.values():
        for p,h in m['code_sha256'].items():assert sha(PROJECT/p)==h,p
    paths.update(['scripts/formal/'+p+'.py' for p in ['models_numeric_trie_pilot','worker_numeric_trie_pilot','prepare_numeric_trie_pilot','report_numeric_trie_pilot']])
    paths.add('tests/test_numeric_trie_pilot.py')
    hashes={p:sha(PROJECT/p) for p in sorted(paths)}
    ROOT.mkdir();arms={};queues={'0,1':[],'6,7':[]}
    for p in paths:
        dst=ROOT/'code'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/p,dst)
    for panel in ['asap','continuous']:
        data,audit=prepare_data(panel)
        for variant,weight,gpus in [('control',0.,'6,7'),('token',.5,'0,1')]:
            name=panel+'_'+variant;out=ROOT/name;out.mkdir()
            m=copy.deepcopy(templates[panel]);cfg=json.loads(Path(m['config']).read_text())
            counts={k:v['rows'] for k,v in audit['splits'].items()};steps=math.ceil(counts['train']/16)
            obj={'kind':'pilot_trie_ce_cdf','ordinal_scale':3.0383850790577127,'cdf_weight':.5,'token_weight':weight}
            cfg.update(output=str(out),data=str(data),label=name,experiment_name=name,score_panel=panel,
                distribution_objective=obj,task_losses={'score':obj['kind']},global_batch=16,train_rows=counts['train'],total_steps=steps,
                save_every=16,loss_diagnostics_every=16,max_microbatch_rows=4,token_budget=8192,eval_token_budget=8192,
                checkpoint_policy='all' if panel=='asap' else 'half',parent_checkpoint=None,
                cache_audit_path=str(data/'audit.json'),cache_audit_sha256=sha(data/'audit.json'),
                evaluation_counts={k:v for k,v in counts.items() if k!='train'},evaluation_protocol='small_numeric_trie_'+panel+'_20261106_v1',
                score_architecture='small_numeric_trie_paired',method_variant='small_numeric_trie_paired',
                worker_variant=str(PROJECT/'scripts/formal/worker_numeric_trie_pilot.py'),
                grid_sizes=[4,5,6,11,31,61] if panel=='asap' else [51,101],
                paired_reference=str(ROOT/(panel+'_control')),
                schedule=f'OneCycleLR {steps} updates, pct_start=0.3',warmup_fraction=.3,
                objective_description='0.5 CE + 0.5 scaled CDF-log + token_weight * numeric token NLL',
                matched_teacher_forced_forward=True)
            atomic(out/'config.json',cfg)
            m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,gpus=gpus,
                config=str(out/'config.json'),config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
                code_sha256=hashes,code_snapshot=str(ROOT/'code'),experiment_name=name,distribution_objective=obj,
                data=str(data),data_counts=counts,data_manifest_sha256=sha(data/'manifest.json'),cache_audit_sha256=cfg['cache_audit_sha256'],
                evaluation_protocol=cfg['evaluation_protocol'],grid_sizes=cfg['grid_sizes'],paired_reference=cfg['paired_reference'],
                comparison='Same small IDs, initialization, architecture, packing, causal suffix forward; token gradient weight only 0 versus 0.5',
                scope='Small replay of N2/T2 mechanisms, not new main training or new-method evidence',
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=PROJECT).strip(),
                tmux_session='deciform_numeric_pilot_g'+gpus.replace(',','')+'_20261106')
            supervisor=PROJECT/'scripts/formal/supervise_numeric_trie.py'
            m['launch_command']=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')]
            m['resume_command']=m['launch_command']+['--resume']
            atomic(out/'experiment_manifest.json',m)
            arms[name]={'panel':panel,'variant':variant,'gpus':gpus,'manifest':str(out/'experiment_manifest.json')};queues[gpus].append(name)
    deps={g:str(PROJECT/'runs/pilot_score_proper_20261106_v1'/('asap_proper' if g=='0,1' else 'asap_cdf')/'status.json') for g in queues}
    atomic(ROOT/'experiment_manifest.json',{'arms':arms,'gpu_queues':queues,'queue_dependencies':deps,'python':sys.executable,
        'code_sha256':hashes,'epochs':1,'global_batch':16,'parent_checkpoint':None,
        'scope':'Small N2/T2 matched replay after current ASAP train AND evaluation; no full-scale queue'})
    print(json.dumps({'root':str(ROOT),'queues':queues,'dependencies':deps},indent=2))

if __name__=='__main__':main()
