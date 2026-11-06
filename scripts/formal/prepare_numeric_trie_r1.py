"""Freeze small prefix CE/CDF + R1 mean regression against completed controls."""
import copy,datetime,json,pickle,shutil,subprocess,sys
from pathlib import Path
from common import PROJECT,atomic,sha
from models_numeric_trie_r1 import R1Objective

ROOT=PROJECT/'runs/pilot_numeric_trie_r1_20261106_v1'
PREVIOUS=PROJECT/'runs/pilot_numeric_trie_20261106_v1'


def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    templates={p:json.loads((PREVIOUS/(p+'_control')/'experiment_manifest.json').read_text()) for p in ['asap','continuous']}
    paths=set().union(*(m['code_sha256'] for m in templates.values()))
    for m in templates.values():
        for p,h in m['code_sha256'].items():assert sha(PROJECT/p)==h,p
    paths.update('scripts/formal/'+p+'.py' for p in ['models_numeric_trie_r1','worker_numeric_trie_r1','prepare_numeric_trie_r1','report_numeric_trie_r1'])
    paths.add('tests/test_numeric_trie_r1.py')
    hashes={p:sha(PROJECT/p) for p in sorted(paths)}
    ROOT.mkdir();arms={};queues={};proof={}
    for p in paths:
        d=ROOT/'code'/p;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/p,d)
    for panel,gpus in [('asap','0,1'),('continuous','6,7')]:
        reference=PREVIOUS/(panel+'_control');name=panel+'_r1';out=ROOT/name;out.mkdir()
        status=json.loads((reference/'status.json').read_text());done=json.loads((reference/'complete.json').read_text())
        assert status['phase']=='complete' and status['exit_code']==0
        m=copy.deepcopy(templates[panel]);cfg=json.loads(Path(m['config']).read_text());old=copy.deepcopy(cfg)
        audit=json.loads(Path(cfg['cache_audit_path']).read_text())
        assert sha(cfg['cache_audit_path'])==cfg['cache_audit_sha256']
        assert sha(Path(cfg['data'])/'manifest.json')==audit['data_manifest_sha256']
        for split,d in audit['splits'].items():
            assert sha(d['views_path'])==d['views_sha256']
            entry=d['arms']['kev'];assert sha(entry['path'])==entry['sha256']
            with open(entry['path'],'rb') as f:enc=pickle.load(f)
            assert len(enc)==d['rows']
            assert all(e['labels']==[None] and e['tasks']==['score'] and not e.get('state_truncated',False) for e in enc)
            if split!='train':
                report=json.loads((reference/'evaluations'/('epoch_1.'+split+'.json')).read_text())
                assert report['rows']==d['rows'] and report['complete_output_grid'] and report['score_decision_rule']=='median'
        assert done['steps']==cfg['total_steps'] and done['sample_visits']==cfg['train_rows']
        obj=R1Objective().inventory()
        cfg.update(output=str(out),label=name,experiment_name=name,distribution_objective=obj,
            task_losses={'score':obj['kind']},method_variant='small_numeric_trie_r1',score_architecture='small_numeric_trie_r1',
            worker_variant=str(PROJECT/'scripts/formal/worker_numeric_trie_r1.py'),paired_reference=str(reference),
            auxiliary_loss='Normalized predictive-mean MSE, weight1; no numeric-sequence loss',
            objective_description='0.5 CE + 0.5 scaled CDF-log + normalized predictive-mean MSE',
            token_supervision_enabled=False)
        # All model/training/data/schedule/inference fields remain exactly the control values.
        allowed={'output','label','experiment_name','distribution_objective','task_losses','method_variant','score_architecture','worker_variant','paired_reference','auxiliary_loss','objective_description','token_supervision_enabled'}
        changed={k for k in cfg if cfg.get(k)!=old.get(k)}
        assert changed<=allowed,changed
        atomic(out/'config.json',cfg)
        supervisor=PROJECT/'scripts/formal/supervise_numeric_trie.py'
        m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,gpus=gpus,
            config=str(out/'config.json'),config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
            supervisor=str(supervisor),code_sha256=hashes,code_snapshot=str(ROOT/'code'),experiment_name=name,
            distribution_objective=obj,paired_reference=str(reference),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            scope='Small R1 add-on only, no new full-data experiment or numeric-sequence supervision',
            comparison='Same completed prefix control records, model, initialization, optimizer, schedule, packing and native median; adds normalized predictive-mean MSE only',
            tmux_session='deciform_numeric_r1_g'+gpus.replace(',','')+'_20261106',
            launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'])
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'panel':panel,'gpus':gpus,'manifest':str(out/'experiment_manifest.json'),'reference':str(reference)}
        queues[gpus]=[name]
        proof[panel]={'config_changes':sorted(changed),'reference':str(reference),'matched_data_sha256':cfg['cache_audit_sha256'],
                      'counts':{k:v['rows'] for k,v in audit['splits'].items()},'complete_native_records_verified':True}
    atomic(ROOT/'preparation_verified.json',proof)
    atomic(ROOT/'experiment_manifest.json',{'arms':arms,'gpu_queues':queues,'python':sys.executable,'code_sha256':hashes,
        'epochs':1,'global_batch':16,'seed':20261101,'parent_checkpoint':None,
        'scope':'Small prefix R1 vs completed CE/CDF control; no resampling/no new full-data queues'})
    print(json.dumps({'root':str(ROOT),'queues':queues,'verification':proof},indent=2))

if __name__=='__main__':main()
