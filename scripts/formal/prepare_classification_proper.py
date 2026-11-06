"""Freeze follow-up objectives against the completed CE pilot; never rebuild data."""
import copy
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path
from common import PROJECT, atomic, sha
from classification_proper import ClassificationObjective

ROOT=PROJECT/'runs/pilot_classification_proper_20261106_v1'
PREVIOUS=PROJECT/'runs/pilot_classification_rlcd_20261106_v1'
ARMS=[('rlcd_scaled','2,3',ClassificationObjective(kind='scaled_rlcd',increment_weight=.1)),
      ('proper','4,5',ClassificationObjective(kind='proper')),
      ('rlcd_both','6,7',ClassificationObjective(kind='scaled_rlcd',increment_weight=1.))]

def prepare():
    assert not (ROOT/'experiment_manifest.json').exists(), 'Already frozen'
    prior=json.loads((PREVIOUS/'ce/experiment_manifest.json').read_text())
    base=json.loads(Path(prior['config']).read_text())
    assert sha(prior['config'])==prior['config_sha256']
    assert json.loads((PREVIOUS/'ce/status.json').read_text())['phase']=='complete'
    data=Path(base['data'])
    assert sha(data/'manifest.json')==prior['data_manifest_sha256']
    assert sha(data/'cache/audit.json')==base['cache_audit_sha256']
    for name,digest in prior['code_sha256'].items():
        assert sha(PROJECT/name)==digest, name
    additions=['scripts/formal/classification_proper.py','scripts/formal/worker_classification_proper.py',
       'scripts/formal/supervise_classification_proper.py','scripts/formal/prepare_classification_proper.py',
       'scripts/formal/report_classification_proper.py','tests/test_classification_proper.py']
    hashes={name:sha(PROJECT/name) for name in sorted(set(prior['code_sha256'])|set(additions))}
    for name in hashes:
        dest=ROOT/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,dest)
    arms={}
    for name,gpus,objective in ARMS:
        out=ROOT/name;out.mkdir()
        cfg=copy.deepcopy(base)
        cfg.update(output=str(out),label='D2 classification proper pilot '+name,experiment_name=name,
            method_variant='classification_proper_pilot',
            worker_variant=str(PROJECT/'scripts/formal/worker_classification_proper.py'),
            rlcd_tasks=['choice','noul'],classification_objective=objective.inventory(),
            objective_description='Both tasks: native RLCD, scaled RLCD increment, or K-normalized direct proper score',
            task_losses={t:objective.inventory() for t in ['choice','noul']})
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(prior);supervisor=PROJECT/'scripts/formal/supervise_classification_proper.py'
        m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,
            gpus=gpus,config=str(out/'config.json'),config_sha256=sha(out/'config.json'),
            worker=cfg['worker_variant'],supervisor=str(supervisor),code_sha256=hashes,
            code_snapshot=str(ROOT/'code'),experiment_name=name,rlcd_tasks=cfg['rlcd_tasks'],
            classification_objective=objective.inventory(),
            tmux_session='deciform_classification_proper_'+name+'_20261106',
            launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            baseline_ce=str(PREVIOUS/'ce'),baseline_ce_config_sha256=prior['config_sha256'],
            comparison='Same frozen small data, initialization, optimizer and single epoch; both tasks treated; no Score',
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'objective':objective.inventory()}
    atomic(ROOT/'experiment_manifest.json',{'created':datetime.datetime.now().astimezone().isoformat(),
       'arms':arms,'code_sha256':hashes,'baseline_ce':str(PREVIOUS/'ce'),
       'baseline_ce_config_sha256':prior['config_sha256'],
       'data_manifest':str(data/'manifest.json'),'data_manifest_sha256':sha(data/'manifest.json'),
       'data_counts':prior['data_counts'],'parent_checkpoint':None,'main_full_unchanged':True,
       'hypothesis':'Reduce noisy-gradient scale or directly optimize a proper categorical score',
       'coefficients_frozen_before_training':True,'checkpoint_selection':'epoch 1 endpoint only'})
    print(json.dumps({'root':str(ROOT),'arms':arms},indent=2))

if __name__=='__main__':prepare()
