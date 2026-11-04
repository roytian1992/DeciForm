"""Freeze S3/S4 matched configurations and the two-GPU allocation policy."""
import copy
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from score_specialist import PROTOCOL, SpecialistObjective


def main():
    reference=PROJECT/'runs/formal_score_distribution_20261104_v1/d2_ordinal_main'
    previous=json.loads((reference/'experiment_manifest.json').read_text())
    base=json.loads((reference/'config.json').read_text())
    for name,digest in previous['code_sha256'].items():assert sha(PROJECT/name)==digest,name
    data=PROJECT/'data/score_specialist_continuous_20261104_v1'
    audit=json.loads((data/'cache/audit.json').read_text());assert audit['status']=='passed'
    assert sha(data/'manifest.json')==audit['data_manifest_sha256']
    n=audit['splits']['train']['rows'];steps=math.ceil(n/64)
    root=PROJECT/'runs/formal_score_specialist_20261104_v1';root.mkdir(exist_ok=False)
    created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    names=['score_specialist.py','models_score_specialist.py','build_score_specialist_data.py',
           'cache_score_specialist.py','worker_score_specialist.py','supervise_score_specialist.py',
           'prepare_score_specialist.py','schedule_score_specialist.py']
    hashes={**previous['code_sha256'],**{'scripts/formal/'+name:sha(PROJECT/'scripts/formal'/name) for name in names},
            'tests/test_score_specialist.py':sha(PROJECT/'tests/test_score_specialist.py')}
    for name,digest in hashes.items():
        q=root/'code'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,q);assert sha(q)==digest
    protocol={'name':PROTOCOL,'created':created,'primary_score_rule':'median','primary_metric':'native MAE',
        'cross_source_metric':'equal-source NMAE','data_manifest_sha256':sha(data/'manifest.json'),
        'labels':'original real-valued labels; no rounding; common adjacent-grid interpolation',
        'support':{'stsb':{'min':0,'max':5,'points':51},'wmt_en_de':{'min':0,'max':100,'points':101},'wmt_en_zh':{'min':0,'max':100,'points':101}},
        'candidate_policy':'complete fixed numeric grid in every train/dev/test request',
        'native_d2_checkpoint_modified':False,'initialization':'fresh original base only',
        'epochs':1,'seed':base['seed'],'world_size_per_arm':2,'global_batch':64,'steps':steps,
        'checkpoint_selection':'sole epoch-1 endpoint; no test tuning',
        'continuous_extension':'soft CE and soft CDF both use the same q; not an evaluation of unchanged five-level checkpoint on raw decimals'}
    atomic(root/'evaluation_protocol.json',protocol)
    arms={}
    for name,kind,gpus in [('s3_continuous_ce','soft_ce','4,5'),('s4_continuous_cdf','soft_cdf',None)]:
        out=root/name;out.mkdir();objective=SpecialistObjective(kind).inventory()
        cfg=copy.deepcopy(base)
        cfg.update(data=str(data),cache_audit_sha256=sha(data/'cache/audit.json'),train_rows=n,total_steps=steps,
            output=str(out),label=name,experiment_name=name,method_variant=name,
            worker_variant=str(PROJECT/'scripts/formal/worker_score_specialist.py'),
            task_losses={'score':kind},distribution_objective=objective,world_size=2,
            evaluation_protocol=PROTOCOL,reference_run=str(reference),parent_checkpoint=None,
            candidate_sampling='none; complete numeric grid for score',deferred_splits=[],
            evaluation_candidates='complete_fixed_numeric_grid',
            schedule=f'OneCycleLR, {steps} steps, pct_start=0.3',
            selection='sole epoch-1 endpoint; no metric selection; dev then complete-grid test',
            score_prediction='fixed-grid predictive median; original real-valued MAE primary',
            primary_metrics={'score':'native MAE; macro-source NMAE across ranges'},
            objective_description=kind,save_every=25,loss_diagnostics_every=25,
            score_head='unchanged D2 ordinal CDF-gap readout on full real-valued support')
        atomic(out/'config.json',cfg)
        manifest=copy.deepcopy(previous)
        manifest.update(created=created,gpus=gpus,world_size=2,config=str(out/'config.json'),
            config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
            supervisor=str(PROJECT/'scripts/formal/supervise_score_specialist.py'),
            python=sys.executable,code_sha256=hashes,code_snapshot=str(root/'code'),
            experiment_name=name,distribution_objective=objective,evaluation_protocol=PROTOCOL,
            baseline_initialization=str(reference/'initialization.json'),parent_checkpoint=None,
            expected_trainable_parameters=46304262,reference_run=str(reference),
            data=str(data),cache_audit_sha256=cfg['cache_audit_sha256'],
            packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','pyarrow']},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            tmux_session='deciform_score_specialist_20261104',
            comparison='Identical D2 architecture, fresh base, data, grids, batch, one epoch, seed and two GPUs; only soft score objective differs.')
        manifest['launch_command']=[sys.executable,manifest['supervisor'],'--manifest',str(out/'experiment_manifest.json')]
        atomic(out/'experiment_manifest.json',manifest)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'output':str(out)}
    atomic(root/'experiment_manifest.json',{'created':created,'arms':arms,'code_sha256':hashes,
        'data':str(data),'train_rows':n,'steps':steps,'world_size_per_arm':2,
        'primary_metric':'macro-source NMAE; source native MAE','protocol_sha256':sha(root/'evaluation_protocol.json'),
        'allocation_policy':{'s3':['4','5'],'s4_preference':[
            {'gpus':['0','1'],'prerequisite':str(PROJECT/'runs/formal_d2_ablations_20261104_v1/a3_no_head_geometry/status.json')},
            {'gpus':['6','7'],'prerequisite':str(PROJECT/'runs/formal_choice_order_20261104_v1/status.json')},
            {'gpus':['4','5'],'prerequisite':str(root/'s3_continuous_ce/status.json')}],
            'max_memory_mib_before_launch':1000},
        'python':sys.executable,'future_asap_arms':'S1/S2 await complete raw essays/rubrics and a separately frozen dataset; not launched with missing inputs'})
    print(json.dumps({'root':str(root),'rows':n,'steps':steps,'epochs':1,'world_size':2}),flush=True)


if __name__=='__main__':main()
