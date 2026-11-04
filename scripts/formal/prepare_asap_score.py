"""Freeze the matched S1/S2 native-score experiment for GPUs 4/5 and 6/7."""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT,atomic,sha
from asap_score import PROTOCOL,RANGES,SpecialistObjective


def main():
    reference=PROJECT/'runs/formal_score_specialist_20261104_v1/s3_continuous_ce'
    previous=json.loads((reference/'experiment_manifest.json').read_text())
    base=json.loads((reference/'config.json').read_text())
    for name,digest in previous['code_sha256'].items():assert sha(PROJECT/name)==digest,name
    data=PROJECT/'data/score_specialist_asap_fold0_20261104_v1'
    audit=json.loads((data/'cache/audit.json').read_text());assert audit['status']=='passed'
    assert sha(data/'manifest.json')==audit['data_manifest_sha256']
    n=audit['splits']['train']['rows'];steps=math.ceil(n/64)
    root=PROJECT/'runs/formal_score_asap_20261104_v1';root.mkdir(exist_ok=False)
    created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    names=['asap_score.py','build_asap_score_data.py','cache_asap_score.py','worker_asap_score.py',
           'prepare_asap_score.py','supervise_asap_score.py']
    hashes={**previous['code_sha256'],**{'scripts/formal/'+name:sha(PROJECT/'scripts/formal'/name) for name in names},
            'tests/test_asap_score.py':sha(PROJECT/'tests/test_asap_score.py')}
    for name,digest in hashes.items():
        dest=root/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,dest);assert sha(dest)==digest
    protocol={'name':PROTOCOL,'created':created,'primary_score_rule':'median',
        'primary_metric':'native MAE per prompt; equal-prompt NMAE',
        'secondary_metrics':['RMSE','Spearman','QWK per prompt','NLL','normalized RPS'],
        'data_manifest_sha256':sha(data/'manifest.json'),
        'labels':'original integer domain1_score; no score normalization or rounding',
        'ranges':RANGES,'author_fold':0,'candidate_policy':'complete native support in train/dev/test',
        'input':'verbatim author-decoded essay; complete prompt/rubric/source passage and adjudication rules',
        'current_full_d2_modified':False,'initialization':'fresh original base only',
        'epochs':1,'seed':base['seed'],'world_size_per_arm':2,'global_batch':64,'steps':steps,
        'checkpoint_selection':'sole epoch-1 endpoint; no test tuning',
        'loss_comparison':'same D2 architecture; categorical CE versus original D2 ordinal loss; native targets make q exactly one-hot',
        'ordinal_scale':3.0383850790577127}
    atomic(root/'evaluation_protocol.json',protocol);arms={}
    for name,kind,gpus,label in [('s1_asap_ce','soft_ce','4,5','ASAP native categorical CE'),
                                 ('s2_asap_cdf','soft_cdf','6,7','ASAP original D2 ordered-CDF loss')]:
        out=root/name;out.mkdir();objective=SpecialistObjective(kind).inventory();cfg=copy.deepcopy(base)
        cfg.update(data=str(data),cache_audit_sha256=sha(data/'cache/audit.json'),train_rows=n,total_steps=steps,
            output=str(out),label=label,experiment_name=name,method_variant=name,
            worker_variant=str(PROJECT/'scripts/formal/worker_asap_score.py'),
            task_losses={'score':label},distribution_objective=objective,world_size=2,
            evaluation_protocol=PROTOCOL,reference_run=str(reference),parent_checkpoint=None,
            candidate_sampling='none; complete native integer support',
            evaluation_candidates='complete_native',deferred_splits=[],
            schedule=f'OneCycleLR, {steps} steps, pct_start=0.3',
            selection='sole epoch-1 endpoint; dev then complete native test; no metric selection',
            score_prediction='native predictive median; original integer MAE primary',
            primary_metrics={'score':'native MAE per prompt; equal-prompt NMAE'},
            objective_description=label,save_every=25,loss_diagnostics_every=25,
            token_budget=4096,eval_token_budget=4096,
            score_head='unchanged D2 ordinal CDF-gap readout on complete native integer support',
            label_kind='native_integer',soft_target_implementation='one-hot at every original native grid point; exact CE/D2')
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(previous)
        m.update(created=created,gpus=gpus,world_size=2,config=str(out/'config.json'),
            config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
            python=sys.executable,code_sha256=hashes,code_snapshot=str(root/'code'),
            experiment_name=name,distribution_objective=objective,evaluation_protocol=PROTOCOL,
            parent_checkpoint=None,reference_run=str(reference),data=str(data),
            cache_audit_sha256=cfg['cache_audit_sha256'],
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            tmux_session='deciform_asap_score_20261104',
            comparison='Identical D2 architecture, fresh base, complete original inputs/support, seed, batch, two GPUs, one epoch; only native Score objective differs.')
        m['launch_command']=[sys.executable,m['supervisor'],'--manifest',str(out/'experiment_manifest.json')]
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'output':str(out)}
    atomic(root/'experiment_manifest.json',{'created':created,'arms':arms,'code_sha256':hashes,
        'data':str(data),'train_rows':n,'steps':steps,'world_size_per_arm':2,
        'protocol_sha256':sha(root/'evaluation_protocol.json'),'python':sys.executable,
        'max_memory_mib_before_launch':1000})
    print(json.dumps({'root':str(root),'train_rows':n,'steps':steps,'gpus':'S1 4,5; S2 6,7'}))


if __name__=='__main__':main()
