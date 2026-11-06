"""Freeze two mechanism transfers against completed same-layout Laya baselines."""
import copy
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path
from common import PROJECT, atomic, sha


def main():
    root=PROJECT/'runs/formal_laya_mechanisms_20261106_v2'
    base=PROJECT/'runs/formal_laya_20261106_v1'
    if root.exists():raise FileExistsError(root)
    baseline_manifests={a:json.loads((base/a/'experiment_manifest.json').read_text()) for a in ('sequential','parallel')}
    hashes={}
    for a,m in baseline_manifests.items():
        assert json.loads((base/a/'status.json').read_text())['phase']=='complete'
        for path,digest in m['code_sha256'].items():
            assert sha(path)==digest,path
            hashes[path]=digest
    for path in ['scripts/formal/laya_mechanisms.py','scripts/formal/worker_laya_mechanisms.py',
                 'scripts/formal/prepare_laya_mechanisms.py','scripts/formal/report_laya_mechanisms.py',
                 'scripts/formal/schedule_research_queue.py','tests/test_laya_mechanisms.py']:
        p=PROJECT/path;hashes[str(p)]=sha(p)
    root.mkdir()
    for path in hashes:
        p=Path(path);target=root/'code'/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    arms={};queues={};deps={}
    for a,gpus,dependency in [('sequential','4,5','continuous_token_ce'),('parallel','6,7','continuous_token_ce_cdf')]:
        out=root/a;out.mkdir()
        cfg=json.loads((base/a/'config.json').read_text())
        cfg.update(output=str(out),study_root=str(root),baseline_reference=str(base/a),
                   baseline_initialization=str(base/a/'initialization.json'),
                   baseline_initialization_sha256=sha(base/a/'initialization.json'),
                   mechanisms={'rank':16,'alpha':32,'projection':256,'quadratic_rank':64},
                   method='task-routed residual + task geometry + choice quadratic readout',
                   added_parameter_lr=cfg['head_lr'],new_modules_zero_output=True,
                   comparison='Same backbone/layout/fine-tuning/data/optimizer/loss; added routing and readout modules; shared capacity also increases')
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(baseline_manifests[a]);supervisor=PROJECT/'scripts/formal/supervise_laya.py'
        m.update(created=datetime.datetime.now().astimezone().isoformat(),config=str(out/'config.json'),
                 config_sha256=sha(out/'config.json'),code_sha256=hashes,worker=str(PROJECT/'scripts/formal/worker_laya_mechanisms.py'),
                 gpus=gpus,tmux_session='deciform_laya_mechanisms_v2_'+a+'_20261106',
                 git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
                 launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
                 resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
                 baseline_manifest=str(base/a/'experiment_manifest.json'))
        atomic(out/'experiment_manifest.json',m)
        arms[a]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus};queues[gpus]=[a]
        deps[gpus]=str(PROJECT/'runs/formal_numeric_trie_consistency_20261106_v4'/dependency/'status.json')
    atomic(root/'experiment_manifest.json',{'created':datetime.datetime.now().astimezone().isoformat(),
        'python':sys.executable,'arms':arms,'gpu_queues':queues,'queue_dependencies':deps,
        'code_sha256':hashes,'code_snapshot':str(root/'code'),'baseline_root':str(base),
        'data_audit':str(base/'cache/audit.json'),'data_audit_sha256':sha(base/'cache/audit.json'),
        'seed':20261101,'epochs':1,'train_rows':200000,'global_batch':64,'parent_checkpoint':None,
        'criterion':'Report each same-layout delta on Choice/Noul accuracy and Score median MAE, all sources and position changes; no test-driven selection or stable-seed claim.'})
    print(json.dumps({'root':str(root),'arms':arms},indent=2))

if __name__=='__main__':main()
