"""Freeze the native-path transfer against the completed parallel Laya control."""
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import PROJECT, atomic, sha


def main():
    root=PROJECT/'runs/formal_laya_native_transfer_20261106_v1'
    base=PROJECT/'runs/formal_laya_20261106_v1'
    if root.exists():
        raise FileExistsError(root)
    m=json.loads((base/'parallel/experiment_manifest.json').read_text())
    assert json.loads((base/'parallel/status.json').read_text())['phase']=='complete'
    hashes=dict(m['code_sha256'])
    for path,digest in hashes.items():
        assert sha(path)==digest,path
    files=['scripts/formal/laya_mechanisms.py','scripts/formal/laya_native_transfer.py',
           'scripts/formal/worker_laya_native_transfer.py','scripts/formal/prepare_laya_native_transfer.py',
           'scripts/formal/report_laya_native_transfer.py','tests/test_laya_native_transfer.py',
           'tests/test_laya_mechanisms.py','tests/test_laya_baseline.py']
    for file in files:
        p=PROJECT/file
        hashes[str(p)]=sha(p)
    root.mkdir()
    for path in hashes:
        p=Path(path);t=root/'code'/p.relative_to(PROJECT)
        t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    out=root/'parallel';out.mkdir()
    cfg=json.loads((base/'parallel/config.json').read_text())
    cfg.update(output=str(out),study_root=str(root),baseline_reference=str(base/'parallel'),
        baseline_initialization=str(base/'parallel/initialization.json'),
        baseline_initialization_sha256=sha(base/'parallel/initialization.json'),
        mechanisms={'rank':16,'alpha':32,'projection':256,'quadratic_rank':64},
        module_lr=1e-3,structural_regularization=1e-3,
        method='native scorer geometry + centered task-only projection contrasts + choice QF',
        added_parameter_lr='A/B: encoder_lr; task matrices/QF: module_lr',
        new_modules_zero_output=True,
        order_batching='identical original full row groups across all classification views; Score never reordered',
        comparison='Same native parallel Laya encoder/head initialization, full fine-tuning/data/RLCD; task-native modules with separate LR and structural regularization. Joint revision, not isolated-component ablation.')
    atomic(out/'config.json',cfg)
    supervisor=PROJECT/'scripts/formal/supervise_laya.py'
    m.update(created=datetime.datetime.now().astimezone().isoformat(),config=str(out/'config.json'),
        config_sha256=sha(out/'config.json'),code_sha256=hashes,
        worker=str(PROJECT/'scripts/formal/worker_laya_native_transfer.py'),gpus='4,5',
        tmux_session='deciform_laya_native_transfer_20261106',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
        resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
        baseline_manifest=str(base/'parallel/experiment_manifest.json'))
    atomic(out/'experiment_manifest.json',m)
    atomic(root/'experiment_manifest.json',{'created':m['created'],'python':sys.executable,
        'arms':{'parallel':{'manifest':str(out/'experiment_manifest.json'),'gpus':'4,5'}},
        'code_sha256':hashes,'code_snapshot':str(root/'code'),'baseline_root':str(base),
        'data_audit':str(base/'cache/audit.json'),'data_audit_sha256':sha(base/'cache/audit.json'),
        'seed':20261101,'epochs':1,'train_rows':200000,'global_batch':64,'parent_checkpoint':None,
        'criterion':'Report all three native task metrics and all seven classification orders; compare completed same-layout baseline and preserve all earlier negative evidence.'})
    print(json.dumps({'root':str(root),'launch':m['launch_command']},indent=2))


if __name__=='__main__':
    main()
