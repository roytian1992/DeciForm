"""Record pinned Laya sources, encoder, data and two authorized GPU arms."""
import argparse
import datetime
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import PROJECT, atomic, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    audit=json.loads((root/'cache/audit.json').read_text())
    assert audit['status']=='passed'
    wheel=root/'vendor/laya-0.4.26-py3-none-any.whl'
    encoder=root/'base'
    weights_sha=sha(encoder/'model.safetensors')
    code_paths=[PROJECT/'scripts/formal'/n for n in (
        'common.py','score_distribution.py','laya_adapter.py','prepare_laya.py','worker_laya.py',
        'supervise_laya.py','freeze_laya.py')]
    code_paths+=list((root/'vendor/laya_release/laya').rglob('*.py'))
    code_paths+=[PROJECT/'tests/test_laya_baseline.py']
    code_hashes={str(p):sha(p) for p in code_paths}
    for p in code_paths:
        dest=root/'code'/p.relative_to(PROJECT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    source={'laya_version':'0.4.26','repository':'https://github.com/NandhaKishorM/laya',
        'distribution':str(wheel),'distribution_sha256':sha(wheel),
        'encoder':'answerdotai/ModernBERT-large',
        'encoder_revision':'45bb4654a4d5aaff24dd11d4781fa46d39bf8c13',
        'encoder_files_sha256':{p.name:sha(p) for p in encoder.iterdir() if p.is_file()},
        'download_transport':'PyPI distribution; HF public mirror for pinned HF encoder revision',
        'data':audit,'no_released_laya_decision_weights':True}
    atomic(root/'inputs.json',source)
    order_cfg=json.loads((PROJECT/'runs/formal_independent_position_20261105_v1/config.json').read_text())
    for layout,gpus in [('sequential','4,5'),('parallel','6,7')]:
        out=root/layout
        out.mkdir(exist_ok=True)
        cfg={'root':str(root),'output':str(out),'vendor':str(root/'vendor/laya_release'),
            'layout':layout,'base':str(encoder),'encoder_revision':source['encoder_revision'],
            'base_weights_sha256':weights_sha,'seed':20261101,'parent_checkpoint':None,
            'initialization':'pretrained ModernBERT encoder; random Laya decision head',
            'train_rows':200000,'epochs':1,'global_batch':64,'total_steps':3125,'world_size':2,
            'micro_batch':8,'eval_micro_batch':16,'token_budget':16384,'eval_token_budget':16384,
            'save_every':100,'encoder_lr':2.5e-5,'head_lr':1e-4,
            'schedule':'upstream CosineAnnealingLR; min_lr=1e-6; one epoch',
            'objective':'upstream rlcd_loss: soft CE + Gaussian-logit REINFORCE; sigma=.4,samples=4,spherical=.75,RPS=1',
            'max_len':max(d['max_tokens'] for d in audit['splits'].values()),
            'cache_audit_sha256':sha(root/'cache/audit.json'),
            'candidate_views':'exact frozen train.views.jsonl (positive + at most 3 negatives for choice)',
            'evaluation_candidates':'complete_native','score_decision_rule':'median',
            'score_other_rules':['argmax','mean','mean_nearest'],
            'orders':order_cfg['orders'],'position_tasks':['choice','noul'],
            'calibration':'raw logits, temperature=1; same no-posthoc headline protocol as local baselines',
            'length_policy':'complete tokens; original RoPE frequencies; record positions beyond pretraining length, no clipping',
            'comparison':'same training views/exposures/seed/global batch; different pretrained backbone and native optimizer/loss',
            'bf16':True,'encoder_gradient_checkpointing':True,'head_gradient_checkpointing':True}
        atomic(out/'config.json',cfg)
        manifest={'created':datetime.datetime.now().astimezone().isoformat(),'project':str(PROJECT),
            'python':sys.executable,'python_version':platform.python_version(),
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','safetensors','numpy')},
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            'config':str(out/'config.json'),'config_sha256':sha(out/'config.json'),
            'code_sha256':code_hashes,'gpus':gpus,'world_size':2,
            'worker':str(PROJECT/'scripts/formal/worker_laya.py'),
            'supervisor':str(PROJECT/'scripts/formal/supervise_laya.py'),
            'environment_variables':{'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','USE_TF':'0',
                                     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4'},
            'tmux_session':'deciform_laya_'+layout+'_20261106','parent_checkpoint':None,
            'resume_command':[sys.executable,str(PROJECT/'scripts/formal/supervise_laya.py'),
                              '--manifest',str(out/'experiment_manifest.json'),'--resume']}
        atomic(out/'experiment_manifest.json',manifest)
    atomic(root/'experiment_manifest.json',{'status':'prepared','inputs':str(root/'inputs.json'),
        'arms':{name:str(root/name/'experiment_manifest.json') for name in ('sequential','parallel')},
        'data_counts':{s:d['rows'] for s,d in audit['splits'].items()},'code_snapshot':str(root/'code')})


if __name__=='__main__':
    main()
