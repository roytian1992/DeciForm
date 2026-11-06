"""Freeze two loss-only transfers against each completed specialist D2."""
import copy
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path
from common import PROJECT, atomic, sha


def main():
    root=PROJECT/'runs/formal_d2_rlcd_score_20261106_v1'
    if root.exists():
        raise FileExistsError(root)
    refs={'asap':PROJECT/'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap',
          'continuous':PROJECT/'runs/formal_score_numeric_20261105_v1/d2_batched'}
    originals={k:json.loads((v/'experiment_manifest.json').read_text()) for k,v in refs.items()}
    hashes={}
    for name,m in originals.items():
        assert (refs[name]/'complete.json').exists()
        for path,h in m['code_sha256'].items():
            p=PROJECT/path
            assert sha(p)==h, str(p)
            hashes[str(p.relative_to(PROJECT))]=h
    additions=['scripts/formal/score_rlcd.py','scripts/formal/models_score_rlcd.py',
       'scripts/formal/worker_score_rlcd.py','scripts/formal/prepare_score_rlcd.py',
       'scripts/formal/supervise_score_rlcd.py','scripts/formal/report_score_rlcd.py',
       'tests/test_score_rlcd.py','scripts/formal/models_independent.py','scripts/formal/asap_score.py']
    additions += [str(p.relative_to(PROJECT)) for p in (PROJECT/'runs/formal_laya_20261106_v1/vendor/laya_release/laya').rglob('*.py')]
    for path in additions:
        hashes[path]=sha(PROJECT/path)
    root.mkdir()
    for path in hashes:
        dst=root/'code'/path;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/path,dst)
    objective=dict(kind='laya_rlcd',sigma=.4,samples=4,w_sph=.75,w_rps=1.)
    arms={}
    for panel,gpus in [('asap','0,1'),('continuous','2,3')]:
        ref=refs[panel];out=root/panel;out.mkdir()
        cfg=json.loads((ref/'config.json').read_text())
        old=copy.deepcopy(cfg)
        cfg.update(output=str(out),experiment_name=panel+'_d2_rlcd',
            label='D2 + native Laya RLCD: '+panel,method_variant='score_loss_transfer',
            score_panel=panel,baseline_reference=str(ref),parent_checkpoint=None,
            distribution_objective=objective,task_losses={'score':'laya_rlcd'},
            objective_description='Native Laya noisy-logit REINFORCE + soft CE; unchanged D2 regularization',
            worker_variant=str(PROJECT/'scripts/formal/worker_score_rlcd.py'),
            max_microbatch_rows=16 if panel=='asap' else 4,
            grid_sizes=[4,5,6,11,31,61] if panel=='asap' else [51,101],
            schedule=f"OneCycleLR, {cfg['total_steps']} steps, pct_start={cfg['warmup_fraction']}",
            no_score_permutations=True)
        unchanged=['data','base','seed','initialization','train_rows','total_steps','global_batch',
            'epochs','token_budget','eval_token_budget','checkpoint_policy','warmup_fraction',
            'base_lr','ablation','structural_penalty','cache_audit_sha256','evaluation_protocol']
        assert all(cfg[k]==old[k] for k in unchanged)
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(originals[panel]);supervisor=PROJECT/'scripts/formal/supervise_score_rlcd.py'
        for key in ['execution_benchmark','execution_benchmark_sha256','paired_reference','completed_dependency','completed_dependency_sha256']:
            m.pop(key,None)
        m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,
            config=str(out/'config.json'),config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
            supervisor=str(supervisor),code_sha256=hashes,code_snapshot=str(root/'code'),
            experiment_name=cfg['experiment_name'],gpus=gpus,world_size=2,
            baseline_reference=str(ref),baseline_initialization=str(ref/'initialization.json'),
            baseline_initialization_sha256=sha(ref/'initialization.json'),
            baseline_config_sha256=sha(ref/'config.json'),
            distribution_objective=objective,
            environment_variables={'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false',
                'FLA_DISABLE_TENSOR_CACHE':'1','USE_TF':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4'},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            tmux_session='deciform_d2_rlcd_'+panel+'_20261106',
            launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            comparison='Same specialist D2 architecture/fresh initialization/data/order/packing/optimizer/schedule/regularizer/median; only data loss changes to native Laya RLCD.',
            scope='Separate RQ-Score loss transfer; no main-model or A1-A3 retraining.')
        atomic(out/'experiment_manifest.json',m)
        arms[panel]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'reference':str(ref)}
    atomic(root/'experiment_manifest.json',{'created':datetime.datetime.now().astimezone().isoformat(),
        'arms':arms,'python':sys.executable,'code_sha256':hashes,'objective':objective,
        'parent_checkpoint':None,'seed':20261101,'epochs':1,'global_batch':64,
        'scope':'Two Score-only loss transfer trainings; complete native evaluation, no Score order tests'})
    print(json.dumps({'root':str(root),'arms':arms},indent=2))


if __name__=='__main__':
    main()
