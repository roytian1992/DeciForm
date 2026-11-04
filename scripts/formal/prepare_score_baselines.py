"""Freeze four same-data Score runs: Kev and JSON LoRA, two panels."""
import copy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha
from score_baselines import PROTOCOL


def main():
    root=PROJECT/'runs/formal_score_baselines_20261105_v1'
    panel_runs={
        'asap': PROJECT/'runs/formal_score_asap_20261104_v1',
        'continuous': PROJECT/'runs/formal_score_specialist_20261104_v1'}
    reference_names={'asap':['s1_asap_ce','s2_asap_cdf'],
                     'continuous':['s3_continuous_ce','s4_continuous_cdf']}
    reference=panel_runs['asap']/reference_names['asap'][0]
    previous=json.loads((reference/'experiment_manifest.json').read_text())
    hashes=copy.deepcopy(previous['code_sha256'])
    for name,digest in hashes.items():
        assert sha(PROJECT/name)==digest,name
    extras=['score_baselines.py','models_score_baselines.py','worker_score_baselines.py',
            'supervise_score_baselines.py','schedule_score_baselines.py','prepare_score_baselines.py',
            'cache_score_baselines.py','models.py']
    for name in extras:hashes['scripts/formal/'+name]=sha(PROJECT/'scripts/formal'/name)
    for name in ('scripts/kev_ours_v2.py','tests/test_score_baselines.py'):
        hashes[name]=sha(PROJECT/name)
    panels={}
    for panel,parent in panel_runs.items():
        refs=[parent/name for name in reference_names[panel]]
        cfg=json.loads((refs[0]/'config.json').read_text())
        audit_path=PROJECT/'data/score_baseline_encodings_20261105_v1'/panel/'audit.json'
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='passed' and sha(audit['original_audit'])==audit['original_audit_sha256']
        assert sha(PROJECT/'scripts/formal/score_baselines.py')==audit['encoder_sha256']
        assert sha(PROJECT/'scripts/formal/cache_score_baselines.py')==audit['code_sha256']
        assert sha(Path(cfg['data'])/'manifest.json')==audit['data_manifest_sha256']
        for split,desc in audit['splits'].items():
            assert sha(desc['views_path'])==desc['views_sha256']
            for arm in ('kev','json_sft'):assert sha(desc['arms'][arm]['path'])==desc['arms'][arm]['sha256']
        for ref in refs:
            done=json.loads((ref/'complete.json').read_text())
            assert done['sample_visits']==cfg['train_rows'] and done['steps']==cfg['total_steps']
        panels[panel]={'config':cfg,'audit':audit,'audit_path':audit_path,'references':refs}
    root.mkdir(exist_ok=False)
    for name,digest in hashes.items():
        dest=root/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/name,dest);assert sha(dest)==digest
    created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    arms={}
    for panel,info in panels.items():
        for arm in ('kev','json_sft'):
            name=panel+'_'+arm;out=root/name;out.mkdir()
            old=info['config']
            # Keep data, optimizer horizon, batching budgets and hardware count.
            cfg={key:old[key] for key in ('data','base','initialization','seed','global_batch',
                'epochs','token_budget','eval_token_budget','save_every','gradient_checkpointing',
                'base_lr','warmup_fraction','train_rows','parent_checkpoint','checkpoint_policy',
                'preflight_memory_limit_gib','world_size','total_steps','evaluation_splits')}
            objective={'kind':'categorical_CE' if panel=='asap' else 'adjacent_grid_soft_CE'} if arm=='kev' else {'kind':'teacher_forced_token_CE','target':'JSON with original numeric label'}
            cfg.update(arm=arm,panel=panel,output=str(out),label=name,experiment_name=name,
                worker_variant=str(PROJECT/'scripts/formal/worker_score_baselines.py'),
                cache_audit_path=str(info['audit_path']),cache_audit_sha256=sha(info['audit_path']),
                objective=objective,task_losses={'score':objective['kind']},task_balanced=False,
                score_decision_rule='median' if arm=='kev' else 'generated_json_number',
                evaluation_protocol=PROTOCOL,max_new_tokens=64,
                candidate_sampling='none',evaluation_candidates='complete declared native range',
                label_kind='native_integer' if panel=='asap' else 'continuous',
                grid_sizes=[4,5,6,11,31,61] if panel=='asap' else [51,101],
                selection='sole epoch-1 endpoint; no metric checkpoint selection',
                reference_runs=[str(p) for p in info['references']],
                invalid_json_policy='full allowed range error; denominator includes every test row; report validity and valid-only MAE separately')
            atomic(out/'config.json',cfg)
            baseline=PROJECT/'runs/formal_main_20261101_v2_one_epoch'/arm/'initialization.json'
            expected=json.loads(baseline.read_text())['trainable_parameters']
            supervisor=str(PROJECT/'scripts/formal/supervise_score_baselines.py')
            m={'created':created,'cwd':str(PROJECT),'python':sys.executable,
                'python_version':platform.python_version(),
                'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','triton')},
                'config':str(out/'config.json'),'config_sha256':sha(out/'config.json'),
                'worker':cfg['worker_variant'],'supervisor':supervisor,
                'world_size':2,'gpus':None,'arm':arm,'panel':panel,'experiment_name':name,
                'code_sha256':hashes,'code_snapshot':str(root/'code'),
                'baseline_initialization':str(baseline),'baseline_initialization_sha256':sha(baseline),
                'expected_trainable_parameters':expected,'initialization':'fresh_base_only','parent_checkpoint':None,
                'objective':objective,'evaluation_protocol':PROTOCOL,
                'environment_variables':{'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','FLA_DISABLE_TENSOR_CACHE':'1'},
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
                'launch_command':[sys.executable,supervisor,'--manifest',str(out/'runtime_manifest.json')],
                'resume_command':[sys.executable,supervisor,'--manifest',str(out/'runtime_manifest.json'),'--resume'],
                'cache_audit_sha256':sha(info['audit_path']),
                'reference_files_sha256':{str(p/f):sha(p/f) for p in info['references'] for f in
                    ('config.json','complete.json','evaluations/epoch_1.test_in_domain.json')},
                'comparison':'Exact frozen records/labels/splits/full inputs, same seed/global batches/one epoch/two GPUs as Score specialist. Kev keeps original pointer and CE; JSON LoRA uses original all-target adapters and token CE. Only output interface differs: native integers for ASAP; original real-valued numbers for continuous JSON, same grid interpolation for pointer CE/CDF. No trained inheritance.'}
            atomic(out/'experiment_manifest.json',m)
            arms[name]={'manifest':str(out/'experiment_manifest.json'),'output':str(out),'panel':panel,'arm':arm}
    atomic(root/'experiment_manifest.json',{'created':created,'python':sys.executable,
        'arms':arms,'priority':['asap_kev','asap_json_sft','continuous_kev','continuous_json_sft'],
        'gpu_pairs':['0,1','2,3'],'max_memory_mib_before_launch':1000,
        'release_dependencies':{
            '0,1':str(PROJECT/'runs/formal_d2_curves_20261104_v1/d2_50k/status.json'),
            '2,3':str(PROJECT/'runs/formal_d2_curves_20261104_v1/d2_100k/status.json')},
        'code_sha256':hashes,'ordinary_lora':'cancelled by user; no model/run',
        'references':{panel:[str(p) for p in info['references']] for panel,info in panels.items()},
        'primary':'native MAE; equal-source/prompt NMAE; median for probabilities, parsed number for generative',
        'wmt_reporting':'English-to-Chinese is the requested main WMT view; retain complete German results and unchanged mixed-source training for comparability',
        'generative_probability_metrics':False,'parent_checkpoint':None,
        'tmux_session':'deciform_score_baselines_20261105'})
    print(root)


if __name__=='__main__':
    main()
