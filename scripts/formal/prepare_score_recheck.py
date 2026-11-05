"""Freeze a fresh JSON-LoRA replay and the missing two-source Score loss runs."""
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha


def main():
    root = PROJECT / 'runs/formal_score_recheck_20261106_v1'
    integrity = PROJECT / 'runs/score_generative_integrity_20261106_v1/audit.json'
    checked = json.loads(integrity.read_text())
    assert checked['status'] == 'passed' and set(checked['panels']) == {'asap', 'continuous'}
    baselines = PROJECT / 'runs/formal_score_baselines_20261105_v1'
    numeric = PROJECT / 'runs/formal_score_numeric_20261105_v1/numeric_cdf'
    numeric_cfg = json.loads((numeric/'config.json').read_text())
    numeric_manifest = json.loads((numeric/'experiment_manifest.json').read_text())
    objectives = PROJECT / 'runs/formal_score_objectives_20261106_v1'
    previous = json.loads((baselines/'asap_json_sft/experiment_manifest.json').read_text())
    for name in ['models.py', 'models_score_baselines.py', 'score_baselines.py', 'worker_score_baselines.py']:
        path = 'scripts/formal/'+name
        assert sha(PROJECT/path) == previous['code_sha256'][path], 'Original baseline behavior changed: '+path
    paths = set(previous['code_sha256']) | set(numeric_manifest['code_sha256']) | set(
        json.loads((objectives/'r1_cdf_mean/experiment_manifest.json').read_text())['code_sha256']) | {
        'scripts/formal/audit_score_generative.py', 'scripts/formal/worker_score_json_recheck.py',
        'scripts/formal/worker_numeric_continuous_objectives.py', 'scripts/formal/cache_score_recheck.py',
        'scripts/formal/prepare_score_recheck.py', 'scripts/formal/schedule_research_queue.py',
        'tests/test_continuous_score_objectives.py'}
    hashes = {p: sha(PROJECT/p) for p in sorted(paths)}
    root.mkdir(exist_ok=False)
    for p in paths:
        target = root/'code'/p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT/p, target)
    arms = {}
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()

    def freeze(name, gpus, cfg, manifest, worker, supervisor):
        out = root/name
        out.mkdir()
        cfg.update(output=str(out), label=name, experiment_name=name, worker_variant=str(PROJECT/worker))
        assert cfg['parent_checkpoint'] is None and cfg['epochs'] == 1 and cfg['global_batch'] == 64
        atomic(out/'config.json', cfg)
        manifest.update(created=created, python=sys.executable, config=str(out/'config.json'),
            config_sha256=sha(out/'config.json'), worker=str(PROJECT/worker), supervisor=str(PROJECT/supervisor),
            code_sha256=hashes, code_snapshot=str(root/'code'), gpus=gpus, experiment_name=name,
            launch_command=[sys.executable,str(PROJECT/supervisor),'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,str(PROJECT/supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            integrity_audit=str(integrity), integrity_audit_sha256=sha(integrity), git_head=git_head,
            tmux_session='deciform_score_recheck_g'+gpus.replace(',','')+'_20261106')
        for key in ['completed_dependency', 'completed_dependency_sha256']:
            manifest.pop(key, None)
        atomic(out/'experiment_manifest.json', manifest)
        arms[name] = {'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'train_rows':cfg['train_rows'],'steps':cfg['total_steps']}

    for panel, name, gpus in [('asap','asap_json_replay','0,1'), ('continuous','continuous_json_replay','4,5')]:
        prior = baselines/(panel+'_json_sft')
        cfg = json.loads((prior/'config.json').read_text())
        m = json.loads((prior/'experiment_manifest.json').read_text())
        if panel == 'continuous':
            data = PROJECT/'data/score_specialist_stsb_enzh_20261105_v1'
            audit = PROJECT/'data/score_baseline_encodings_stsb_enzh_20261106_v1/audit.json'
            proof = json.loads(audit.read_text())
            assert proof['status']=='passed' and proof['data_manifest_sha256']==sha(data/'manifest.json')
            assert {s:d['rows'] for s,d in proof['splits'].items()}=={'train':8129,'dev':2500,'test_in_domain':2379}
            cfg.update(data=str(data), cache_audit_path=str(audit), cache_audit_sha256=sha(audit),
                       train_rows=8129,total_steps=128)
            m['cache_audit_sha256']=sha(audit)
        cfg['replay_of']=str(prior)
        cfg['replay_scope']=('Same exact ASAP population/seed/initialization/schedule/prompt/decoder; fresh independent execution'
            if panel=='asap' else 'Fresh repeat on the current STS-B/en-zh panel; old en-de rows excluded, with prompt/decoder/optimizer unchanged')
        m.update(comparison=cfg['replay_scope'],reference_run=str(prior),
                 generation_input_policy='Only cached prompt tokens; remove labels and answer tokens before predict')
        freeze(name,gpus,cfg,m,'scripts/formal/worker_score_json_recheck.py','scripts/formal/supervise_score_baselines.py')

    for source, name, gpus in [('r1_cdf_mean','continuous_r1_cdf_mean','4,5'),('r2_crps','continuous_r2_crps','6,7')]:
        parent = json.loads((objectives/source/'experiment_manifest.json').read_text())
        source_cfg = json.loads((objectives/source/'config.json').read_text())
        cfg = copy.deepcopy(numeric_cfg)
        cfg.update(distribution_objective=source_cfg['distribution_objective'],
                   task_losses=source_cfg['task_losses'], objective_description=source_cfg['objective_description'],
                   method_variant='numeric_score_objective_research', score_panel='continuous', grid_sizes=[51,101],
                   paired_reference=str(numeric), label_kind='continuous')
        m = copy.deepcopy(parent)
        m.update(data=cfg['data'],data_counts={'train':8129,'dev':2500,'test_in_domain':2379},
                 data_manifest_sha256=sha(Path(cfg['data'])/'manifest.json'),cache_audit_sha256=cfg['cache_audit_sha256'],
                 evaluation_protocol=cfg['evaluation_protocol'],paired_reference=str(numeric),
                 distribution_objective=cfg['distribution_objective'],
                 numeric_initialization_reference=str(numeric/'initialization.json'),
                 execution_benchmark=numeric_manifest['execution_benchmark'],
                 execution_benchmark_sha256=numeric_manifest['execution_benchmark_sha256'],
                 comparison='Same current STS-B/en-zh rows/labels/full grids/numeric architecture/seed/epoch/optimizer/median decoder as numeric CDF; only the frozen ASAP objective changes.',
                 source_asap_config=str(objectives/source/'config.json'))
        freeze(name,gpus,cfg,m,'scripts/formal/worker_numeric_continuous_objectives.py','scripts/formal/supervise_numeric_objectives.py')
    atomic(root/'experiment_manifest.json',{'created':created,'python':sys.executable,'arms':arms,
        'gpu_queues':{'0,1':['asap_json_replay'],'4,5':['continuous_r1_cdf_mean','continuous_json_replay'],
                      '6,7':['continuous_r2_crps']},'queue_dependencies':{},'code_sha256':hashes,
        'integrity_audit':str(integrity),'integrity_audit_sha256':sha(integrity),
        'fresh_base_only':True,'parent_checkpoint':None,'epochs':1,'seed':20261101,
        'scope':'User-requested generative rerun plus missing two-source objective results; preserve previous artifacts; no new German jobs or main-model reruns'})
    print(root)


if __name__ == '__main__':
    main()
