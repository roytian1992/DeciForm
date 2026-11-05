"""Freeze evidence pooling, median-distribution and combined ASAP experiments."""
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import BASE, PROJECT, atomic, sha
from asap_research_modules import EssayObjective, QueryEvidencePool, MedianLaplaceReadout
from numeric_score_distribution import NumericCDFReadout


def main():
    root = PROJECT/'runs/formal_asap_exploration_20261106_v1'
    reference = PROJECT/'runs/formal_score_numeric_asap_20261105_v1'
    previous = PROJECT/'runs/formal_score_objectives_20261106_v1/r1_cdf_mean'
    original = json.loads((previous/'config.json').read_text())
    manifest = json.loads((previous/'experiment_manifest.json').read_text())
    base_cfg = json.loads((BASE/'config.json').read_text())
    hidden_size = base_cfg['text_config']['hidden_size']
    query_size = 256
    count = lambda m: sum(p.numel() for p in m.parameters())
    numeric_count = json.loads((reference/'initialization.json').read_text())['trainable_parameters']
    pool_count = count(QueryEvidencePool(hidden_size,query_size))
    replacement_delta = count(MedianLaplaceReadout(query_size))-count(NumericCDFReadout(query_size))
    data = Path(original['data'])
    audit = json.loads((data/'cache/audit.json').read_text())
    assert sha(data/'cache/audit.json') == original['cache_audit_sha256']
    assert {s:d['rows'] for s,d in audit['splits'].items()} == {'train':7784,'dev':2593,'test_in_domain':2598}
    additions = ['asap_research_modules.py','models_asap_exploration.py','worker_asap_exploration.py',
                 'supervise_asap_exploration.py','prepare_asap_exploration.py','schedule_research_queue.py']
    paths = set(manifest['code_sha256']) | {'scripts/formal/'+p for p in additions} | {'tests/test_asap_research_modules.py'}
    for p,digest in manifest['code_sha256'].items():
        assert sha(PROJECT/p) == digest, 'Existing reference dependency changed: '+p
    hashes = {p:sha(PROJECT/p) for p in sorted(paths)}
    root.mkdir(exist_ok=False)
    for p in paths:
        target = root/'code'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/p,target)
    specs = [('e1_evidence_cdf','4,5','evidence_cdf',numeric_count+pool_count),
             ('e2_median_laplace','6,7','median_laplace',numeric_count+replacement_delta),
             ('e3_evidence_median_laplace','2,3','evidence_median_laplace',numeric_count+pool_count+replacement_delta)]
    arms,queues = {},{}
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    for name,gpus,kind,parameters in specs:
        out=root/name;out.mkdir()
        objective=EssayObjective(kind,median_weight=0. if kind=='evidence_cdf' else 1.)
        cfg=copy.deepcopy(original)
        cfg.update(output=str(out),label='ASAP exploration: '+name,experiment_name=name,
            distribution_objective=objective.inventory(),task_losses={'score':kind},
            worker_variant=str(PROJECT/'scripts/formal/worker_asap_exploration.py'),
            method_variant='asap_evidence_median_research',score_architecture=kind,
            score_head=('adaptive CDF with full-prefix query-conditioned evidence pooling' if kind=='evidence_cdf'
                        else 'latent normalized median and scale; Laplace interval masses over all native levels'),
            objective_description=('unchanged CDF-log' if kind=='evidence_cdf' else 'CDF-log + normalized latent-median absolute error, lambda=1'),
            paired_reference=str(reference),max_microbatch_rows=4,token_budget=8192,eval_token_budget=8192)
        assert cfg['parent_checkpoint'] is None and cfg['train_rows']==7784 and cfg['epochs']==1 and cfg['total_steps']==122
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(manifest)
        supervisor=str(PROJECT/'scripts/formal/supervise_asap_exploration.py')
        m.update(created=created,python=sys.executable,config=str(out/'config.json'),config_sha256=sha(out/'config.json'),
            worker=cfg['worker_variant'],supervisor=supervisor,gpus=gpus,code_sha256=hashes,code_snapshot=str(root/'code'),
            experiment_name=name,distribution_objective=objective.inventory(),expected_trainable_parameters=parameters,
            numeric_initialization_reference=str(reference/'initialization.json'),paired_reference=str(reference),
            tmux_session='deciform_asap_explore_g'+gpus.replace(',','')+'_20261106',
            launch_command=[sys.executable,supervisor,'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,supervisor,'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            comparison='Same native ASAP data/order/seed/epoch/optimizer/batching/decoder; new module and explicitly registered objective. No historical checkpoint inheritance.',
            execution_benchmark_scope='Each exact new model must pass zero-update distributed longest-input, dense-batch and first-global-batch backward before any optimizer update.',
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        m.pop('execution_benchmark',None);m.pop('execution_benchmark_sha256',None)
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'parameters':parameters,'objective':objective.inventory()}
        queues[gpus]=[name]
    atomic(root/'experiment_manifest.json',{'created':created,'python':sys.executable,'arms':arms,'gpu_queues':queues,
        'queue_dependencies':{},'data':str(data),'data_counts':{'train':7784,'dev':2593,'test_in_domain':2598},
        'data_manifest_sha256':sha(data/'manifest.json'),'cache_audit_sha256':original['cache_audit_sha256'],
        'code_sha256':hashes,'fresh_base_only':True,'parent_checkpoint':None,'epochs':1,'global_batch':64,'steps':122,
        'main_d2_unchanged':True,'shared_initialization_reference':str(reference),
        'scope':'ASAP-only separate study: E1 representation, E2 median-distribution, E3 combination. No additional data or repeated epochs.'})
    print(json.dumps({'root':str(root),'arms':arms},indent=2))


if __name__=='__main__':
    main()
