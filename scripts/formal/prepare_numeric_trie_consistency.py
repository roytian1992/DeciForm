"""Prepare a second, predeclared numeric-trie consistency study."""
import copy, json, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from common import PROJECT, atomic, sha
from models_score_distribution import A4_FLAGS


def main():
    root = PROJECT/'runs/formal_numeric_trie_consistency_20261106_v4'
    if root.exists(): raise FileExistsError(root)
    previous_root = PROJECT/'runs/formal_numeric_trie_20261106_v1'
    specs = {
      'asap': {'data': PROJECT/'data/score_numeric_trie_20261106_v1/asap/audit.json', 'counts': {'train':7784,'dev':2593,'test_in_domain':2598}, 'steps':122, 'budget':8192, 'micro':4},
      'continuous': {'data': PROJECT/'data/score_numeric_trie_20261106_v1/continuous/audit.json', 'counts': {'train':8129,'dev':2500,'test_in_domain':2379}, 'steps':128, 'budget':16384, 'micro':16},
    }
    # Build from the already frozen v1 configs; this keeps prompt, optimizer,
    # batch, and evaluation semantics byte-for-byte comparable.
    base_configs = {p: json.loads((previous_root/(p+'_n1_trie_ce')/'config.json').read_text()) for p in specs}
    base_manifests = {p: json.loads((previous_root/(p+'_n1_trie_ce')/'experiment_manifest.json').read_text()) for p in specs}
    # Audit actual decimal targets before GPU allocation, preserving precision.
    from transformers import AutoTokenizer
    from models_numeric_trie_consistency import suffix_tokens
    tokenizer = AutoTokenizer.from_pretrained(base_configs['asap']['base'],local_files_only=True)
    suffix_audit = {}
    for panel,spec in specs.items():
        audit = json.loads(spec['data'].read_text())
        rows = [json.loads(line) for line in open(audit['splits']['train']['views_path'])]
        lengths = [len(suffix_tokens(row,tokenizer)[0]) for row in rows]
        suffix_audit[panel] = {'rows':len(rows),'max_tokens':max(lengths),'exact_json_serialization':True}
    root.mkdir()
    atomic(root/'suffix_audit.json',suffix_audit)
    additions = ['numeric_trie.py','models_numeric_trie.py','models_numeric_trie_consistency.py',
                 'worker_numeric_trie_consistency_asap.py','worker_numeric_trie_consistency_continuous.py',
                 'supervise_numeric_trie.py','prepare_numeric_trie_consistency.py','schedule_research_queue.py']
    paths = set().union(*(m['code_sha256'] for m in base_manifests.values())) | {'scripts/formal/'+p for p in additions} | {'tests/test_numeric_trie_consistency.py'}
    for m in base_manifests.values():
        for p,d in m['code_sha256'].items():
            if sha(PROJECT/p) != d: raise ValueError('Reference dependency changed: '+p)
    hashes = {p: sha(PROJECT/p) for p in sorted(paths)}
    for p in paths:
        target=root/'code'/p; target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/p,target)
    arms, queues = {}, {}
    created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    runs=[('asap','asap_token_ce','0,1','trie_token_ce',0.),('asap','asap_token_ce_cdf','2,3','trie_token_ce_cdf',.5),
          ('continuous','continuous_token_ce','4,5','trie_token_ce',0.),('continuous','continuous_token_ce_cdf','6,7','trie_token_ce_cdf',.5)]
    for panel,name,gpus,kind,cdf_weight in runs:
        out=root/name;out.mkdir()
        cfg=copy.deepcopy(base_configs[panel]); data=specs[panel]['data']
        cfg.update(output=str(out),label=name,experiment_name=name,
          distribution_objective={'kind':kind,'ordinal_scale':3.0383850790577127,'cdf_weight':cdf_weight,'token_weight':.5},
          task_losses={'score':kind},worker_variant=str(PROJECT/f'scripts/formal/worker_numeric_trie_consistency_{panel}.py'),
          method_variant='numeric_trie_joint_token_supervision',score_architecture='numeric_trie_joint_token_nll',shared_causal_forward=True,
          score_head='Frozen pretrained root-digit logits + compact decimal trie; JSON numeric suffix teacher-forcing auxiliary',
          objective_description='Full-support CE + 0.5 token NLL' if not cdf_weight else '0.5 CE + 0.5 CDF-log + 0.5 token NLL',
          cache_audit_path=str(data),cache_audit_sha256=sha(data),numeric_anchor_path=str(PROJECT/'data/score_numeric_trie_20261106_v1/numeric_anchors.pt'),
          numeric_anchor_sha256=sha(PROJECT/'data/score_numeric_trie_20261106_v1/numeric_anchors.pt'),
          paired_reference=str(previous_root/(panel+'_n1_trie_ce')),
          auxiliary_loss='Teacher-forced exact JSON decimal suffix; label-dependent only inside the auxiliary training path; omitted at inference')
        atomic(out/'config.json',cfg)
        m=copy.deepcopy(base_manifests[panel]); m.update(created=created,config=str(out/'config.json'),config_sha256=sha(out/'config.json'),
          worker=cfg['worker_variant'],supervisor=str(PROJECT/'scripts/formal/supervise_numeric_trie.py'),gpus=gpus,code_sha256=hashes,code_snapshot=str(root/'code'),
          experiment_name=name,distribution_objective=cfg['distribution_objective'],expected_trainable_parameters=46161164,
          numeric_anchor_sha256=cfg['numeric_anchor_sha256'],paired_reference=cfg['paired_reference'],
          tmux_session='deciform_numeric_trie_consistency_v4_g'+gpus.replace(',','')+'_20261106',
          launch_command=[sys.executable,str(PROJECT/'scripts/formal/supervise_numeric_trie.py'),'--manifest',str(out/'experiment_manifest.json')],
          resume_command=[sys.executable,str(PROJECT/'scripts/formal/supervise_numeric_trie.py'),'--manifest',str(out/'experiment_manifest.json'),'--resume'],
          comparison='Same v1 numeric-trie path and data; adds a predeclared exact-score token-NLL auxiliary for the backbone. Fresh Base, no parent checkpoint.',
          git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        m.pop('execution_benchmark',None);m.pop('execution_benchmark_sha256',None)
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'objective':cfg['distribution_objective']};queues[gpus]=[name]
    atomic(root/'experiment_manifest.json',{'created':created,'python':sys.executable,'arms':arms,'gpu_queues':queues,'queue_dependencies':{'4,5':str(PROJECT/'runs/formal_laya_20261106_v1/sequential/status.json'),'6,7':str(PROJECT/'runs/formal_laya_20261106_v1/parallel/status.json')},'code_sha256':hashes,'fresh_base_only':True,'parent_checkpoint':None,'epochs':1,'global_batch':64,'main_d2_unchanged':True,'criterion':'A common mechanism must lower native test MAE on ASAP, STS-B and WMT en-zh versus fresh JSON replay. No per-source cherry-picking; single-seed evidence is not cross-seed stability.','scope':'Numeric trie plus exact score-token teacher-forcing supervision in one shared causal forward; two predeclared loss variants on two panels.'})
    print(json.dumps({'root':str(root),'arms':arms},indent=2))

if __name__=='__main__': main()
