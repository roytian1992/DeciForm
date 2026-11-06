"""Freeze inputs, actual checkpoint hashes, protocol and exact launch command."""
import argparse,collections,copy,datetime,json,platform,subprocess,sys
from pathlib import Path
import torch,transformers
from common import PROJECT,SEED,atomic,sha
from zero_shot_logits import RULES,PANELS
from zero_shot_protocol import views_for


def prepare(root):
    if (root/'experiment_manifest.json').exists():raise FileExistsError('Use a new run root or resume its exact manifest')
    preflight=json.loads((root/'preflight.json').read_text())
    assert preflight['status']=='passed'
    original=PROJECT/'runs/formal_zero_shot_instruct_20261105_v3/experiment_manifest.json'
    old=json.loads(original.read_text())
    model=copy.deepcopy(old['models']['instruct'])
    for p,d in model['files_sha256'].items():assert sha(p)==d,p
    model['verified_file_stats']={p:{'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns}
                                  for p in model['files_sha256']}
    panels={}
    for panel in PANELS:
        desc=copy.deepcopy(old['panels'][panel])
        assert sha(desc['views_path'])==desc['views_sha256']
        rows=[json.loads(l) for l in open(desc['views_path'])]
        desc['raw_file_rows']=len(rows)
        if desc.get('task'):rows=[r for r in rows if r['task']==desc['task']]
        if panel=='continuous':
            desc['sources']=['stsb','wmt_en_zh']
            desc['excluded_source_counts']=dict(collections.Counter(r['source_family'] for r in rows if r['source_family'] not in desc['sources']))
            rows=[r for r in rows if r['source_family'] in desc['sources']]
        desc['rows']=len(rows);desc['predictions']=len(rows)*len(views_for(panel))
        desc['records_both_rules']=desc['predictions']*len(RULES)
        desc['selected_source_counts']=dict(collections.Counter(r['source_family'] for r in rows))
        desc['selected_ids_sha256']=__import__('hashlib').sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest()
        panels[panel]=desc
    files=['zero_shot_logits.py','preflight_zero_shot_logits.py','worker_zero_shot_logits.py',
           'prepare_zero_shot_logits.py','schedule_zero_shot_logits.py','zero_shot_protocol.py',
           'common.py','score_baselines.py','score_specialist.py','score_distribution.py',
           'classification_order_audit.py','order_audit.py','typed_losses.py']
    files=['scripts/formal/'+f for f in files]+['tests/test_zero_shot_logits.py']
    hashes={p:sha(PROJECT/p) for p in files}
    for p in files:
        dst=root/'code'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((PROJECT/p).read_bytes())
    manifest={'created':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'model':model,'input_manifest':str(original),'input_manifest_sha256':sha(original),
        'panels':panels,'rules':list(RULES),'primary_rule':'sequence_likelihood',
        'score_decision':'native_median_primary_mean_argmax_mean_nearest_diagnostics',
        'probability_semantics':'canonical full JSON continuation likelihood conditional on the allowed finite support; common-prefix factors cancel; closing delimiter included; no EOS term',
        'secondary_rule':'all valid next tokens locally normalized at every trie node, with no pruning',
        'optimizer_updates':0,'trainable_parameters':0,'training_rows':0,'fewshot_examples':0,
        'no_truncation':True,'no_candidate_sampling':True,'no_gold_in_prompt':True,'no_fitted_calibration':True,
        'python':sys.executable,'cwd':str(PROJECT),'seed':SEED,'shards':4,'gpus':[0,1,2,3],
        'position_batch_policy':'identical row groups across views, sorted by maximum full-view length',
        'batching':{'budget':100000,'max_rows':8,'max_nodes':256},'max_context_tokens':262144,
        'code_sha256':hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'packages':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__},
        'environment_variables':{'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','FLA_DISABLE_TENSOR_CACHE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1'},
        'preflight_sha256':sha(root/'preflight.json'),'tests_sha256':sha(root/'tests.log'),
        'expected_predictions':sum(d['predictions'] for d in panels.values()),
        'expected_records':sum(d['records_both_rules'] for d in panels.values()),
        'launch_command':[sys.executable,str(PROJECT/'scripts/formal/schedule_zero_shot_logits.py'),'--root',str(root)],
        'tmux_session':'deciform_zero_shot_logits_20261106'}
    atomic(root/'experiment_manifest.json',manifest)
    print(json.dumps({'root':str(root),'predictions':manifest['expected_predictions'],'records':manifest['expected_records'],
                      'panels':{k:v['rows'] for k,v in panels.items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root.resolve())
