"""Freeze existing Score data; fresh ModernBERT and random native Laya heads."""
import argparse,collections,datetime,importlib.metadata,json,math,pickle,shutil,subprocess,sys,time
from pathlib import Path
from common import PROJECT,atomic,sha
from laya_score import make_parts


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    origin=PROJECT/'runs/formal_laya_20261106_v1'
    vendor=origin/'vendor/laya_release';base=origin/'base'
    sys.path.insert(0,str(vendor))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    datasets={'asap':PROJECT/'data/score_specialist_asap_fold0_20261104_v1',
              'continuous':PROJECT/'data/score_specialist_stsb_enzh_20261105_v1'}
    all_audits={}
    for panel,data in datasets.items():
        cache=root/'cache'/panel;cache.mkdir(parents=True,exist_ok=True)
        audit={'panel':panel,'data':str(data),'splits':{},'truncated':0,'dropped':0,'target_rounding':False,
               'target_encoding':'one-hot native integers; adjacent-grid linear interpolation for real labels'}
        texts={}
        for split in ('train','dev','test_in_domain'):
            path=data/(split+'.jsonl');rows=[json.loads(l) for l in path.open()]
            assert len({r['id'] for r in rows})==len(rows)
            assert all(r['task']=='score' and r['candidate_mode']=='complete' for r in rows)
            if panel=='continuous':assert all(r['source_family'] in ('stsb','wmt_en_zh') for r in rows)
            parts=[]
            for start in range(0,len(rows),128):
                chunk=rows[start:start+128]
                ids=tok([r['state']['text'].replace(tok.mask_token,' ') for r in chunk],add_special_tokens=False,truncation=False)['input_ids']
                parts.extend(make_parts(r,tok,s,texts) for r,s in zip(chunk,ids))
                atomic(root/'prepare_progress.json',{'panel':panel,'split':split,'completed':len(parts),'total':len(rows),'updated_unix':time.time()})
            dest=cache/(split+'.pkl')
            with dest.open('wb') as f:pickle.dump(parts,f,protocol=5)
            longest=sorted(range(len(rows)),key=lambda i:parts[i]['length'],reverse=True)[:10]
            audit['splits'][split]={'path':str(dest),'sha256':sha(dest),'views_path':str(path),'views_sha256':sha(path),
                'rows':len(rows),'max_tokens':max(p['length'] for p in parts),'tokens':sum(p['length'] for p in parts),
                'longest_indices':longest,'sources':dict(collections.Counter(r['source_family'] for r in rows)),
                'k_counts':dict(collections.Counter(len(p['options']) for p in parts)),
                'over_8192':sum(p['length']>8192 for p in parts)}
        audit['status']='passed';atomic(cache/'audit.json',audit);all_audits[panel]=audit
    files=[PROJECT/'scripts/formal'/p for p in ('common.py','laya_adapter.py','worker_laya.py','laya_score.py',
        'score_distribution.py','score_specialist.py','typed_losses.py','prepare_laya_score.py','worker_laya_score.py',
        'schedule_laya_score.py')]
    files+=[PROJECT/'tests/test_laya_score.py']
    files+=list((vendor/'laya').rglob('*.py'))
    hashes={str(p):sha(p) for p in files}
    for p in files:
        dest=root/'code'/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    base_hashes={str(p):sha(p) for p in base.iterdir() if p.is_file()}
    queues={}
    for layout,gpus in [('sequential','0,1'),('parallel','2,3')]:
        queues[layout]=[]
        for panel in datasets:
            name=layout+'_'+panel;out=root/name;out.mkdir(exist_ok=True)
            audit=all_audits[panel];n=audit['splits']['train']['rows']
            cfg={'root':str(root),'output':str(out),'vendor':str(vendor),'base':str(base),'layout':layout,'panel':panel,
                 'parent_checkpoint':None,'initialization':'pretrained ModernBERT encoder; random Laya decision head',
                 'encoder_revision':'45bb4654a4d5aaff24dd11d4781fa46d39bf8c13','base_weights_sha256':base_hashes[str(base/'model.safetensors')],
                 'seed':20261101,'epochs':1,'train_rows':n,'global_batch':64,'total_steps':math.ceil(n/64),'world_size':2,
                 'micro_batch':8,'eval_micro_batch':16,'token_budget':16384,'eval_token_budget':16384,
                 'encoder_lr':2.5e-5,'head_lr':1e-4,'min_lr':1e-6,'save_every':40,
                 'objective':'unchanged upstream RLCD sigma=.4 samples=4 spherical=.75 RPS=1; soft targets for fractional labels',
                 'cache_audit':str(root/'cache'/panel/'audit.json'),'cache_audit_sha256':sha(root/'cache'/panel/'audit.json'),
                 'max_len':max(s['max_tokens'] for s in audit['splits'].values()),'score_decision_rule':'median',
                 'full_candidates':True,'no_truncation':True,'no_target_rounding':True,'no_posthoc_calibration':True}
            atomic(out/'config.json',cfg)
            queues[layout].append({'name':name,'gpus':gpus,'config':str(out/'config.json'),'config_sha256':sha(out/'config.json')})
    manifest={'created':datetime.datetime.now().astimezone().isoformat(),'python':sys.executable,'project':str(PROJECT),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','safetensors')},
        'base_files_sha256':base_hashes,'code_sha256':hashes,'queues':queues,
        'parent_checkpoint':None,'formal_200k_checkpoint_used':False,'released_laya_decision_weights_used':False,
        'environment_variables':{'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','USE_TF':'0','OMP_NUM_THREADS':'4'},
        'comparison':'same frozen Score data, one epoch, global batch64, native median; native Laya backbone, full finetuning, objective and optimizer retained',
        'data':all_audits,'launch_command':[sys.executable,str(PROJECT/'scripts/formal/schedule_laya_score.py'),'--root',str(root)]}
    atomic(root/'experiment_manifest.json',manifest)
    print(json.dumps({'root':str(root),'queues':queues,'counts':{p:{s:d['rows'] for s,d in a['splits'].items()} for p,a in all_audits.items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root.resolve())
