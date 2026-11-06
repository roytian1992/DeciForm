"""Freeze small matched Score pilots; reuse full original encodings verbatim."""
import collections
import copy
import datetime
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from common import PROJECT, SEED, atomic, sha
from score_proper import ScoreProperObjective

ROOT=PROJECT/'runs/pilot_score_proper_20261106_v1'
DATA=PROJECT/'data/pilot_score_proper_20261106_v1'
PARENTS=PROJECT/'runs/formal_d2_rlcd_score_20261106_v1'
COUNTS={'continuous':{'train':2048,'dev':256,'test_in_domain':512},
        'asap':{'train':512,'dev':64,'test_in_domain':128}}

def stratified(rows,size,split,panel):
    # Exact labels for ASAP; equal-width ten bins for continuous label coverage.
    buckets=collections.defaultdict(list)
    for i,r in enumerate(rows):
        v=r['score_support'];label=r['label']
        label_key=label if panel=='asap' else min(9,int(10*(label-v[0])/(v[-1]-v[0])))
        buckets[(r['source_family'],label_key)].append(i)
    quotas={}
    families=sorted({k[0] for k in buckets})
    for source in families:
        keys=[k for k in sorted(buckets) if k[0]==source]
        # Continuous: half each source. ASAP: proportional prompt distribution.
        population=sum(len(buckets[k]) for k in keys)
        source_size=size//len(families) if panel=='continuous' else None
        for k in keys:
            denominator=population if panel=='continuous' else len(rows)
            target=source_size if panel=='continuous' else size
            quotas[k]=len(buckets[k])*target//denominator
        if panel=='continuous':
            order=sorted(keys,key=lambda k:(-(len(buckets[k])*source_size%population),k))
            for k in order[:source_size-sum(quotas[k] for k in keys)]:quotas[k]+=1
    if panel=='asap':
        order=sorted(buckets,key=lambda k:(-(len(buckets[k])*size%len(rows)),k))
        for k in order[:size-sum(quotas.values())]:quotas[k]+=1
    selected=[]
    for key,ids in sorted(buckets.items()):
        assert quotas[key]<=len(ids)
        ids=sorted(ids,key=lambda i:hashlib.sha256(f'{SEED}:{panel}:{split}:{rows[i]["id"]}'.encode()).digest())
        selected.extend(ids[:quotas[key]])
    assert len(selected)==size and len(set(selected))==size
    assert {rows[i]['source_family'] for i in selected}==set(families)
    return sorted(selected)

def prepare_data(panel,cfg):
    out=DATA/panel;out.mkdir(parents=True,exist_ok=False);(out/'cache').mkdir()
    source=Path(cfg['data']);audit_path=source/'cache/audit.json';audit=json.loads(audit_path.read_text())
    assert sha(audit_path)==cfg['cache_audit_sha256'] and sha(source/'manifest.json')==audit['data_manifest_sha256']
    manifest={'panel':panel,'seed':SEED,'source_data':str(source),'source_audit_sha256':sha(audit_path),
      'selection':'deterministic source/label-stratified ID hash; equal continuous source quotas, proportional ASAP prompts',
      'candidates':'complete native supports in ALL splits','modified_text':0,'modified_labels':0,'truncated_rows':0,'splits':{}}
    splits={};ids_by_split={};groups_by_split={}
    for split,size in COUNTS[panel].items():
        d=audit['splits'][split];entry=d['arms']['kev']
        assert sha(d['views_path'])==d['views_sha256'] and sha(entry['path'])==entry['sha256']
        rows=[json.loads(line) for line in open(d['views_path'])]
        selected=stratified(rows,size,split,panel);rs=[rows[i] for i in selected]
        with open(entry['path'],'rb') as f:allenc=pickle.load(f)
        assert len(rows)==len(allenc)==d['rows']
        enc=[allenc[i] for i in selected]
        assert all(r['task']=='score' and r['candidate_mode']=='complete' and not e['state_truncated'] and len(e['opt_idx'][0])==len(r['score_support']) for r,e in zip(rs,enc))
        view=out/'cache'/f'{split}.views.jsonl';cache=out/'cache'/f'{split}.kev.pkl'
        with view.open('w') as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        with cache.open('wb') as f:pickle.dump(enc,f,protocol=pickle.HIGHEST_PROTOCOL)
        index=out/f'{split}.indices.json'
        atomic(index,{'source_rows':len(rows),'source_views_sha256':d['views_sha256'],'indices':selected,'ids':[r['id'] for r in rs]})
        lengths=[len(e['ids']) for e in enc]
        splits[split]={'rows':len(rs),'views_path':str(view),'views_sha256':sha(view),
           'native_evaluation':split!='train','complete_output_grid':True,
           'max_json_response_tokens':d['max_json_response_tokens'],
           'arms':{'kev':{'path':str(cache),'sha256':sha(cache),'max_tokens':max(lengths)}}}
        manifest['splits'][split]={'rows':len(rs),'parent_rows':len(rows),'parent_views_sha256':d['views_sha256'],
           'parent_cache_sha256':entry['sha256'],'indices_path':str(index),'indices_sha256':sha(index),
           'sources':dict(collections.Counter(r['source_family'] for r in rs)),
           'grid_sizes':dict(collections.Counter(len(r['score_support']) for r in rs)),
           'max_tokens':max(lengths),'total_encoded_tokens':sum(lengths)}
        ids_by_split[split]={r['id'] for r in rs};groups_by_split[split]={r['group_id'] for r in rs}
        assert len(ids_by_split[split])==len(rs)
    for a,b in [('train','dev'),('train','test_in_domain'),('dev','test_in_domain')]:
        assert not ids_by_split[a]&ids_by_split[b] and not groups_by_split[a]&groups_by_split[b]
    manifest['cross_split_id_group_overlap']=0
    atomic(out/'manifest.json',manifest)
    atomic(out/'cache/audit.json',{'status':'passed','data_manifest_sha256':sha(out/'manifest.json'),'splits':splits})
    return out,manifest

def main():
    assert not (ROOT/'experiment_manifest.json').exists() and not DATA.exists(),'Already prepared'
    ROOT.mkdir(exist_ok=True)
    originals={p:json.loads((PARENTS/p/'experiment_manifest.json').read_text()) for p in COUNTS}
    hashes={}
    for m in originals.values():
        for path,h in m['code_sha256'].items():
            assert sha(PROJECT/path)==h,path
            hashes[path]=h
    additions=['score_proper','models_score_proper','worker_score_proper','supervise_score_proper','prepare_score_proper_pilot','report_score_proper_pilot','queue_score_proper_pilot']
    for path in ['scripts/formal/'+name+'.py' for name in additions]+['tests/test_score_proper.py','scripts/formal/classification_proper.py','scripts/formal/classification_rlcd.py']:
        hashes[path]=sha(PROJECT/path)
    for path in hashes:
        dest=ROOT/'code'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/path,dest)
    arms={};data_manifests={}
    for panel in ['continuous','asap']:
        prior=originals[panel];old=json.loads(Path(prior['config']).read_text())
        data,dm=prepare_data(panel,old);data_manifests[panel]=dm
        for variant in ['cdf','proper']:
            name=panel+'_'+variant;out=ROOT/name;out.mkdir()
            obj=ScoreProperObjective(kind='cdf_reference' if variant=='cdf' else 'direct_log_spherical').inventory()
            cfg=copy.deepcopy(old);cfg.pop('paired_reference',None)
            cfg.update(data=str(data),cache_audit_sha256=sha(data/'cache/audit.json'),output=str(out),
              experiment_name=name,label='Small Score pilot '+name,method_variant='small_score_proper_pilot',
              train_rows=COUNTS[panel]['train'],global_batch=16,total_steps=math.ceil(COUNTS[panel]['train']/16),
              schedule=f"OneCycleLR {math.ceil(COUNTS[panel]['train']/16)} steps, pct_start=0.3",
              distribution_objective=obj,task_losses={'score':obj['kind']},
              objective_description='Same small-data D2 architecture: original CDF-log versus normalized direct log+spherical',
              worker_variant=str(PROJECT/'scripts/formal/worker_score_proper.py'),save_every=16,loss_diagnostics_every=16,
              evaluation_protocol='small_score_proper_'+panel+'_20261106_v1',
              grid_sizes=sorted(int(k) for k in dm['splits']['train']['grid_sizes']),
              evaluation_counts={s:COUNTS[panel][s] for s in ['dev','test_in_domain']},
              order_audit='not applicable; Score is not reordered',parent_checkpoint=None)
            atomic(out/'config.json',cfg)
            m=copy.deepcopy(prior);supervisor=PROJECT/'scripts/formal/supervise_score_proper.py'
            m.update(created=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,gpus='6,7',
              config=str(out/'config.json'),config_sha256=sha(out/'config.json'),worker=cfg['worker_variant'],
              supervisor=str(supervisor),code_sha256=hashes,code_snapshot=str(ROOT/'code'),experiment_name=name,
              distribution_objective=obj,evaluation_protocol=cfg['evaluation_protocol'],
              data_manifest_sha256=sha(data/'manifest.json'),cache_audit_sha256=sha(data/'cache/audit.json'),
              data_counts=COUNTS[panel],scope='Small fresh-base Score loss pilot only; original full-data scores are not matched comparisons',
              comparison='Paired small-data CDF/proper: identical records, full support, base, optimizer, batch16, seed and epoch1',
              git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
              launch_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json')],
              resume_command=[sys.executable,str(supervisor),'--manifest',str(out/'experiment_manifest.json'),'--resume'])
            m['tmux_session']='deciform_score_proper_pilot_20261106'
            atomic(out/'experiment_manifest.json',m)
            arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':'6,7','panel':panel,'variant':variant,'matched_reference':str(ROOT/(panel+'_cdf'))}
    atomic(ROOT/'experiment_manifest.json',{'created':datetime.datetime.now().astimezone().isoformat(),
       'arms':arms,'queue':list(arms),'python':sys.executable,'code_sha256':hashes,'data':str(DATA),
       'data_counts':COUNTS,'global_batch':16,'seed':SEED,'epochs':1,'parent_checkpoint':None,
       'scope':'Small matched Score loss screening; full-data runs not started or queued',
       'note':'Batch16 shared by all arms to allow useful updates at small data size; no direct comparison to full batch64 runs'})
    print(json.dumps({'root':str(ROOT),'counts':COUNTS,'queue':list(arms),'data_manifests':data_manifests},indent=2))

if __name__=='__main__':main()
