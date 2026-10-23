"""Prepare only unused rows, checking every inherited training stage."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / 'src'))
from deciform.joint_data import atomic_json, file_sha256, index_source
from deciform.evaluation import load_inventory, text_key

def read(path):
    return json.loads(Path(path).read_text())

def verified(spec):
    p = Path(spec['path'])
    if file_sha256(p) != spec['sha256']:
        raise ValueError(f'changed file: {p}')
    with p.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != spec['count']:
        raise ValueError(f'changed count: {p}')
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit_path = out / 'audit.json'
    atomic_json(audit_path, {'status': 'running', 'training_launched': False})
    preparation = read(PROJECT / 'docs/four_source_supplement_audit_20261023.json')
    test_path = Path(preparation['test_manifest'])
    test, test_rows, _ = load_inventory(test_path)
    test_ids = {r['id'] for r in test_rows}
    test_texts = {text_key(r) for r in test_rows}
    candidates = {s['source']: verified(s['train']) for s in preparation['sources']}
    candidate_ids = {r['id'] for rows in candidates.values() for r in rows}
    candidate_texts = {text_key(r) for rows in candidates.values() for r in rows}
    parent = PROJECT / 'runs/joint_static_delta_choice_4gpu_v3_recovered_seed20261015/checkpoints/step_00007153'
    for relative, digest in read(parent/'hashes.json').items():
        if file_sha256(parent/relative) != digest:
            raise ValueError(f'changed parent checkpoint: {relative}')
    progress = read(parent/'progress.json')
    if progress['status'] != 'complete' or progress['sample_visits'] != progress['expected_sample_visits']:
        raise ValueError('parent not complete')
    ancestry, directory, seen = [], parent.parent.parent, set()
    while directory not in seen:
        seen.add(directory)
        path = directory/'run_manifest.json'
        manifest = read(path)
        ancestry.append({'path': str(path), 'sha256': file_sha256(path),
                         'parent_checkpoint': manifest.get('parent_checkpoint'),
                         'runtime_migration': manifest.get('runtime_migration')})
        ancestor = (manifest.get('runtime_migration') or {}).get('checkpoint') or manifest.get('parent_checkpoint')
        if not ancestor:
            theta_path = WORKSPACE/manifest['config']['theta0_config']
            break
        directory = Path(ancestor).parent.parent
    else:
        raise ValueError('cyclic checkpoint lineage')
    theta = read(theta_path)
    for key, hash_key in [('initial_run_report','initial_run_report_sha256'),
                          ('base_protocol_manifest','base_protocol_manifest_sha256'),
                          ('head_checkpoint','head_checkpoint_sha256')]:
        if file_sha256(WORKSPACE/theta[key]) != theta[hash_key]:
            raise ValueError(f'initialization changed: {key}')
    base_manifest = read(WORKSPACE/theta['base_protocol_manifest'])
    base_rows = verified(base_manifest['outputs']['train'])
    initial_counts = Counter(r.get('provenance',{}).get('rule_id', r.get('source','unknown')) for r in base_rows)
    historical_specs = []
    reserved_specs = list(preparation['reserved_files'])
    reserved_specs.extend(base_manifest['outputs'][k] for k in ('dev','cal'))
    for stage, ref in test['training_plans'].items():
        p = Path(ref['path'])
        if file_sha256(p) != ref['sha256']:
            raise ValueError('history plan hash changed')
        for source in read(p)['sources']:
            historical_specs.append({'stage': stage, **source})
            ep = read(source['episode_manifest'])
            for role, spec in ep.get('outputs',{}).items():
                if role != 'D_train' and 'path' in spec and 'sha256' in spec and 'count' in spec:
                    reserved_specs.append(spec)
    historical_specs.append({'stage':'initialization', 'name':'base_train', **base_manifest['outputs']['train']})
    reserved_ids, reserved_texts = set(test_ids), set(test_texts)
    seen_paths = set()
    for spec in reserved_specs:
        if spec['path'] in seen_paths: continue
        seen_paths.add(spec['path'])
        for row in verified(spec):
            reserved_ids.add(row['id']); reserved_texts.add(text_key(row))
    def scan(spec):
        digest, count = hashlib.sha256(), 0
        ids, texts, test_id_hits, test_text_hits = set(), set(), set(), set()
        with Path(spec['path']).open('rb') as handle:
            for line in handle:
                digest.update(line)
                if not line.strip():continue
                row = json.loads(line); count += 1
                key = text_key(row)
                if row['id'] in candidate_ids:ids.add(row['id'])
                if key in candidate_texts:texts.add(key)
                if row['id'] in test_ids:test_id_hits.add(row['id'])
                if key in test_texts:test_text_hits.add(key)
        if count != spec['count'] or digest.hexdigest() != spec['sha256']:
            raise ValueError(f"history changed: {spec['path']}")
        summary = {'stage':spec['stage'],'source':spec['name'],'path':spec['path'],
                   'sha256':digest.hexdigest(),'count':count,'candidate_id_hits':len(ids),
                   'candidate_text_hits':len(texts),'test_id_hits':len(test_id_hits),
                   'test_text_hits':len(test_text_hits)}
        print(json.dumps(summary), flush=True)
        return summary, ids, texts, test_id_hits, test_text_hits
    history_ids, history_texts, test_id_hits, test_text_hits = set(), set(), set(), set()
    scans = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for summary, ids, texts, ti, tt in executor.map(scan, historical_specs):
            scans.append(summary);history_ids.update(ids);history_texts.update(texts)
            test_id_hits.update(ti);test_text_hits.update(tt)
    affected_test_ids = [r['id'] for r in test_rows if r['id'] in test_id_hits or text_key(r) in test_text_hits]
    report = {'status':'history_scanned','created_at':datetime.datetime.now().astimezone().isoformat(),
              'parent_checkpoint':str(parent),'parent_hashes_sha256':file_sha256(parent/'hashes.json'),
              'ancestry':ancestry,'theta0_config':str(theta_path),'theta0_config_sha256':file_sha256(theta_path),
              'initialization_training_unique_rows':len(base_rows),'initialization_family_counts':dict(initial_counts),
              'history_sources':scans,'training_rows_scanned':sum(s['count'] for s in scans),
              'test_manifest':str(test_path),'test_manifest_sha256':file_sha256(test_path),
              'test_records':len(test_rows),'affected_test_ids':affected_test_ids,
              'test_history_overlap_count':len(affected_test_ids),
              'normalization':'Unicode casefold plus whitespace collapse of state.text',
              'prior_training_unique_rows_not_sample_visits':True,
              'raw_candidate_rows':sum(map(len,candidates.values())),'sources':[],
              'training_launched':False}
    atomic_json(audit_path,report)
    if affected_test_ids:
        raise ValueError('initialization/history touches locked test; inspect audit before scheduling')
    protocol = out/'protocol'; protocol.mkdir()
    (out/'indices').mkdir()
    all_ids, all_texts = set(), set()
    for source, rows in candidates.items():
        source_dir = protocol/source;source_dir.mkdir()
        excluded, count = [], 0
        destination = source_dir/'D_train.jsonl'
        with destination.open('x') as handle:
            for row in rows:
                key = text_key(row);reasons=[]
                for match, reason in [(row['id'] in history_ids,'historical_id'),
                        (key in history_texts,'historical_text'),(row['id'] in reserved_ids,'reserved_id'),
                        (key in reserved_texts,'reserved_text'),(row['id'] in all_ids,'candidate_duplicate_id'),
                        (key in all_texts,'candidate_duplicate_text')]:
                    if match:reasons.append(reason)
                if reasons:excluded.append({'id':row['id'],'reasons':reasons});continue
                handle.write(json.dumps(row,ensure_ascii=False)+chr(10));count+=1
                all_ids.add(row['id']);all_texts.add(key)
        source_input = next(s for s in preparation['sources'] if s['source']==source)
        source_manifest=read(source_input['protocol_manifest'])
        if count==0:raise ValueError(f'no unused samples remain for {source}')
        atomic_json(source_dir/'manifest.json',{'version':1,'status':'complete','source':source,
            'max_prompt_tokens':None,'option_count':source_manifest['option_count'],
            'parent_manifest':source_input['protocol_manifest'],
            'parent_manifest_sha256':source_input['protocol_manifest_sha256'],
            'outputs':{'D_train':{'path':str(destination),'count':count,'sha256':file_sha256(destination)}},
            'excluded_rows':excluded,'upstream_filtering':source_manifest.get('filtering',{})})
        report['sources'].append({'source':source,'raw_rows':len(rows),'count':count,'excluded_rows':excluded})
    report.update(status='pass',retained_rows=sum(s['count'] for s in report['sources']),
                  historical_training_record_overlap_checked=True,reserved_file_count=len(seen_paths))
    atomic_json(audit_path,report)
    atomic_json(protocol/'manifest.json',{'version':1,'status':'complete','max_prompt_tokens':None,
        'source_count':len(candidates),'total_rows':report['retained_rows'],
        'audit':str(audit_path),'audit_sha256':file_sha256(audit_path),
        'episodes':{s:{'manifest_sha256':file_sha256(protocol/s/'manifest.json')} for s in candidates}})
    validation_path=out/'validation_plan.json'
    atomic_json(validation_path,{'version':1,'P_train':sorted(candidates),'P_dev':[],'P_assess_locked':[],
        'training_role':'D_train only','reserved_roles':['D_mv','D_cal_fit','D_cal_risk','D_assess_visible'],
        'test_manifest':str(test_path),'test_manifest_sha256':file_sha256(test_path),
        'selection':'fixed one-pass continuation, no test-based checkpoint selection'})
    sources=[index_source(s,protocol/s,out/'indices') for s in sorted(candidates)]
    parent_manifest=read(parent.parent.parent/'run_manifest.json')
    plan={'version':1,'status':'complete','scope':'registered_continuation',
        'protocol_root':str(protocol),'protocol_manifest_sha256':file_sha256(protocol/'manifest.json'),
        'validation_plan':str(validation_path),'validation_plan_sha256':file_sha256(validation_path),
        'source_count':len(sources),'total_rows':sum(s['count'] for s in sources),'sources':sources,
        'sampling':'global_permutation_without_replacement','training_role':'D_train',
        'excluded_roles':['D_mv','D_cal_fit','D_cal_risk','D_assess_visible'],
        'parent_checkpoint':str(parent),'parent_plan_sha256':parent_manifest['plan_sha256'],
        'parent_run_manifest_sha256':file_sha256(parent.parent.parent/'run_manifest.json'),
        'audit':str(audit_path),'audit_sha256':file_sha256(audit_path)}
    atomic_json(out/'plan.json',plan)
    config=dict(parent_manifest['config'])
    config.update(name='joint_static_four_source_20261023_v1',joint_plan=str((out/'plan.json').relative_to(WORKSPACE)),
                  expected_train_rows=plan['total_rows'])
    atomic_json(out/'training_config.json',config)
    isolation={'status':'pass','test_manifest':str(test_path),'test_manifest_sha256':file_sha256(test_path),
        'test_records':len(test_rows),'parent_checkpoint':str(parent),
        'continuation_plan':str(out/'plan.json'),'continuation_plan_sha256':file_sha256(out/'plan.json'),
        'history_audit':str(audit_path),'history_audit_sha256':file_sha256(audit_path),
        'historical_test_overlaps':0,'new_training_test_overlaps':0,'initialization_included':True}
    atomic_json(out/'evaluation_isolation_audit.json',isolation)
    print(json.dumps({'status':'complete','retained_rows':plan['total_rows'],
                      'plan':str(out/'plan.json'),'config':str(out/'training_config.json')}),flush=True)

if __name__=='__main__':
    main()
