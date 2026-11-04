"""Freeze real-valued STS-B/WMT labels and remove training/heldout overlap."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import pyarrow.parquet as pq

from common import PROJECT, atomic, sha
from score_specialist import interpolation


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Components:
    def __init__(self):self.parents={}
    def root(self,x):
        self.parents.setdefault(x,x)
        while self.parents[x]!=x:
            self.parents[x]=self.parents[self.parents[x]];x=self.parents[x]
        return x
    def join(self,entities):
        root=self.root(entities[0])
        for value in entities[1:]:self.parents[self.root(value)]=root


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=PROJECT/'data/score_specialist_continuous_20261104_v1');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False)
    rows=[];inputs={};components=Components()
    stsb=PROJECT/'data/score_scales_stsb_20261101_v1'
    desc=json.loads((stsb/'manifest.json').read_text());raw=stsb/'raw/stsbenchmark.tsv.gz'
    assert sha(raw)==desc['sha256'];inputs[str(raw)]=sha(raw)
    with gzip.open(raw,'rt',encoding='utf-8',newline='') as stream:
        for i,r in enumerate(csv.DictReader(stream,delimiter='\t',quoting=csv.QUOTE_NONE)):
            split={'dev':'dev','train':'train','test':'test_in_domain'}[r['split']]
            a,b=r['sentence1'],r['sentence2'];y=float(r['score'])
            entities=['stsb:sentence:'+digest(norm(s)) for s in (a,b)]
            pair=digest('\n'.join(sorted((norm(a),norm(b)))))
            rows.append({'id':f'stsb/{r["split"]}/{i:07d}','task':'score','source_family':'stsb',
                'source_split':r['split'],'split':split,'label':y,'label_kind':'continuous',
                'state':{'text':f'Sentence A: {a}\nSentence B: {b}'},
                'question':{'type':'score','instructions':'Rate the semantic similarity of the two sentences from 0 (unrelated meaning) to 5 (equivalent meaning). Fractional scores are valid.',
                    'scale':{'min':0.,'max':5.,'step':None,'kind':'continuous'}},
                'score_support':[j/10 for j in range(51)],'candidate_mode':'complete',
                'provenance':{'raw_file_sha256':inputs[str(raw)],'raw_row_index':i,
                              'original_score_text':r['score'],'sid':r['sid'],'dataset':r['dataset']},
                '_entities':entities,'_pair':pair})
            components.join(entities)
    wmt=PROJECT/'data/typed_source_research_20261101_v1'
    audit=json.loads((wmt/'additional_audit.json').read_text())
    for entry in sorted(audit['wmt'],key=lambda r:r['file']):
        p=wmt/entry['path'];assert sha(p)==entry['sha256'];inputs[str(p)]=entry['sha256']
        lang,filename=entry['file'].split('/');original_split=filename.split('-')[0]
        split={'train':'train','validation':'dev','test':'test_in_domain'}[original_split]
        target_lang=lang.split('-')[1]
        for i,r in enumerate(pq.read_table(p).to_pylist()):
            a,b=r['translation']['en'],r['translation'][target_lang];y=float(r['mean'])
            if not 0<=y<=100 or not math.isfinite(y):raise ValueError('Missing WMT target')
            scores=list(map(float,r['scores']))
            if not scores or any(not math.isfinite(v) or not 0<=v<=100 for v in scores):raise ValueError('Invalid annotator scores')
            mean=sum(scores)/len(scores);std=math.sqrt(sum((v-mean)**2 for v in scores)/len(scores))
            entities=['wmt:source:'+digest(norm(a))]
            if r['doc_id']:entities.append('wmt:doc:'+digest(norm(r['doc_id'])))
            rows.append({'id':f'wmt20/{lang}/{original_split}/{r["segid"]}',
                'task':'score','source_family':'wmt_'+lang.replace('-','_'),'source_split':original_split,
                'split':split,'label':y,'label_kind':'continuous','rater_scores':scores,'rater_std':std,
                'state':{'text':f'English source: {a}\n'+('German' if target_lang=='de' else 'Chinese')+f' translation: {b}'},
                'question':{'type':'score','instructions':'Rate how accurately and fluently the translation conveys the source on a scale from 0 to 100. 0-10: incorrect translation; 11-29: few correct keywords; 30-50: major mistakes; 51-69: understandable overall meaning but errors; 70-90: closely preserves meaning; 91-100: near-perfect translation. Fractional scores are valid.',
                    'scale':{'min':0.,'max':100.,'step':None,'kind':'continuous'}},
                'score_support':list(range(101)),'candidate_mode':'complete',
                'provenance':{'raw_file_sha256':entry['sha256'],'raw_row_index':i,'segid':r['segid'],
                              'doc_id':r['doc_id'],'z_mean':float(r['z_mean']),'target':'published mean, not z_mean'},
                '_entities':entities,'_pair':digest(norm(a)+'\n'+norm(b))})
            components.join(entities)
    assert len(rows)==19749+3500+3379 and len({r['id'] for r in rows})==len(rows)
    for row in rows:
        row['group_id']='component:'+digest(components.root(row['_entities'][0]))
        assert abs(sum(p*v for p,v in zip(interpolation(row['score_support'],row['label']),row['score_support']))-row['label'])<1e-8
    raw_counts=collections.Counter((r['split'],r['source_family']) for r in rows)
    held_groups={r['group_id'] for r in rows if r['split']!='train'}
    # Audit exact normalized rendered states against all existing formal splits.
    # These are fresh-base fits; old formal training data are not inherited.
    new_states={digest(norm(r['state']['text'])) for r in rows};old_matches={}
    for oldsplit in ('train','dev','test_in_domain','test_source_holdout'):
        p=PROJECT/'data/formal_200k_20261101_v1'/f'{oldsplit}.jsonl'
        found=[]
        for line in p.open():
            r=json.loads(line)
            if digest(norm(r['state']['text'])) in new_states:found.append(r['id'])
        old_matches[oldsplit]={'rows':len(found),'ids':found,'path':str(p),'sha256':sha(p)}
    kept=[];excluded=[];seen=set()
    for row in rows:
        reason=None;key=(row['source_family'],row['_pair'],row['label'])
        if row['split']=='train':
            if row['group_id'] in held_groups:reason='connected sentence/document group overlaps dev or test'
            elif key in seen:reason='duplicate training input with identical label'
            else:seen.add(key)
        if reason:excluded.append({'id':row['id'],'source':row['source_family'],'reason':reason,'group_id':row['group_id']})
        else:kept.append({k:v for k,v in row.items() if not k.startswith('_')})
    training_groups={r['group_id'] for r in kept if r['split']=='train'}
    assert not training_groups & held_groups
    metadata={}
    for split in ('train','dev','test_in_domain'):
        items=[r for r in kept if r['split']==split];path=out/f'{split}.jsonl'
        with path.open('x') as handle:
            for row in items:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        metadata[split]={'path':str(path),'sha256':sha(path),'rows':len(items),
            'by_source':dict(collections.Counter(r['source_family'] for r in items)),
            'fractional_labels':sum(float(r['label'])!=int(r['label']) for r in items)}
    assert metadata['dev']['rows']==3500 and metadata['test_in_domain']['rows']==3379
    with (out/'exclusions.jsonl').open('x') as handle:
        for row in excluded:handle.write(json.dumps(row)+'\n')
    groups={s:{r['group_id'] for r in kept if r['split']==s} for s in metadata}
    report={'status':'data_frozen_pending_token_audit','purpose':'independent S3/S4 score study; no main D2 modification',
        'inputs_sha256':inputs,'splits':metadata,'raw_counts':{f'{s}/{f}':n for (s,f),n in raw_counts.items()},
        'excluded_training_rows':len(excluded),'exclusion_counts':dict(collections.Counter(r['reason'] for r in excluded)),
        'exclusions_sha256':sha(out/'exclusions.jsonl'),'test_rows_omitted':0,'dev_rows_omitted':0,
        'train_heldout_group_overlap':0,'dev_test_shared_groups':len(groups['dev']&groups['test_in_domain']),
        'old_formal_exact_state_overlap':old_matches,'label_policy':'published real values retained; identical interpolation for both objectives',
        'group_policy':'STS sentence-connected components; WMT shared source sentence/document across language pairs',
        'test_selection':False,'rounded_labels':0,'truncated_rows':0,'seed':20261101,
        'code_sha256':sha(Path(__file__))}
    atomic(out/'manifest.json',report)
    print(json.dumps({'data':str(out),'splits':{s:{k:v[k] for k in ('rows','by_source','fractional_labels')} for s,v in metadata.items()},
                      'excluded_training_rows':len(excluded),'dev_test_shared_groups':report['dev_test_shared_groups']},indent=2),flush=True)


if __name__=='__main__':main()
