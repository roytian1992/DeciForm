"""Reconstruct the preregistered author fold from original essays and full rubrics."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from common import PROJECT,atomic,sha
from asap_score import RANGES

ORIGINAL_SHA='cf04127eaff3224ce4aab64654e62e5738d3d672c61bb998546dd1674c60d312'
DESCRIPTIONS_SHA='1ee7dac2284c983fc9e4aaa8ba915b079d5d7b67e2624d58ec0c35652f7c2211'
FOLD_SHA={'train':'3e048c1f8de0e8b2ccfc71524fdaffb3f3d2a909e0cdedc7c5d1a5aa129ab7ac',
          'dev':'4e03bb0a5c42b0c97aee01feee3a3190a1981c926919faab7be44b0531626028',
          'test':'f2a9b55496ef36e45cdb0260eb004f715ae62013980edfc2f463b15fcb8fd296'}


def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def paragraphs(docx):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(io.BytesIO(docx)) as z:root=ET.fromstring(z.read('word/document.xml'))
    result=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in root.findall('.//w:p',ns)]
    return [v for v in result if v.strip()]


def instructions(parts,number):
    """Remove only introductory corpus metadata; retain all task/rubric paragraphs."""
    marker='Source Essay' if number in (3,4,5,6) else 'Prompt'
    start=parts.index(marker)
    assert 'Prompt' in parts[start:] and any(s.startswith('Rubric Guidelines') for s in parts[start:]) and 'Adjudication Rules' in parts[start:]
    grade=parts[parts.index('Grade level:')+1]
    lo,hi=RANGES[number]
    domain=' For Set 2, predict Domain 1 (Writing Applications), not Domain 2.' if number==2 else ''
    return (f'Assess the student essay for ASAP Essay Set {number}, grade {grade}. '
        f'Predict its final resolved domain1_score as an integer from {lo} to {hi}, inclusive. '
        'Use the original prompt, rubric and resolution rules below to assess the essay; '
        'the target is the final resolved score, not an individual rater score.'+domain+'\n\n'+'\n'.join(parts[start:]))


def read_tsv(path,encoding,errors='strict'):
    with path.open(encoding=encoding,errors=errors,newline='') as f:
        return list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--intake',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=PROJECT/'data/score_specialist_asap_fold0_20261104_v1')
    args=ap.parse_args();intake=args.intake.resolve();out=args.output.resolve()
    original=intake/'training_set_rel3.tsv';descs=intake/'Essay_Set_Descriptions.zip'
    assert sha(original)==ORIGINAL_SHA and sha(descs)==DESCRIPTIONS_SHA
    # Match the author's published preprocess_asap.py exactly. Raw bytes are
    # retained and every undefined byte/replacement is explicitly audited below.
    source=read_tsv(original,'cp1252',errors='replace');mapping={r['essay_id']:r for r in source}
    assert len(mapping)==len(source)==12978
    decoding={'policy':'author cp1252 decode with errors=replace; no additional text cleaning',
        'replacement_characters':sum(r['essay'].count('\ufffd') for r in source),
        'affected_essay_ids':[r['essay_id'] for r in source if '\ufffd' in r['essay']],
        'raw_bytes_preserved_sha256':ORIGINAL_SHA}
    input_hashes={str(original):sha(original),str(descs):sha(descs)}
    split_ids={};author_checks={}
    for split in ('train','dev','test'):
        path=intake/f'{split}_ids.txt';assert sha(path)==FOLD_SHA[split];input_hashes[str(path)]=sha(path)
        ids=path.read_text().splitlines();assert len(ids)==len(set(ids)) and set(ids)<=set(mapping)
        split_ids[split]=ids
        author=intake/f'fold0_{split}.tsv'
        if author.exists():
            author_rows=read_tsv(author,'utf-8')
            assert [r['essay_id'] for r in author_rows]==ids
            for r in author_rows:
                native=mapping[r['essay_id']]
                assert r['essay']==native['essay'] and r['essay_set']==native['essay_set'] and r['domain1_score']==native['domain1_score'],r['essay_id']
            input_hashes[str(author)]=sha(author);author_checks[split]={'rows':len(ids),'exact_essay_and_target_match':True}
    assert sum(map(len,split_ids.values()))==len(source)
    assert set().union(*map(set,split_ids.values()))==set(mapping)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not set(split_ids[a])&set(split_ids[b])
    prompt_inventory={};raw_rows=[]
    with ZipFile(descs) as z:
        for number in range(1,9):
            entry=f'Essay_Set_Descriptions/Essay Set #{number}--ReadMeFirst.docx'
            raw=z.read(entry);parts=paragraphs(raw);text=instructions(parts,number)
            prompt_inventory[str(number)]={'instructions':text,'source_entry':entry,
                'docx_sha256':hashlib.sha256(raw).hexdigest(),'instruction_sha256':digest(text),
                'original_paragraphs':parts,'complete_prompt_rubric_and_source':True,
                'native_min':RANGES[number][0],'native_max':RANGES[number][1]}
    for original_split,ids in split_ids.items():
        split={'test':'test_in_domain'}.get(original_split,original_split)
        for essay_id in ids:
            r=mapping[essay_id];number=int(r['essay_set']);y=int(r['domain1_score']);lo,hi=RANGES[number]
            assert lo<=y<=hi and r['essay'].strip()
            raw_rows.append({'id':f'asap/{essay_id}','essay_id':essay_id,'essay_set':number,
                'task':'score','source_family':f'asap_set_{number}','split':split,
                'source_split':original_split,'label':y,'label_kind':'native_integer',
                'state':{'text':r['essay']},'question':{'type':'score',
                    'instructions':prompt_inventory[str(number)]['instructions'],
                    'scale':{'min':lo,'max':hi,'step':1}},'score_support':list(range(lo,hi+1)),
                'candidate_mode':'complete','group_id':'essay_text:'+digest(normalized(r['essay'])),
                'provenance':{'original_tsv_sha256':ORIGINAL_SHA,'author_fold_revision':'7a89a331be1f81871764b3eb4257f6732cabe9fb',
                    'instruction_sha256':prompt_inventory[str(number)]['instruction_sha256'],
                    'target_column':'domain1_score','essay_sha256':digest(r['essay'])}})
    held={r['group_id'] for r in raw_rows if r['split']!='train'};seen=set();kept=[];excluded=[]
    for r in raw_rows:
        reason=None;key=(r['essay_set'],r['group_id'],r['label'])
        if r['split']=='train':
            if r['group_id'] in held:reason='essay text overlaps dev/test'
            elif key in seen:reason='duplicate training essay with same label'
            else:seen.add(key)
        if reason:excluded.append({'id':r['id'],'source':r['source_family'],'reason':reason,'group_id':r['group_id']})
        else:kept.append(r)
    assert not {r['group_id'] for r in kept if r['split']=='train'}&held
    out.mkdir(exist_ok=False);atomic(out/'prompt_inventory.json',prompt_inventory);metadata={}
    for split in ('train','dev','test_in_domain'):
        items=[r for r in kept if r['split']==split];path=out/f'{split}.jsonl'
        with path.open('x') as f:
            for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        assert set(r['essay_set'] for r in items)==set(range(1,9))
        metadata[split]={'path':str(path),'sha256':sha(path),'rows':len(items),
                         'by_source':dict(collections.Counter(r['source_family'] for r in items))}
    atomic(out/'exclusions.json',excluded)
    groups={s:{r['group_id'] for r in kept if r['split']==s} for s in metadata}
    raw_counts={s:sum(r['split']==s for r in raw_rows) for s in metadata}
    # Exact essay matching against old formal states, separate from split audit.
    texts={normalized(r['state']['text']) for r in raw_rows};old_matches={}
    for split in ('train','dev','test_in_domain','test_source_holdout'):
        path=PROJECT/'data/formal_200k_20261101_v1'/f'{split}.jsonl'
        found=[r['id'] for r in map(json.loads,path.open()) if normalized(r['state']['text']) in texts]
        old_matches[split]={'sha256':sha(path),'exact_state_ids':found}
    manifest={'status':'data_frozen_pending_token_audit','inputs_sha256':input_hashes,
        'original_rows':len(source),'raw_counts':raw_counts,'splits':metadata,
        'author_fold':0,'author_copy_verification':author_checks,'source_encoding':decoding,
        'excluded_training_rows':len(excluded),'exclusion_counts':dict(collections.Counter(r['reason'] for r in excluded)),
        'test_rows_omitted':0,'dev_rows_omitted':0,'train_heldout_group_overlap':0,
        'dev_test_shared_groups':len(groups['dev']&groups['test_in_domain']),
        'prompt_inventory_sha256':sha(out/'prompt_inventory.json'),'old_formal_exact_state_overlap':old_matches,
        'score_ranges':RANGES,'label_policy':'original integer domain1_score; complete native support',
        'input_policy':'verbatim author-decoded essay plus all prompt/source/rubric/adjudication paragraphs; administrative corpus-size metadata excluded',
        'truncated_rows':0,'rounded_labels':0,'test_selection':False,'code_sha256':sha(Path(__file__))}
    atomic(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('raw_counts','splits','excluded_training_rows','author_copy_verification')},indent=2))


if __name__=='__main__':main()
