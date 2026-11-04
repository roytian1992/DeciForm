import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from common import encode, parse_answer, support, target
from classification_order_audit import (encode_view, view_orders, normalize_prediction,
                                        summarize_records, input_hash)
from report_classification_order import classification_only
from test_order_audit import Tokenizer, record


def choice(k=5):
    return {'id':f'choice-{k}','task':'choice','source_family':'toy','group_id':'group',
            'state':{'text':'Which department can solve this request?'},
            'question':{'type':'choice','instructions':'Choose the best department.',
                        'options':[f'department {i}' for i in range(k)],
                        'option_ids':[f'id{i}' for i in range(k)],
                        'option_definitions':[f'department {i} handles service {i}' for i in range(k)]},
            'json_keys':[f'c{i}' for i in range(k)],'label':'id2'}


@pytest.mark.parametrize('k',[3,4,5,21,77])
def test_choice_views_keep_complete_inventory_and_match_across_models(k):
    r=choice(k);ref=view_orders('kev','native',r)
    assert len(ref)==5 and len({tuple(p) for _,p in ref})==5
    for arm in ['lora','kev','tcdl','json_sft']:
        assert view_orders(arm,'native',r)==ref
    for _,p in ref:
        assert sorted(p)==list(range(k))


@pytest.mark.parametrize('arm',['lora','kev','tcdl','json_sft'])
@pytest.mark.parametrize('k',[3,77])
def test_real_input_reorder_preserves_semantics_and_identity(arm,k):
    r=choice(k);before=copy.deepcopy(r);tok=Tokenizer();base=encode(r,arm,tok)
    for view,p in view_orders(arm,'native',r):
        out=encode_view(r,tok,arm,'native',p,original=base)
        if view=='original':assert out==base
        else:assert input_hash(out)!=input_hash(base)
        if arm=='lora':
            assert out['ids_list']==[base['ids_list'][i] for i in p]
        elif arm=='json_sft':
            assert out['answer']==base['answer']
            prompt=''.join(chr(i) for i in out['prompt'][1:])
            offsets=[prompt.index(f'"c{i}":') for i in p]
            assert offsets==sorted(offsets)
            assert parse_answer('{"choice_id":"c2"}',r)==(target(r),None)
        else:
            for i in range(k):
                old=[t for t,s in zip(base['ids'],base['opt']) if s==i]
                new=[t for t,s in zip(out['ids'],out['opt']) if s==i]
                assert old==new and out['opt'][out['opt_idx'][0][i]]==i
    assert r==before and base==encode(r,arm,tok)


def test_independent_candidate_scores_remap_to_semantic_ids():
    r=choice(3);p=[2,0,1]
    out=normalize_prediction(r,{'valid':True,'prediction_index':0,
                               'probabilities':[.7,.1,.2]},'lora',p)
    assert out['prediction_index']==2 and out['probabilities']==[.1,.2,.7]
    # An actual tie-breaking flip must not be erased by sorting probabilities.
    tied=normalize_prediction(r,{'valid':True,'prediction_index':0,
                                 'probabilities':[.5,.5,0.]},'lora',p)
    assert tied['prediction_index']==2


def test_score_is_rejected_and_classification_report_drops_score_groups():
    with pytest.raises(ValueError,match='restricted'):
        summarize_records([record('x','original',1,task='score')])
    noul=summarize_records([record('x','original',1),record('x','reversed',0)],False)
    raw={'groups':{'native/noul':noul,'native/score':{'task':'score','evaluated_predictions':10}},
         'by_source':{},'evaluated_predictions':12,'repeat_controls':[{'task':'score'},{'task':'noul'}]}
    before=copy.deepcopy(raw);clean=classification_only(raw)
    assert list(clean['groups'])==['native/noul'] and clean['evaluated_predictions']==2
    assert clean['repeat_controls']==[{'task':'noul'}] and raw==before


def test_generated_failure_and_semantic_flip_are_retained():
    rows=[record('x','original',1,task='choice'),record('x','reversed',None,task='choice',valid=False)]
    out=summarize_records(rows,False)
    assert out['order_metrics']['reverse_flip_rate']==1
    assert out['per_view']['reversed']['accuracy']==0
    assert out['order_metrics']['any_invalid_rate']==1
    assert 'mae' not in out['per_view']['original']
