import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from common import encode, json_prompt, support, target
from order_audit import (orders, permute_pointer, encode_view, input_hash,
    applicable, view_orders, summarize_records, normalize_prediction)


class Tokenizer:
    eos_token_id=999999
    def __init__(self):
        self.special={}
    def __call__(self,text,add_special_tokens=False,**kwargs):
        assert not kwargs.get('truncation',False)
        if isinstance(text,list):
            ids=[self(t,add_special_tokens=add_special_tokens).input_ids for t in text]
        else:
            ids=([999998] if add_special_tokens else [])+[ord(c) for c in text]
        return SimpleNamespace(input_ids=ids)
    def convert_tokens_to_ids(self,text):
        if text not in self.special:
            self.special[text]=10000+len(self.special)
        return self.special[text]


def row(task):
    return {'id':task,'task':task,'source_family':'toy','group_id':task,
            'state':{'text':'Evidence must remain unchanged.'},
            'question':{'instructions':'Assess the evidence.', 'scale':{'min':0,'max':4}},
            'label':2 if task=='score' else True}


@pytest.mark.parametrize('k,count',[(2,2),(5,10)])
def test_permutation_design_is_complete_unique_and_position_balanced(k,count):
    variants=orders(k)
    assert len(variants)==count and len({tuple(p) for _,p in variants})==count
    assert dict(variants)['reversed']==list(reversed(range(k)))
    for position in range(k):
        assert [sum(p[position]==candidate for _,p in variants) for candidate in range(k)]==[count//k]*k


@pytest.mark.parametrize('task',['noul','score'])
def test_pointer_permutation_moves_tokens_but_preserves_semantic_readouts(task):
    r=row(task);tok=Tokenizer();base=encode(r,'kev',tok)
    first=min(i for i,opt in enumerate(base['opt']) if opt>=0)
    last=max(i for i,opt in enumerate(base['opt']) if opt>=0)+1
    for name,p in orders(len(support(r))):
        changed=permute_pointer(base,p)
        assert changed['ids'][:first]==base['ids'][:first]
        assert changed['ids'][last:]==base['ids'][last:]
        assert changed['pos']==base['pos'] and changed['seg']==base['seg']
        assert changed['labels']==base['labels']==[target(r)]
        for semantic in range(len(p)):
            old=[token for token,opt in zip(base['ids'],base['opt']) if opt==semantic]
            new=[token for token,opt in zip(changed['ids'],changed['opt']) if opt==semantic]
            assert old==new
            end=changed['opt_idx'][0][semantic]
            assert changed['opt'][end]==semantic and changed['ids'][end]==old[-1]
        if name=='original':
            assert changed==base
        else:
            assert changed['ids']!=base['ids']
    assert base==encode(r,'kev',tok)
    with pytest.raises(ValueError):permute_pointer(base,[0]*len(support(r)))


@pytest.mark.parametrize('arm',['lora','kev','json_sft','tcdl'])
@pytest.mark.parametrize('task',['noul','score'])
def test_native_applicability_and_common_instruction_perturbation(arm,task):
    r=row(task);before=copy.deepcopy(r);tok=Tokenizer();identity=list(range(len(support(r))))
    native=encode(r,arm,tok)
    a=encode_view(r,tok,arm,'native',identity)
    assert a==native
    b=encode_view(r,tok,arm,'native',identity[::-1])
    assert (input_hash(a)!=input_hash(b))==applicable(arm,'native',task)
    common_a=encode_view(r,tok,arm,'instruction_list',identity)
    common_b=encode_view(r,tok,arm,'instruction_list',identity[::-1])
    assert input_hash(common_a)!=input_hash(common_b)
    assert r==before and target(r)==(2 if task=='score' else 1)
    assert len(view_orders(arm,'native',r))==(len(orders(len(identity))) if applicable(arm,'native',task) else 1)


def test_generation_reorders_only_controlled_score_list():
    r=row('score');tok=Tokenizer()
    a=encode_view(r,tok,'json_sft','native',[4,3,2,1,0])
    text=''.join(chr(i) for i in a['prompt'][1:])
    assert text==json_prompt(r).replace('Allowed scores: [0, 1, 2, 3, 4]',
                                      'Allowed scores: [4, 3, 2, 1, 0]')
    assert a['answer']==encode(r,'json_sft',tok)['answer']


def record(identifier,view,prediction,label=1,valid=True,task='noul',can_test=True):
    return {'id':identifier,'view':view,'prediction_index':prediction,'label_index':label,
            'valid':valid,'task':task,'label':label,'support':[False,True],
            'source':'toy','group_id':identifier,'order_test_applicable':can_test}


def test_paired_flip_detects_changes_hidden_by_equal_accuracy():
    records=[record('a','original',1),record('a','reversed',0),
             record('b','original',0),record('b','reversed',1)]
    report=summarize_records(records)
    assert report['per_view']['original']['accuracy']==report['per_view']['reversed']['accuracy']==.5
    assert report['order_metrics']['reverse_flip_rate']==1
    assert report['order_metrics']['reverse_error_delta_positive_is_worse']==0
    assert report['order_metrics']['worst_order_error']==1
    assert report['order_metrics']['reverse_flip_rate_95ci']['low']==1
    with pytest.raises(ValueError):summarize_records(records[:-1])


def test_native_noop_is_na_not_a_zero_flip_advantage():
    report=summarize_records([record('a','original',1,can_test=False)])
    assert report['order_metrics'] is None and report['applicable'] is False


def test_invalid_score_remains_in_denominator_with_native_penalty():
    records=[{**record('a','original',1,label=1,task='score'),'support':[0,20,100],'label':20},
             {**record('a','reversed',None,label=1,valid=False,task='score'),'support':[0,20,100],'label':20}]
    report=summarize_records(records)
    assert report['per_view']['reversed']['mae']==100
    assert report['per_view']['reversed']['rows']==1
    assert report['order_metrics']['any_invalid_rate']==1


def test_score_prediction_uses_numeric_median_and_keeps_other_rules():
    r=row('score');r['question']['scale']['max']=2
    out=normalize_prediction(r,{'prediction_index':0,'valid':True,'probabilities':[.4,.35,.25]})
    assert out['prediction_index']==1 and out['argmax_prediction_index']==0
    assert out['score_decision_rule']=='median'
