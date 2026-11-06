"""Protocol tests: full inputs, full support, permutation remapping, score units."""
import itertools
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/formal'))
sys.path.insert(0,str(ROOT/'runs/formal_laya_20261106_v1/vendor/laya_release'))
from laya_adapter import make_parts, encode_parts, canonical_probabilities, predict_record
from laya.common import DecisionModel, build_sequence, collate_items
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(ROOT/'runs/formal_laya_20261106_v1/base',local_files_only=True)


def row(task='choice'):
    r={'id':'test/1','group_id':'test/1','source_family':'test','task':task,
       'state':{'text':'A detailed billing complaint.'},'label':'billing','json_keys':['c0','c1','c2'],
       'question':{'instructions':'Select the category.','options':['billing','technical','other']}}
    if task=='noul':
        r['question']={'instructions':'Is this a billing issue?'}
        r['label']=True
    elif task=='score':
        r['question']={'instructions':'Rate on a 1 to 5 scale.','scale':{'min':1,'max':5}}
        r['label']=4
    return r


def test_compatible_short_template(tok):
    from laya_adapter import question
    for task in ('choice','noul','score'):
        r=row(task)
        ours=encode_parts(make_parts(r,tok))
        ids,markers=build_sequence(tok,r['state']['text'],question(r),max_len=8192,head_max_len=8192)
        assert ours['ids']==ids and ours['markers']==markers


def test_no_text_or_candidate_clipping(tok):
    r=row()
    r['state']['text']='long document '*5000
    r['question']['options']=['label '+str(i)+' '+('distinct description '*80) for i in range(77)]
    r['json_keys']=['c'+str(i) for i in range(77)]
    r['label']=r['question']['options'][42]
    parts=make_parts(r,tok)
    enc=encode_parts(parts)
    assert len(enc['markers'])==77 and sum(enc['target'])==1 and enc['target'][42]==1
    assert len(parts['options'][0])>48
    assert len(enc['ids'])>8192
    assert parts['suffix'][1:-1]==tok(r['state']['text'],add_special_tokens=False)['input_ids']


def test_permutation_alignment_and_score_units(tok):
    r=row();parts=make_parts(r,tok)
    for order in itertools.permutations(range(3)):
        enc=encode_parts(parts,order=list(order))
        assert enc['target'][list(order).index(0)]==1
        logits=torch.tensor([5.,1.,-1.])[list(order)]
        result=predict_record(r,canonical_probabilities(logits,list(order)))
        assert result['prediction_index']==0 and result['correct']
    result=predict_record(row('score'),[.05,.1,.2,.6,.05])
    assert result['support']==[1,2,3,4,5] and result['mae']==0 and result['prediction_index']==3
    assert predict_record(row('noul'),[.3,.7])['correct']


def test_parallel_equivariance_real_encoder(tok):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg=ModernBertConfig(vocab_size=50368,hidden_size=64,intermediate_size=128,num_hidden_layers=3,
                        num_attention_heads=4,local_attention=8,global_attn_every_n_layers=2,
                        pad_token_id=tok.pad_token_id,attn_implementation='sdpa')
    model=DecisionModel(ModernBertModel(cfg),head_layers=2).eval()
    parts=make_parts(row(),tok)
    def run(order,layout):
        b=collate_items([[encode_parts(parts,layout,order)]],tok.pad_token_id)
        extra={k:b[k] for k in ('position_ids','option_ids') if k in b}
        with torch.inference_mode():
            z,_=model(*(b[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')),**extra)
        return torch.tensor(canonical_probabilities(z[0],order))
    ref=run([0,1,2],'parallel')
    for order in itertools.permutations(range(3)):
        assert torch.allclose(ref,run(list(order),'parallel'),atol=1e-6,rtol=1e-5)
    assert not torch.allclose(run([0,1,2],'sequential'),run([2,1,0],'sequential'),atol=1e-6,rtol=1e-5)
