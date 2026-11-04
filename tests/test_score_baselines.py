"""Score baselines retain labels/ranges and count malformed JSON outputs."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from score_baselines import json_prompt, json_answer, parse_score, prediction_row, metrics
from score_specialist import interpolation, data_loss, SpecialistObjective, metrics as original_metrics
from models_score_baselines import strip_terminal_tokens, FormalModel


def row(native=True, value=31):
    return {'id':'r', 'task':'score', 'source_family':'asap' if native else 'wmt',
        'group_id':'r', 'state':{'text':'complete essay or translation'},
        'question':{'instructions':'Original complete rubric'},
        'label_kind':'native_integer' if native else 'continuous',
        'label':value, 'score_support':list(range(61 if native else 101))}


def test_json_targets_preserve_fractional_labels_and_prompt_has_no_target():
    r=row(False,92.66666412353516)
    assert json.loads(json_answer(r))['score']==r['label']
    assert parse_score(json_answer(r),r)==(r['label'],None)
    assert json_prompt(r)==json_prompt({**r,'label':0.125})
    assert 'finite JSON number' in json_prompt(r)
    assert 'Fractional scores are valid' in json_prompt(r)
    native=row()
    assert json.dumps(native['score_support']) in json_prompt(native)
    assert 'allowed integer' in json_prompt(native)


@pytest.mark.parametrize('text',[
    '{"score":true}', '{"score":NaN}', '{"score":Infinity}', '{"score":1e999}',
    '{"score":-1}', '{"score":61}', '{"score":31,"score":32}',
    '{"score":31,"reason":"x"}', 'prefix {"score":31}', '```json\n{"score":31}\n```',
    '[31]', '{"score":"31"}', '{"score":31.5}', '{"score":31.0}', '{}'])
def test_native_json_rejects_invalid_without_repair(text):
    value,error=parse_score(text,row())
    assert value is None and error


def test_generative_invalid_predictions_stay_in_denominator():
    r=row(False,50.25)
    a=prediction_row(r,{'valid':True,'prediction_value':50.5})
    b=prediction_row({**r,'id':'bad'}, {'valid':False,'prediction_value':None})
    report=metrics([a,b])
    assert report['task/score']['rows']==2
    assert report['task/score']['validity']==.5
    assert report['task/score']['mae']==50.125
    assert report['task/score']['mae_on_valid_only']==.25
    assert 'probabilities' not in b and 'rules' not in report['task/score']


def test_kev_metrics_match_frozen_specialist_rules():
    r=row(False,50.25)
    p=[0.]*101;p[50]=.75;p[51]=.25
    sample=prediction_row(r,{'valid':True,'probabilities':p,'prediction_index':50})
    old=original_metrics([sample]);new=metrics([sample])
    for key,value in old.items():
        if isinstance(value,dict):
            for name,expected in value.items():assert new[key][name]==expected
        else:assert new[key]==value


@pytest.mark.parametrize('k',[4,5,6,11,31,61,101])
def test_kev_objective_is_exact_ce_at_native_labels(k):
    torch.manual_seed(8)
    z=torch.randn(k,dtype=torch.float64,requires_grad=True);y=k//2
    actual=data_loss(z,list(range(k)),y,SpecialistObjective('soft_ce'))
    expected=torch.nn.functional.cross_entropy(z[None],torch.tensor([y]))
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,z,retain_graph=True)[0],
                               torch.autograd.grad(expected,z)[0])


def test_fractional_kev_ce_uses_same_interpolation_as_ours():
    q=interpolation(list(range(101)),92.25)
    assert q[92]==.75 and q[93]==.25 and sum(q)==1.


def test_only_actual_terminal_tokens_are_stripped_in_any_order():
    assert strip_terminal_tokens('{"score":3}<eos><pad><pad>', ['<eos>','<pad>'])=='{"score":3}'
    assert strip_terminal_tokens('explanation {"score":3}', ['<eos>'])=='explanation {"score":3}'


def test_cancelled_ordinary_model_cannot_be_constructed():
    with pytest.raises(ValueError,match='Only Kev'):
        FormalModel('lora','not_a_model_path')
