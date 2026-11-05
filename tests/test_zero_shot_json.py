"""Matched prompts, semantic permutations, and honest invalid-output metrics."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from common import json_prompt, support
from zero_shot_protocol import (CHOICE_ORDERS, NEW_ORDERS, LEGACY_ORDERS, order_for,
    prompt_body, prompt_ids, record_prediction, summarize, strip_terminal)
from classification_order_audit import encode_view, view_orders
from order_audit import encode_view as encode_noul
from test_classification_order_audit import choice
from test_order_audit import Tokenizer
from test_score_baselines import row as score_row


@pytest.mark.parametrize('k', [3, 5, 21, 77])
def test_choice_all_orders_keep_full_semantics_and_match_legacy(k):
    row = choice(k)
    original = copy.deepcopy(row)
    tok = Tokenizer()
    assert prompt_body(row, 'main') == json_prompt(row)
    assert prompt_body(row, 'choice_order') == json_prompt(row)
    for view in CHOICE_ORDERS:
        order = order_for(row, view)
        assert sorted(order) == list(range(k))
        text = prompt_body(row, 'choice_order', view)
        positions = [text.index(f'"c{i}":') for i in order]
        assert positions == sorted(positions)
        result = record_prediction(row, 'choice_order', view, '{"choice_id":"c2"}', '{"choice_id":"c2"}')
        assert result['prediction_index'] == 2 and result['correct']
    for name, order in view_orders('json_sft', 'native', row):
        ours = 'legacy_' + name if name.startswith('shuffle_') else name
        actual = prompt_ids(row, 'choice_order', ours, tok, 'matched_base')
        expected = encode_view(row, tok, 'json_sft', 'native', order)['prompt']
        assert actual == expected
    assert original == row


def test_new_d2_permutations_match_frozen_implementation():
    from order_audit_independent import order_for as frozen_order
    for k in [3, 21, 77]:
        row = choice(k)
        for name in NEW_ORDERS:
            frozen_name = 'reverse' if name == 'reversed' else name
            assert order_for(row, name) == frozen_order(frozen_name, {'branches': [None] * k}, row)


def noul():
    return {'id': 'n', 'task': 'noul', 'source_family': 'source', 'group_id': 'group',
            'state': {'text': 'Full evidence'}, 'question': {'instructions': 'Is it true?'}, 'label': True}


def test_noul_prompt_changes_list_order_and_matches_prior_intervention():
    row, tok = noul(), Tokenizer()
    for view in ['original', 'reversed']:
        expected = encode_noul(row, tok, 'json_sft', 'instruction_list', order_for(row, view))['prompt']
        assert prompt_ids(row, 'noul_order', view, tok, 'matched_base') == expected
    assert 'Allowed answers: [false, true].' in prompt_body(row, 'noul_order', 'original')
    assert 'Allowed answers: [true, false].' in prompt_body(row, 'noul_order', 'reversed')
    assert 'Allowed answers:' not in prompt_body(row, 'main')


def test_inputs_never_include_gold_and_fractional_labels_are_preserved():
    row = score_row(False, 92.123456789)
    assert prompt_body(row, 'continuous') == prompt_body({**row, 'label': .2}, 'continuous')
    raw = '{"score":92.125}'
    prediction = record_prediction(row, 'continuous', 'original', raw, raw)
    assert prediction['label'] == 92.123456789
    assert prediction['prediction_value'] == 92.125
    assert 'probabilities' not in prediction


def test_invalid_output_cannot_disappear_or_fake_invariance():
    row = noul()
    invalid = [record_prediction(row, 'noul_order', v, 'bad', 'bad') for v in ['original', 'reversed']]
    report = summarize('noul_order', invalid)
    assert report['rows'] == 1 and report['per_view']['original']['validity'] == 0
    assert report['order_metrics']['both_valid_reverse_pairs'] == 0
    assert report['order_metrics']['reverse_flip_rate_on_valid_pairs'] is None
    valid = record_prediction({**row, 'id': 'v'}, 'main', 'original', '{"answer":true}', '{"answer":true}')
    bad = record_prediction(row, 'main', 'original', 'bad', 'bad')
    assert summarize('main', [valid, bad])['task/noul']['accuracy'] == .5
    s = score_row(False, 50.5)
    score_bad = record_prediction(s, 'continuous', 'original', 'bad', 'bad')
    assert summarize('continuous', [score_bad])['task/score']['mae'] == 100


def test_strict_parser_preserves_explanation_failures_and_boolean_semantics():
    for text in ['prefix {"answer":true}', '{"answer":1}', '{"answer":true,"answer":false}']:
        assert not record_prediction(noul(), 'main', 'original', text, text)['valid']
    assert strip_terminal('{"answer":true}<end><pad>', ['<pad>', '<end>']) == '{"answer":true}'
    assert strip_terminal('prefix {"answer":true}', ['<end>']) == 'prefix {"answer":true}'


def test_separate_position_reports_use_their_exact_frozen_view_sets():
    row = choice(5)
    records = [record_prediction(row, 'choice_order', v, '{"choice_id":"c2"}', '{"choice_id":"c2"}')
               for v in CHOICE_ORDERS]
    reports = summarize('choice_order', records)
    assert set(reports['new_d2_seven_orders']['per_view']) == set(NEW_ORDERS)
    assert set(reports['legacy_baseline_five_orders']['per_view']) == set(LEGACY_ORDERS)


def test_chat_mode_explicitly_disables_reasoning_without_changing_user_body():
    class Chat:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': json_prompt(noul())}]
            assert kwargs == {'tokenize': True, 'add_generation_prompt': True, 'enable_thinking': False}
            return [7, 8]
    assert prompt_ids(noul(), 'main', 'original', Chat(), 'official_chat') == [7, 8]
