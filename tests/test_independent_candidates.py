"""Shared prefix parity, fan-out gradients, candidate isolation and routing."""
import copy
import inspect
from pathlib import Path
import sys

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from common import TARGETS
from kev_ours_v5 import AnchoredTaskRouter
from independent_candidates import (shared_hidden, independent_reference,
    permute_candidates, split_candidates, stable_argmax)


@pytest.fixture(autouse=True)
def cpu_reference_kernels(monkeypatch):
    # Official Torch reference functions, for CPU mathematical tests only.
    # Production keeps the installed CUDA/FLA kernels unchanged.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name in ('causal_conv1d_fn', 'torch_chunk_gated_delta_rule',
                 'torch_recurrent_gated_delta_rule'):
        monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))


def tiny():
    torch.manual_seed(71)
    cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=128, attention_dropout=0.,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [2, 3, 3]})
    cfg._attn_implementation = 'sdpa'
    lm = get_peft_model(Qwen3_5TextModel(cfg), LoraConfig(task_type='FEATURE_EXTRACTION',
        r=2, lora_alpha=4, lora_dropout=0., target_modules=TARGETS))
    router = AnchoredTaskRouter(lm)
    with torch.no_grad():
        router.matrix_raw.normal_(std=.1)
        for name, parameter in lm.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.05)
    return lm, router


def records():
    return [{'prefix': [3, 5, 8, 11], 'query_token': 2,
             'branches': [[21, 22], [31, 32, 33]], 'semantics': [0, 1]},
            {'prefix': [7, 9], 'query_token': 2,
             'branches': [[41, 42, 43, 44], [51, 52], [61]], 'semantics': [0, 1, 2]}]


def flatten(outputs):
    q, hs = outputs
    return torch.cat([q.flatten(), *[h.flatten() for h in hs]])


@pytest.mark.parametrize('checkpointing', [False, True])
def test_shared_prefix_matches_independent_forward_and_all_trainable_gradients(checkpointing):
    torch.set_num_threads(2)
    lm, router = tiny()
    lm.train()
    for i, layer in enumerate(lm.get_base_model().layers):
        layer.gradient_checkpointing = checkpointing and i == 0
    tasks = torch.tensor([0, 2])
    actual = flatten(shared_hidden(lm, records(), 0, tasks, router))
    params = [p for p in [*lm.parameters(), *router.parameters()] if p.requires_grad]
    weights = torch.linspace(-.2, .3, actual.numel())
    grads = torch.autograd.grad((actual * weights).sum(), params)
    # Reference performs full independent forwards without recomputation.
    for layer in lm.get_base_model().layers:
        layer.gradient_checkpointing = False
    expected = flatten(independent_reference(lm, records(), 0, tasks, router))
    ref_grads = torch.autograd.grad((expected * weights).sum(), params)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=3e-5)
    assert grads[-1].abs().sum() > 0, 'Task routing must receive shared-prefix gradients'
    assert router.maps is None


def test_candidate_permutation_and_content_isolation():
    lm, router = tiny()
    lm.eval()
    rs = records()
    tasks = torch.tensor([0, 1])
    with torch.no_grad():
        query, options = shared_hidden(lm, rs, 0, tasks, router)
        changed = [permute_candidates(rs[0], [1, 0]), permute_candidates(rs[1], [2, 0, 1])]
        q2, h2 = shared_hidden(lm, changed, 0, tasks, router)
        torch.testing.assert_close(query, q2, rtol=0, atol=1e-6)
        for a, b in zip(options, h2):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        changed = copy.deepcopy(rs)
        changed[0]['branches'][1] = [63, 64, 65]
        q3, h3 = shared_hidden(lm, changed, 0, tasks, router)
        torch.testing.assert_close(query, q3, rtol=0, atol=1e-6)
        torch.testing.assert_close(options[0][0], h3[0][0], rtol=0, atol=1e-6)
        assert not torch.allclose(options[0][1], h3[0][1])


def test_lossless_split_and_stable_ties():
    enc = {'ids': [3, 4, 10, 11, 12, 20, 21, 2],
           'opt': [-1, -1, 1, 1, 1, 0, 0, -2], 'opt_idx': [[6, 4]],
           'decide_idx': [7], 'tasks': ['choice'], 'state_truncated': False}
    split = split_candidates(enc)
    assert split['prefix'] == [3, 4] and split['query_token'] == 2
    assert split['branches'] == [[10, 11, 12], [20, 21]]
    assert split['semantics'] == [1, 0] and split['original_tokens'] == 8
    assert stable_argmax([.5, .5], ['b', 'a']) == 1
    assert stable_argmax([.5, .5], ['a', 'b']) == 0
    with pytest.raises(ValueError):
        split_candidates({**enc, 'state_truncated': True})
