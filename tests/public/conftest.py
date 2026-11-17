import inspect
import json
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(params=["modernbert", "qwen"])
def tiny_config(tmp_path, request, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        ModernBertConfig,
        ModernBertForMaskedLM,
        Qwen3_5TextConfig,
        Qwen3_5ForCausalLM,
    )
    from deciform.core.encoding import SPECIAL

    special = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[EOS]", *SPECIAL]
    vocab = {s: i for i, s in enumerate(special)}
    for c in "0123456789.}abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ: ,{\n":
        if c not in vocab:
            vocab[c] = len(vocab)
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        additional_special_tokens=list(SPECIAL),
    )
    base = tmp_path / "base"
    base.mkdir()
    tok.save_pretrained(base)
    torch.manual_seed(71)
    if request.param == "modernbert":
        cfg = ModernBertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            global_attn_every_n_layers=2,
            local_attention=8,
            max_position_embeddings=512,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            reference_compile=False,
        )
        full = ModernBertForMaskedLM(cfg)
        private = {"enabled": True, "shared_rank": 2, "rank": 1}
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        for name in (
            "causal_conv1d_fn",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
        cfg = Qwen3_5TextConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            max_position_embeddings=512,
            attention_dropout=0.0,
            pad_token_id=0,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        )
        full = Qwen3_5ForCausalLM(cfg)
        private = {"enabled": True, "last_layers": 1, "rank": 1, "scale": 2.0}
    full.save_pretrained(base)
    return {
        "backbone": request.param,
        "base_model": str(base),
        "local_files_only": True,
        "seed": 71,
        "bf16": False,
        "query_size": 8,
        "adapter": {"rank": 2, "alpha": 4, "dropout": 0.0},
        "private": private,
        "output_scale": True,
        "gradient_checkpointing": True,
    }


@pytest.fixture
def rows():
    return [
        {
            "id": "c",
            "task": "choice",
            "state": "A red fruit",
            "question": {
                "instructions": "Pick the fruit",
                "options": ["apple", "stone", "book"],
                "option_ids": ["a", "s", "b"],
            },
            "label": "a",
        },
        {
            "id": "n",
            "task": "noul",
            "state": "Snow is cold",
            "question": {"instructions": "Is this true?"},
            "label": True,
        },
        {
            "id": "s",
            "task": "score",
            "state": "Useful answer",
            "question": {"instructions": "Rate the quality"},
            "score_range": {"min": 0, "max": 10, "step": 1},
            "label": 6.25,
        },
    ]
