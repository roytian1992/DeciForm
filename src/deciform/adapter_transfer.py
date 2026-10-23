"""Function-preserving conversion of fixed-rank linear adapter programs.

All currently registered weight programs admit a factorization B_eff @ A.
Conversion carries that complete update forward, without merging it into the
backbone or increasing rank. This is a parameterization search, not evidence
of a new function class.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import parametrize

from deciform.adapter_program import (
    STANDARD_PROGRAM, install_program, iter_program_modules,
    load_program_adapter, validate_program,
)


def new_adapter(model, program: str, rank: int, targets):
    validate_program(program, rank)
    model.requires_grad_(False)
    if program == STANDARD_PROGRAM:
        return get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
            target_modules=list(targets), bias="none",
        ))
    install_program(model, program, rank, targets)
    return model


def _peft_layers(adapter):
    return {name: layer for name, layer in adapter.get_base_model().named_modules()
            if hasattr(layer, "lora_A") and "default" in layer.lora_A}


def initialize_adapter(model, program: str, rank: int, targets,
                       parent_dir: Path | None = None):
    """Return (trainable model, conversion audit); always retain parent weights."""
    validate_program(program, rank)
    if parent_dir is None:
        return new_adapter(model, program, rank, targets), {"mode": "fresh", "parent": None}
    config = json.loads((parent_dir / "adapter_config.json").read_text())
    source_program = config.get("adapter_program", STANDARD_PROGRAM)
    source_rank = config.get("r") if source_program == STANDARD_PROGRAM else config.get("rank")
    if source_rank != rank or set(config["target_modules"]) != set(targets):
        raise ValueError("adapter conversion requires identical rank and target modules")
    if source_program == STANDARD_PROGRAM:
        if config.get("use_dora") or config.get("use_rslora") or config.get("bias", "none") != "none":
            raise ValueError("source is not a supported ordinary LoRA adapter")
        loaded = PeftModel.from_pretrained(model, parent_dir, is_trainable=True)
    else:
        load_program_adapter(model, parent_dir, source_program, rank, targets)
        loaded = model
    audit = {"parent": str(parent_dir), "source_program": source_program,
             "target_program": program, "rank": rank}
    if source_program == program and (
        program == STANDARD_PROGRAM or config.get("adapter_program_version") == 2
    ):
        return loaded, {**audit, "mode": "continue", "max_weight_error": 0.0}
    with torch.no_grad():
        if source_program == STANDARD_PROGRAM:
            factors = {name: (layer.lora_A.default.weight.detach().clone(),
                              layer.lora_B.default.weight.detach().clone() * layer.scaling["default"])
                       for name, layer in _peft_layers(loaded).items()}
            model = loaded.unload()  # Restore the unchanged base, never merge/drop the update.
        else:
            factors = {name: tuple(x.detach().clone() for x in module.factors())
                       for name, module in iter_program_modules(loaded)}
            for name in list(factors):
                parametrize.remove_parametrizations(loaded.get_submodule(name), "weight",
                                                   leave_parametrized=False)
            model = loaded
        target = new_adapter(model, program, rank, targets)
        layers = _peft_layers(target) if program == STANDARD_PROGRAM else dict(iter_program_modules(target))
        if set(layers) != set(factors):
            raise ValueError("parent and target adapter modules differ")
        maximum_error = 0.0
        for name, layer in layers.items():
            a, b = factors[name]
            if program == STANDARD_PROGRAM:
                layer.lora_A.default.weight.copy_(a)
                layer.lora_B.default.weight.copy_(b / layer.scaling["default"])
                actual_a = layer.lora_A.default.weight
                actual_b = layer.lora_B.default.weight * layer.scaling["default"]
            else:
                for index, (dest_a, dest_b) in enumerate(zip(layer.A, layer.B, strict=True)):
                    start = index * layer.branch_rank
                    part_a = a[start:start + layer.branch_rank]
                    part_b = b[:, start:start + layer.branch_rank] / layer.scale
                    # Keep RMS clipping inactive at conversion, even for large
                    # parent factors; reciprocal rescaling preserves B @ A.
                    rescale = (part_b.abs().max() * 2.0).clamp_min(1.0)
                    dest_a.copy_(part_a * rescale)
                    dest_b.copy_(part_b / rescale)
                actual_a, actual_b = layer.factors()
            # Blockwise comparison bounds memory on large projection matrices.
            for start in range(0, b.shape[0], 256):
                expected = b[start:start + 256] @ a
                actual = actual_b[start:start + 256] @ actual_a
                error = float((expected - actual).abs().max())
                maximum_error = max(maximum_error, error)
                if not torch.allclose(expected, actual, atol=2e-6, rtol=2e-5):
                    raise RuntimeError(f"adapter conversion changed the parent update: {name}, {error}")
    return target, {**audit, "mode": "exact_factor_conversion", "max_weight_error": maximum_error,
                    "module_count": len(factors), "target_semantics_version": 2}
