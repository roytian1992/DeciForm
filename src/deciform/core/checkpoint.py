"""Portable trainable exports and atomic exact-resume checkpoints."""

import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from .data import atomic_json, sha256


def trainable_state(model):
    return {
        n: p.detach().cpu().contiguous().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def restore(model, state):
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if named.keys() != state.keys():
        raise ValueError("Checkpoint trainable inventory differs from configured model")
    with torch.no_grad():
        for n, p in named.items():
            if p.shape != state[n].shape:
                raise ValueError("Shape mismatch: " + n)
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))


def save_export(model, path):
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temp = path / "trainables.safetensors.tmp"
    try:
        save_file(trainable_state(model), str(temp))
        os.replace(temp, path / "trainables.safetensors")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    atomic_json(path / "config.json", model.config)
    atomic_json(path / "inventory.json", model.inventory())
    model.tokenizer.save_pretrained(path / "tokenizer")
    atomic_json(
        path / "format.json",
        {
            "version": 1,
            "weights": "trainables.safetensors",
            "weights_sha256": sha256(path / "trainables.safetensors"),
            "frozen_parameters": "reconstructed from explicit base_model/revision",
            "backbone": model.backbone,
        },
    )


def load_export(path, device="cpu", base_model=None, local_files_only=None):
    from safetensors.torch import load_file
    from .model import DecisionModel

    path = Path(path)
    metadata = json.loads((path / "format.json").read_text())
    if metadata.get("version") != 1:
        raise ValueError("Unsupported public checkpoint format")
    if metadata.get("weights_sha256") != sha256(path / "trainables.safetensors"):
        raise ValueError("Checkpoint weight checksum mismatch")
    cfg = json.loads((path / "config.json").read_text())
    if base_model:
        cfg["base_model"] = base_model
    if local_files_only is not None:
        cfg["local_files_only"] = local_files_only
    model = DecisionModel(cfg, device=device, tokenizer_source=path / "tokenizer")
    restore(model, load_file(str(path / "trainables.safetensors")))
    return model


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
