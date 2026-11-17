"""Fused shared-A projections for ModernBERT; full encoder training supported."""

import math
import torch
from torch import nn
from .modern_core import ModernCore


class FusedSharedProjection(nn.Module):
    def __init__(self, base, rank=12, alpha=24, dropout=0.05, freeze_base=True):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base_layer = base.requires_grad_(not freeze_base)
        self.freeze_base = bool(freeze_base)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Linear(
            base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32
        )
        self.lora_B = nn.Linear(
            rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        result = self.base_layer(x)
        # Exactly one shared latent/dropout draw for all fused output branches.
        latent = self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))
        return result + (self.lora_B(latent) * self.scale).to(result.dtype)


def install_shared_adapters(encoder, rank=12, alpha=24, dropout=0.05, freeze_backbone=True):
    encoder.requires_grad_(not freeze_backbone)
    inventory = []
    for i, layer in enumerate(encoder.layers):
        for parent, name, branches in (
            ("attn", "Wqkv", ("query", "key", "value")),
            ("attn", "Wo", ("attention_output",)),
            ("mlp", "Wi", ("input", "gate")),
            ("mlp", "Wo", ("mlp_output",)),
        ):
            owner = getattr(layer, parent)
            base = getattr(owner, name)
            assert isinstance(base, nn.Linear), "Already wrapped or unsupported projection"
            wrapped = FusedSharedProjection(base, rank, alpha, dropout, freeze_base=freeze_backbone)
            setattr(owner, name, wrapped)
            inventory.append(
                {
                    "layer": i,
                    "projection": f"{parent}.{name}",
                    "output_branches": list(branches),
                    "input_width": base.in_features,
                    "output_width": base.out_features,
                    "rank": rank,
                    "alpha": alpha,
                    "parameters": sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
                }
            )
    return inventory


def base_encoder_parameters(model):
    """Original names/order allow exact comparison to the native pretrained hash."""
    for name, parameter in model.encoder.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            yield name.replace(".base_layer.", "."), parameter


def parameter_inventory(model):
    groups = optimizer_groups(model)
    return {
        **{g["name"]: sum(p.numel() for p in g["params"]) for g in groups},
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }


def optimizer_groups(model):
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    base_params = [p for _, p in base_encoder_parameters(model) if p.requires_grad]
    encoder_group_name = "shared_adapters" if not base_params else "backbone_shared_adapters"
    groups = [
        {"name": encoder_group_name, "params": encoder_params, "lr": 2e-5},
        {
            "name": "pointer",
            "params": list(model.q.parameters()) + list(model.k.parameters()),
            "lr": 2e-5,
        },
        {"name": "decimal_readout", "params": list(model.numeric_score.parameters()), "lr": 1e-3},
    ]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    if not base_params:
        assert all(not p.requires_grad for _, p in base_encoder_parameters(model))
    assert all(not p.requires_grad for p in model.numeric_transform.parameters())
    return groups
