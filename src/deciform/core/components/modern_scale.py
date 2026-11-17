"""Scale native projection outputs, separately from low-rank residuals."""

import torch
from torch import nn
from .modern_shared import FusedSharedProjection


class OutputScaleProjection(nn.Module):
    """Keep all old state names/RNG; scale only the native output branch."""

    def __init__(self, original):
        super().__init__()
        if not isinstance(original, FusedSharedProjection):
            raise TypeError("Output-scale requires an existing shared-A projection")
        if original.freeze_base or not all(
            p.requires_grad for p in original.base_layer.parameters()
        ):
            raise ValueError("ModernBERT output-scale requires a trainable backbone")
        self.base_layer = original.base_layer
        self.lora_A = original.lora_A
        self.lora_B = original.lora_B
        self.dropout = original.dropout
        self.freeze_base = False
        self.in_features, self.out_features = original.in_features, original.out_features
        self.scale = original.scale
        self.output_gain = nn.Parameter(
            torch.zeros(
                self.out_features, device=self.base_layer.weight.device, dtype=torch.float32
            )
        )
        self.gain_enabled = True

    def forward(self, x):
        result = self.base_layer(x)
        if self.gain_enabled:
            # Compute near-one gains in FP32 before BF16 output conversion.
            result = (result.float() * (1 + self.output_gain)).to(result.dtype)
        latent = self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))
        return result + (self.lora_B(latent) * self.scale).to(result.dtype)


def base_encoder_parameters(model):
    for name, parameter in model.encoder.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name and not name.endswith(".output_gain"):
            yield name.replace(".base_layer.", "."), parameter


def install_output_scales(encoder):
    """Wrap exactly two residual output projections per native layer."""
    sites = [
        (i, getattr(layer, parent), "Wo", parent + ".Wo")
        for i, layer in enumerate(encoder.layers)
        for parent in ("attn", "mlp")
    ]
    # Validate the complete installation before mutating the module tree.
    for _, owner, name, _ in sites:
        original = getattr(owner, name)
        if not isinstance(original, FusedSharedProjection):
            raise TypeError("Expected unscaled FusedSharedProjection at " + name)
        if original.freeze_base or not all(
            p.requires_grad for p in original.base_layer.parameters()
        ):
            raise ValueError("Frozen ModernBERT is not supported by this experiment")
    inventory = []
    for i, owner, name, label in sites:
        wrapped = OutputScaleProjection(getattr(owner, name))
        setattr(owner, name, wrapped)
        inventory.append(
            {
                "layer": i,
                "projection": label,
                "parameters": wrapped.output_gain.numel(),
                "dtype": "float32",
                "initialization": "zero",
                "tasks": "all",
            }
        )
    return inventory


def optimizer_groups(model, gain_lr=None):
    if gain_lr is None:
        gain_lr = getattr(model, "output_scale_lr", 1e-3)
    assert gain_lr > 0
    base = list(base_encoder_parameters(model))
    assert base and all(
        p.requires_grad for _, p in base
    ), "All native encoder parameters must train"
    encoder = list(model.encoder.named_parameters())
    gains = [p for n, p in encoder if n.endswith(".output_gain")]
    assert gains and all(p.requires_grad for p in gains)
    groups = [
        {
            "name": "backbone_shared_adapters",
            "params": [p for n, p in encoder if p.requires_grad and not n.endswith(".output_gain")],
            "lr": 2e-5,
        },
        {
            "name": "pointer",
            "params": list(model.q.parameters()) + list(model.k.parameters()),
            "lr": 2e-5,
        },
        {"name": "decimal_readout", "params": list(model.numeric_score.parameters()), "lr": 1e-3},
        {"name": "output_scales", "params": gains, "lr": float(gain_lr)},
    ]
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in model.numeric_transform.parameters())
    return groups


def parameter_inventory(model):
    groups = optimizer_groups(model)
    return {
        **{g["name"]: sum(p.numel() for p in g["params"]) for g in groups},
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }
