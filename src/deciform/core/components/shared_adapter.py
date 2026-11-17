"""Same-input shared-A adaptation; scoped caches are safe under recomputation."""

import math
import torch
from torch import nn


class InputSubspace(nn.Module):
    def __init__(self, width, rank, dropout, device):
        super().__init__()
        self.lora_A = nn.Linear(width, rank, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.clear()
        self.calls = 0

    def clear(self):
        self._input = None
        self._latent = None

    def forward(self, x):
        if self._input is not None:
            if self._input is not x:
                raise RuntimeError("Shared-A group received different input tensors")
            return self._latent
        latent = self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))
        self._input, self._latent = x, latent
        self.calls += 1
        return latent


class SharedProjection(nn.Module):
    def __init__(self, base, group, rank, alpha):
        super().__init__()
        self.base_layer = base
        self.base_layer.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_B = nn.Linear(
            rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32
        )
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        # The parent owns the group once; no duplicate module/parameter paths.
        object.__setattr__(self, "input_group", group)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        delta = self.lora_B(self.input_group(x)) * self.scale
        return (result + delta.to(result.dtype)).to(result.dtype)


def _start(parent, args):
    for group in parent.core_input_groups.values():
        group.clear()


def _finish(parent, args, result):
    for group in parent.core_input_groups.values():
        group.clear()


def install_shared_adapters(lm, rank=12, alpha=24, dropout=0.05):
    inventory = []
    for i, layer in enumerate(lm.layers):
        layouts = [
            (layer.mlp, "mlp", [("input", ("gate_proj", "up_proj")), ("output", ("down_proj",))])
        ]
        if hasattr(layer, "self_attn"):
            layouts.append(
                (
                    layer.self_attn,
                    "self_attn",
                    [("input", ("q_proj", "k_proj", "v_proj")), ("output", ("o_proj",))],
                )
            )
        else:
            layouts.append(
                (
                    layer.linear_attn,
                    "linear_attn",
                    [
                        ("input", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
                        ("output", ("out_proj",)),
                    ],
                )
            )
        for parent, kind, groups in layouts:
            if hasattr(parent, "core_input_groups"):
                raise RuntimeError("Adapters already installed")
            parent.core_input_groups = nn.ModuleDict()
            for key, names in groups:
                originals = [getattr(parent, n) for n in names]
                assert all(isinstance(m, nn.Linear) for m in originals)
                assert len({m.in_features for m in originals}) == 1
                group = InputSubspace(
                    originals[0].in_features, rank, dropout, originals[0].weight.device
                )
                parent.core_input_groups[key] = group
                for name, original in zip(names, originals):
                    setattr(parent, name, SharedProjection(original, group, rank, alpha))
                inventory.append(
                    {
                        "layer": i,
                        "parent": kind,
                        "group": key,
                        "projections": list(names),
                        "input_width": originals[0].in_features,
                        "output_widths": [m.out_features for m in originals],
                    }
                )
            parent.register_forward_pre_hook(_start)
            parent.register_forward_hook(_finish, always_call=True)
    return inventory


def assert_caches_clear(lm):
    assert all(
        m._input is None and m._latent is None for m in lm.modules() if isinstance(m, InputSubspace)
    )
