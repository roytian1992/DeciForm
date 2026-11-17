"""Single-factor CORE adapter studies; no candidate or task routing.

Output scale adds diag(g) W0 directions to the existing low-rank update,
only at attention and MLP residual outputs. LoRA+ is a separate optimizer
control, not a new architecture or an original invention.
"""

import contextlib
import torch
from torch import nn


class OutputScaleProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.base_layer = original.base_layer
        self.lora_B = original.lora_B
        self.scale = original.scale
        self.in_features = original.in_features
        self.out_features = original.out_features
        object.__setattr__(self, "input_group", original.input_group)
        self.output_gain = nn.Parameter(
            torch.zeros(
                self.out_features, device=self.base_layer.weight.device, dtype=torch.float32
            )
        )
        self.gain_enabled = True

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        if self.gain_enabled:
            # FP32 gain arithmetic prevents near-one BF16 rounding of the gain.
            result = (result.float() * (1 + self.output_gain)).to(result.dtype)
        delta = self.lora_B(self.input_group(x)) * self.scale
        return (result + delta.to(result.dtype)).to(result.dtype)


def install_output_scales(lm):
    inventory = []
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "self_attn"):
            sites = [(layer.self_attn, "o_proj", "self_attn.o_proj")]
        elif hasattr(layer, "linear_attn"):
            sites = [(layer.linear_attn, "out_proj", "linear_attn.out_proj")]
        else:
            raise ValueError("This installation expects native Qwen attention layouts")
        sites.append((layer.mlp, "down_proj", "mlp.down_proj"))
        for parent, name, label in sites:
            original = getattr(parent, name)
            assert not isinstance(original, OutputScaleProjection)
            wrapped = OutputScaleProjection(original)
            setattr(parent, name, wrapped)
            inventory.append(
                {"layer": i, "projection": label, "parameters": wrapped.output_gain.numel()}
            )
    return inventory


@contextlib.contextmanager
def scales_disabled(lm):
    modules = [m for m in lm.modules() if isinstance(m, OutputScaleProjection)]
    previous = [m.gain_enabled for m in modules]
    try:
        for m in modules:
            m.gain_enabled = False
        yield
    finally:
        for m, enabled in zip(modules, previous):
            m.gain_enabled = enabled


def split_adapter_groups(lm, lr_a=2e-5, lr_b=3.2e-4):
    aa, bb = [], []
    for name, parameter in lm.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A." in name:
            aa.append(parameter)
        elif ".lora_B." in name:
            bb.append(parameter)
        else:
            raise ValueError("Unexpected trainable parameter in LoRA+ control: " + name)
    assert aa and bb
    return [
        {"name": "shared_A", "params": aa, "lr": lr_a},
        {"name": "separate_B", "params": bb, "lr": lr_b},
    ]
