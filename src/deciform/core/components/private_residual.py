"""Optional additive private adapters; original parameter paths remain intact.

Only the public Choice/Noul/Score enum selects a rank-r residual. No routing
touches the original rank-12 shared adapter. Zero B makes the baseline a strict
initial submodel. Local initialization and no extra dropout preserve its RNG.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .task_context import _TASKS, task_context

PUBLIC_TYPES = ("choice", "noul", "score")
_ENABLED = ContextVar("optional_private_residual_enabled", default=True)


@contextmanager
def private_context(enabled):
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


class PrivateResidual(nn.Module):
    """Independent factors for three public types; computes only chosen rank."""

    def __init__(self, in_features, out_features, rank=2, scale=2.0, seed=0, device=None):
        super().__init__()
        if rank < 1 or scale <= 0:
            raise ValueError("Positive rank and scale required")
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.scale = rank, float(scale)
        a = torch.empty(3, rank, in_features, device=device, dtype=torch.float32)
        generator = torch.Generator(device=a.device).manual_seed(seed)
        # Same input variance as nn.Linear, separately for each public type.
        a.uniform_(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features), generator=generator)
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(
            torch.zeros(3, out_features, rank, device=device, dtype=torch.float32)
        )

    def forward(self, x):
        types = _TASKS.get()
        if (
            types is None
            or types.dtype != torch.long
            or types.ndim != 1
            or len(types) != len(x)
            or x.ndim != 3
        ):
            raise ValueError("Explicit public Choice/Noul/Score type required for each stream")
        # Index selection keeps gradients of other types exactly zero. The
        # baseline dropout and its global random stream are never consumed here.
        a, b = self.A[types], self.B[types]
        latent = torch.bmm(x.to(a.dtype), a.transpose(1, 2))
        return torch.bmm(latent, b.transpose(1, 2)) * self.scale


def _add_residual(module, inputs, output):
    if not _ENABLED.get():
        return output  # no private matmul, no extra RNG, same tensor object
    return output + module.private_residual(inputs[0]).to(output.dtype)


def attach_private_residual(projection, rank=2, scale=2.0, seed=0):
    """Reusable projection integration for Qwen and ModernBERT-style outputs."""
    if hasattr(projection, "private_residual"):
        raise ValueError("Private residual already attached")
    before = {n: id(p) for n, p in projection.named_parameters()}
    p = next(projection.parameters())
    projection.private_residual = PrivateResidual(
        projection.in_features, projection.out_features, rank, scale, seed, p.device
    )
    handle = projection.register_forward_hook(_add_residual)
    object.__setattr__(projection, "_private_residual_hook", handle)
    assert all(id(dict(projection.named_parameters())[n]) == pid for n, pid in before.items())


def remove_private_residual(projection):
    """Remove the extra module without changing original parameter names/values."""
    projection._private_residual_hook.remove()
    del projection._private_residual_hook
    del projection.private_residual


def optional_checkpoint(function, *args, **kwargs):
    types, enabled = _TASKS.get(), _ENABLED.get()
    if types is None:
        raise ValueError("Checkpoint requires public types")

    @contextmanager
    def context():
        with task_context(types), private_context(enabled):
            yield

    kwargs["use_reentrant"] = False
    kwargs["context_fn"] = lambda: (context(), context())
    return checkpoint(function, *args, **kwargs)


def install_qwen_private_residual(lm, last_layers=8, rank=2, scale=2.0, seed=20261101):
    from .adapter_mechanisms import OutputScaleProjection

    if not 1 <= last_layers <= len(lm.layers):
        raise ValueError("Invalid number of upper layers")
    sites = []
    for i in range(len(lm.layers) - last_layers, len(lm.layers)):
        layer = lm.layers[i]
        parent, key = (
            (layer.self_attn, "o_proj")
            if hasattr(layer, "self_attn")
            else (layer.linear_attn, "out_proj")
        )
        sites.extend([(i, parent, key), (i, layer.mlp, "down_proj")])
    if any(
        not isinstance(getattr(p, k), OutputScaleProjection)
        or hasattr(getattr(p, k), "private_residual")
        for _, p, k in sites
    ):
        raise ValueError("Expected intact output-scale projections, install once")
    inventory = []
    for i, parent, key in sites:
        projection = getattr(parent, key)
        attach_private_residual(projection, rank, scale, seed + 104729 * len(inventory))
        inventory.append(
            {
                "layer": i,
                "projection": key,
                "private_rank_per_type": rank,
                "shared_rank_unchanged": projection.lora_B.in_features,
                "in_features": projection.in_features,
                "out_features": projection.out_features,
                "new_parameters": sum(p.numel() for p in projection.private_residual.parameters()),
            }
        )
    for layer in lm.layers:
        layer._gradient_checkpointing_func = optional_checkpoint
    return inventory


def baseline_parameters(model):
    return ((n, p) for n, p in model.named_parameters() if ".private_residual." not in n)


def private_parameters(model):
    return ((n, p) for n, p in model.named_parameters() if ".private_residual." in n)


def private_gradient_report(lm):
    result = {
        t: {"A_squared": 0.0, "B_squared": 0.0, "A_parameters": 0, "B_parameters": 0}
        for t in PUBLIC_TYPES
    }
    for module in lm.modules():
        if not isinstance(module, PrivateResidual):
            continue
        for i, t in enumerate(PUBLIC_TYPES):
            for letter in ("A", "B"):
                p = getattr(module, letter)
                result[t][letter + "_parameters"] += p[i].numel()
                if p.grad is not None:
                    result[t][letter + "_squared"] += float(p.grad[i].float().square().sum())
    return result
