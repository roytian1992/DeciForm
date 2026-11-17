"""Shared and public-task-type private subspaces for full ModernBERT training."""

import contextlib
import contextvars
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
from .modern_core import ModernCore, directed_masks
from . import modern_shared as shared
from . import modern_scale as scaled

PUBLIC_TYPES = ("choice", "noul", "score")
_SUBSPACE_CONTEXT = contextvars.ContextVar("deciform_public_type_subspace_mask", default=None)


def type_masks(shared_rank, private_rank, device=None):
    if shared_rank < 1 or private_rank < 1:
        raise ValueError("Shared and private ranks must be positive")
    masks = torch.zeros(
        len(PUBLIC_TYPES), shared_rank + len(PUBLIC_TYPES) * private_rank, device=device
    )
    masks[:, :shared_rank] = 1
    for t in range(len(PUBLIC_TYPES)):
        start = shared_rank + t * private_rank
        masks[t, start : start + private_rank] = 1
    return masks


class SharedPrivateLayer(ModernBertEncoderLayer):
    def forward(self, *args, core_subspace_mask=None, **kwargs):
        if core_subspace_mask is None:
            raise ValueError(
                "Public request-type mask required during forward and checkpoint replay"
            )
        token = _SUBSPACE_CONTEXT.set(core_subspace_mask)
        try:
            return super().forward(*args, **kwargs)
        finally:
            _SUBSPACE_CONTEXT.reset(token)


class SharedPrivateProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        if type(original) not in (shared.FusedSharedProjection, scaled.OutputScaleProjection):
            raise TypeError("Expected a baseline shared-A projection")
        for name in ("base_layer", "lora_A", "lora_B", "dropout"):
            setattr(self, name, getattr(original, name))
        for name in ("freeze_base", "in_features", "out_features", "scale"):
            setattr(self, name, getattr(original, name))
        if hasattr(original, "output_gain"):
            self.output_gain = original.output_gain
            self.gain_enabled = original.gain_enabled
        self.routing_enabled = True

    def forward(self, x):
        result = self.base_layer(x)
        if hasattr(self, "output_gain") and self.gain_enabled:
            result = (result.float() * (1 + self.output_gain)).to(result.dtype)
        latent = self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))
        if self.routing_enabled:
            mask = _SUBSPACE_CONTEXT.get()
            if mask is None or latent.ndim != 3 or mask.shape != (len(latent), latent.size(-1)):
                raise ValueError("Missing or incompatible public type mask")
            latent = latent * mask[:, None, :].to(latent.dtype)
        return result + (self.lora_B(latent) * self.scale).to(result.dtype)


class SharedPrivateCore(ModernCore):
    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        position_ids,
        option_ids,
        items,
    ):
        # This enum encodes only the explicitly supplied Choice/Noul/Score type.
        # No dataset name, example ID, task-instance ID, target or label is read.
        types = [PUBLIC_TYPES.index(it["task"]) for it in items]
        mask = self.public_type_masks[torch.tensor(types, device=input_ids.device)]
        masks = directed_masks(
            attention_mask, position_ids, option_ids, self.encoder.config.sliding_window
        )
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=masks,
            position_ids=position_ids,
            core_subspace_mask=mask,
        ).last_hidden_state
        qs = torch.tensor([it["query_position"] for it in items], device=h.device)
        context = h[torch.arange(len(items), device=h.device), qs]
        q = self.q(context)
        result = [None] * len(items)
        score_indices = []
        for i, it in enumerate(items):
            if it["task"] == "score":
                score_indices.append(i)
            else:
                k = len(it["support"])
                candidate = h[i, marker_pos[i, :k]]
                result[i] = (self.k(candidate) @ q[i]).float() * self.scale
        if score_indices:
            hs = self.numeric_transform(context[score_indices])
            hs = torch.cat([hs.float(), hs.new_ones(len(hs), 1).float()], -1)
            scores = self.numeric_score(
                hs, q[score_indices], [items[i]["support"] for i in score_indices]
            )
            for i, z in zip(score_indices, scores):
                result[i] = z
        return (
            torch.stack([F.pad(z, (0, marker_mask.size(1) - len(z)), value=-1e4) for z in result]),
            None,
        )


def _expand_projection(original, rank, seed):
    """Keep all baseline coordinates; add locally initialized input directions.

    A dedicated generator leaves the training/dropout RNG untouched. Alpha/r
    stays at the original value rather than silently rescaling the old path.
    """
    old_rank = original.lora_A.out_features
    if rank == old_rank:
        return
    if rank < old_rank:
        raise ValueError("Cannot discard baseline coordinates during expansion")
    weight = original.lora_A.weight
    extra = weight.new_empty(rank - old_rank, original.in_features)
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    nn.init.kaiming_uniform_(extra, a=math.sqrt(5), generator=generator)
    a = nn.Linear(original.in_features, rank, bias=False, device="meta")
    b = nn.Linear(rank, original.out_features, bias=False, device="meta")
    a.weight = nn.Parameter(torch.cat([weight.detach(), extra], 0))
    b.weight = nn.Parameter(
        torch.cat(
            [
                original.lora_B.weight.detach(),
                original.lora_B.weight.new_zeros(original.out_features, rank - old_rank),
            ],
            1,
        )
    )
    original.lora_A, original.lora_B = a, b


def baseline_adapter_parameters(model, baseline_rank=12):
    for name, p in model.encoder.named_parameters():
        if ".lora_A." in name:
            yield name, p[:baseline_rank]
        elif ".lora_B." in name:
            yield name, p[:, :baseline_rank]


def install_shared_private(
    model, shared_rank=6, private_rank=2, allow_expansion=False, seed=20261101
):
    if type(model) is not ModernCore:
        raise ValueError("Install once on the baseline ModernCore")
    rank = shared_rank + len(PUBLIC_TYPES) * private_rank
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inventory = []
    expected_extra = 0
    for i, layer in enumerate(model.encoder.layers):
        if type(layer) is not ModernBertEncoderLayer:
            raise TypeError("Unexpected encoder layer")
        for parent, name in [("attn", "Wqkv"), ("attn", "Wo"), ("mlp", "Wi"), ("mlp", "Wo")]:
            owner = getattr(layer, parent)
            projection = getattr(owner, name)
            if projection.lora_A.out_features != rank and not allow_expansion:
                raise ValueError("Shared + 3*private must equal the existing rank budget")
            old_rank = projection.lora_A.out_features
            if rank < old_rank or (allow_expansion and shared_rank < old_rank):
                raise ValueError("Expanded arm must retain all original shared coordinates")
            expected_extra += (rank - old_rank) * (projection.in_features + projection.out_features)
            _expand_projection(projection, rank, seed + 104729 * len(inventory))
            setattr(owner, name, SharedPrivateProjection(projection))
            inventory.append(
                {
                    "layer": i,
                    "projection": parent + "." + name,
                    "shared_rank": shared_rank,
                    "private_rank_per_public_type": private_rank,
                    "stored_rank": rank,
                    "active_rank": shared_rank + private_rank,
                    "scaling": projection.scale,
                    "adapter_parameters": projection.lora_A.weight.numel()
                    + projection.lora_B.weight.numel(),
                }
            )
        layer.__class__ = SharedPrivateLayer
    model.__class__ = SharedPrivateCore
    device = next(model.parameters()).device
    model.register_buffer(
        "public_type_masks", type_masks(shared_rank, private_rank, device), persistent=False
    )
    model.subspace_inventory = inventory
    model.shared_rank, model.private_rank = shared_rank, private_rank
    assert before + expected_extra == sum(p.numel() for p in model.parameters() if p.requires_grad)
    return inventory


@contextlib.contextmanager
def routing_disabled(model):
    modules = [m for m in model.modules() if isinstance(m, SharedPrivateProjection)]
    previous = [m.routing_enabled for m in modules]
    try:
        for m in modules:
            m.routing_enabled = False
        yield
    finally:
        for m, v in zip(modules, previous):
            m.routing_enabled = v


def adapter_gradient_report(model):
    """Summarize actual gradients by shared/private block, never model features."""
    result = {
        k: {"A_squared": 0.0, "B_squared": 0.0, "A_parameters": 0, "B_parameters": 0}
        for k in ("shared",) + PUBLIC_TYPES
    }
    slices = {"shared": slice(0, model.shared_rank)}
    for t, name in enumerate(PUBLIC_TYPES):
        start = model.shared_rank + t * model.private_rank
        slices[name] = slice(start, start + model.private_rank)
    for m in model.modules():
        if not isinstance(m, SharedPrivateProjection):
            continue
        for name, sl in slices.items():
            for letter, parameter, selected in [
                ("A", m.lora_A.weight, (sl, slice(None))),
                ("B", m.lora_B.weight, (slice(None), sl)),
            ]:
                result[name][letter + "_parameters"] += parameter[selected].numel()
                if parameter.grad is not None:
                    result[name][letter + "_squared"] += float(
                        parameter.grad[selected].float().square().sum()
                    )
    return result
