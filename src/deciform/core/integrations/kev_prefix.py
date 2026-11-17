"""Adapted from receptron/Kev shared_prefix.py; Apache-2.0.
Only Prefix and mask conversion are included. See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch.nn import functional as F


class Prefix:
    """One decoder layer's view of the shared state, duck-typing the cache calls a Qwen3.5 layer makes. The state pass
    stores what the layer leaves (keys and values, or the conv window and the recurrent state); `branch` gathers it once
    per branch row; the branch pass reads it. Nothing is written in place."""

    record_past = True  # keeps a one-token pass off the in-place decode kernel

    def __init__(self):
        self.keys = self.values = self.conv = None
        self.recurrent_states = {}

    @property
    def layers(
        self,
    ):  # layer code reads `cache.layers[layer_idx].recurrent_states[0]`: every index is this prefix
        return self

    def __getitem__(self, layer_idx):
        return self

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return self.conv is not None

    def update_conv_state(self, conv_input, layer_idx, conv_kernel_size, **kwargs):
        if self.conv is not None:
            return torch.cat([self.conv, conv_input], -1)  # the branch sees the state's last inputs
        self.conv = F.pad(conv_input, (max(0, conv_kernel_size - conv_input.shape[-1]), 0))[
            ..., -conv_kernel_size:
        ]
        return conv_input

    def update_recurrent_state(self, recurrent_state, layer_idx, **kwargs):
        self.recurrent_states.setdefault(
            0, recurrent_state
        )  # the state's final state; a branch's own is discarded
        return recurrent_state

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if self.keys is None:
            self.keys, self.values = keys, values
            return keys, values
        return torch.cat([self.keys, keys], -2), torch.cat([self.values, values], -2)

    def branch(self, owner):
        """Gather the stored state per branch row (owner[i] = the record of branch i)."""
        pick = lambda t: None if t is None else t.index_select(0, owner)
        self.keys, self.values, self.conv = pick(self.keys), pick(self.values), pick(self.conv)
        self.recurrent_states = {k: pick(v) for k, v in self.recurrent_states.items()}


def _masks(allow, dtype, attn):
    """Additive float mask for eager attention, boolean for SDPA (which also takes a float mask, but only in the query's
    dtype, and autocast changes that)."""
    allow = allow[:, None]
    return (
        allow
        if attn == "sdpa"
        else torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(
            ~allow, torch.finfo(dtype).min
        )
    )
