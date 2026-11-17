"""Candidate-independent Qwen3.5 branches with a differentiable shared prefix.

The prefix contains state + question only. One query branch reads <decide>;
each other branch reads one complete <opt> ... </opt> span. All branches start
from the same attention KV, convolution and recurrent states and positions.
The optional private-residual enable flag is captured for checkpoint replay.
No task router is installed; only the supplied public type is used.
"""

from contextlib import contextmanager
from .task_context import task_context
from .private_residual import private_context, _ENABLED
import json

import torch
from torch.utils.checkpoint import checkpoint

from ..integrations.kev_prefix import Prefix, _masks


def split_candidates(enc):
    if len(enc["opt_idx"]) != 1 or len(enc["tasks"]) != 1 or enc["state_truncated"]:
        raise ValueError("Expected one complete question, without truncation")
    positions = [i for i, value in enumerate(enc["opt"]) if value >= 0]
    first, last = positions[0], positions[-1] + 1
    if enc["decide_idx"] != [last] or last + 1 != len(enc["ids"]):
        raise ValueError("Expected option spans followed by one decide token")
    branches, semantics = [], []
    start = first
    while start < last:
        semantic = enc["opt"][start]
        end = start + 1
        while end < last and enc["opt"][end] == semantic:
            end += 1
        if semantic < 0 or enc["opt_idx"][0][semantic] != end - 1:
            raise ValueError("Candidate span/readout mismatch")
        branches.append(enc["ids"][start:end])
        semantics.append(semantic)
        start = end
    if sorted(semantics) != list(range(len(enc["opt_idx"][0]))):
        raise ValueError("Candidates must appear exactly once")
    prefix = enc["ids"][:first]
    decide = enc["ids"][last]
    assert prefix + [token for branch in branches for token in branch] + [decide] == enc["ids"]
    return {
        "prefix": prefix,
        "branches": branches,
        "semantics": semantics,
        "query_token": decide,
        "tasks": enc["tasks"],
        "original_tokens": len(enc["ids"]),
    }


def cost(enc, arm=None):
    return 1, enc["original_tokens"]


def permute_candidates(enc, order):
    if sorted(order) != list(range(len(enc["branches"]))):
        raise ValueError("A complete candidate permutation is required")
    return {
        **enc,
        "branches": [enc["branches"][i] for i in order],
        "semantics": [enc["semantics"][i] for i in order],
    }


@contextmanager
def routing(router, tasks, dtype, indices=None):
    if router is not None:
        raise ValueError("Only explicit public-type routing is supported")
    yield


def _layer_step(
    layer, router, hs, hb, ropes, masks, positions, real, owner, tasks, private_enabled
):
    # Local cache and routing contexts must be recreated during checkpoint replay.
    prefix = Prefix()
    dtype = (
        torch.get_autocast_dtype("cuda")
        if hs.is_cuda and torch.is_autocast_enabled("cuda")
        else hs.dtype
    )
    indices = getattr(layer, "_independent_route_indices", None)
    if indices is None and router is not None:
        indices = tuple(
            module.route_index for module in layer.modules() if hasattr(module, "route_index")
        )
        layer._independent_route_indices = indices
    with (
        routing(router, tasks, dtype, indices),
        task_context(tasks),
        private_context(private_enabled),
    ):
        hs = layer.forward(
            hs,
            position_embeddings=ropes[0],
            attention_mask=masks[0],
            position_ids=positions[0],
            past_key_values=prefix,
        )
    prefix.branch(owner)
    with (
        routing(router, tasks[owner], dtype, indices),
        task_context(tasks[owner]),
        private_context(private_enabled),
    ):
        hb = layer.forward(
            hb,
            position_embeddings=ropes[1],
            attention_mask=masks[1],
            position_ids=positions[1],
            past_key_values=prefix,
        )
    return hs * real[..., None], hb


def shared_hidden(lm, encs, pad_id, tasks, router=None):
    """Return prefix-only query and candidate hidden states, in semantic order."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    device = next(base.parameters()).device
    all_branches, owners, ranges = [], [], []
    for i, enc in enumerate(encs):
        begin = len(all_branches)
        branches = [[enc["query_token"]], *enc["branches"]]
        all_branches.extend(branches)
        owners.extend([i] * len(branches))
        ranges.append((begin, len(all_branches)))
    owner = torch.tensor(owners, device=device)
    ls = max(len(enc["prefix"]) for enc in encs)
    lb = max(map(len, all_branches))
    si = torch.full((len(encs), ls), pad_id, dtype=torch.long, device=device)
    sp = torch.zeros_like(si)
    sr = torch.zeros_like(si, dtype=torch.bool)
    bi = torch.full((len(all_branches), lb), pad_id, dtype=torch.long, device=device)
    bp = torch.zeros_like(bi)
    br = torch.zeros_like(bi, dtype=torch.bool)
    for i, enc in enumerate(encs):
        n = len(enc["prefix"])
        si[i, -n:] = torch.tensor(enc["prefix"], device=device)
        sp[i, -n:] = torch.arange(n, device=device)
        sr[i, -n:] = True
    for i, branch in enumerate(all_branches):
        n = len(encs[owners[i]]["prefix"])
        bi[i, : len(branch)] = torch.tensor(branch, device=device)
        bp[i, : len(branch)] = torch.arange(n, n + len(branch), device=device)
        br[i, : len(branch)] = True
    hs = base.embed_tokens(si) * sr[..., None]
    hb = base.embed_tokens(bi) * br[..., None]
    dtype, attn = hs.dtype, base.config._attn_implementation
    causal = lambda n: torch.ones(n, n, dtype=torch.bool, device=device).tril()
    eye = lambda n: torch.eye(n, dtype=torch.bool, device=device)
    plain = attn == "sdpa" and bool(sr.all())
    allow_b = torch.cat(
        [
            sr[owner][:, None, :].expand(-1, lb, -1),
            (causal(lb)[None] & br[:, None, :]) | eye(lb)[None],
        ],
        -1,
    )
    ms = {
        "full_attention": (
            None
            if plain
            else _masks((causal(ls)[None] & sr[:, None, :]) | eye(ls)[None], dtype, attn)
        ),
        "linear_attention": sr.to(dtype),
    }
    mb = {"full_attention": _masks(allow_b, dtype, attn), "linear_attention": br.to(dtype)}
    ropes = (
        base.rotary_emb(hs, sp[None].expand(3, -1, -1)),
        base.rotary_emb(hb, bp[None].expand(3, -1, -1)),
    )
    for layer in base.layers:
        args = (
            layer,
            router,
            hs,
            hb,
            ropes,
            (ms[layer.block_type], mb[layer.block_type]),
            (sp, bp),
            sr,
            owner,
            tasks,
            _ENABLED.get(),
        )
        recompute = (
            layer.training
            and torch.is_grad_enabled()
            and getattr(layer, "gradient_checkpointing", False)
        )
        hs, hb = (
            checkpoint(_layer_step, *args, use_reentrant=False) if recompute else _layer_step(*args)
        )
    # Apply the final norm only at the actual readout positions.
    ends = torch.tensor([len(branch) - 1 for branch in all_branches], device=device)
    readouts = base.norm(hb[torch.arange(len(all_branches), device=device), ends]).float()
    queries, options = [], []
    for enc, (begin, end) in zip(encs, ranges):
        queries.append(readouts[begin])
        inverse = sorted(range(len(enc["semantics"])), key=enc["semantics"].__getitem__)
        options.append(readouts[begin + 1 : end][inverse])
    return torch.stack(queries), options


def independent_reference(lm, encs, pad_id, tasks, router=None):
    """Slow repeated-prefix reference, used only to verify logits and gradients."""
    device = next(lm.parameters()).device
    queries, options = [], []
    for i, enc in enumerate(encs):
        values = []
        for branch in [[enc["query_token"]], *enc["branches"]]:
            tokens = torch.tensor([enc["prefix"] + branch], device=device)
            dtype = (
                torch.get_autocast_dtype("cuda")
                if tokens.is_cuda and torch.is_autocast_enabled("cuda")
                else next(lm.parameters()).dtype
            )
            with routing(router, tasks[i : i + 1], dtype), task_context(tasks[i : i + 1]):
                h = lm(
                    input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
                ).last_hidden_state
            values.append(h[0, -1].float())
        queries.append(values[0])
        inverse = sorted(range(len(enc["semantics"])), key=enc["semantics"].__getitem__)
        options.append(torch.stack(values[1:])[inverse])
    return torch.stack(queries), options


def stable_argmax(probabilities, semantic_values):
    best = max(probabilities)
    tied = [i for i, value in enumerate(probabilities) if value == best]
    return min(
        tied, key=lambda i: json.dumps(semantic_values[i], ensure_ascii=False, sort_keys=True)
    )
