"""Finite-support numeric string probabilities with a pretrained root prior.

Only the root reuses an actual LM next-token logit. Subsequent digits use a
small trainable GRU, not a claimed approximation to cached backbone decoding.
All legal values are enumerated; no beam pruning or target-conditioned input.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..objectives import SpecialistObjective, data_loss as ordinal_loss, interpolation

ALPHABET = "0123456789.}"


@lru_cache(maxsize=128)
def numeric_tree(values):
    if len(values) < 2 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Finite nonnegative score supports with at least two values required")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("The full numeric support must be strictly increasing")
    strings = [format(v, ".12g") + "}" for v in values]
    if len(set(strings)) != len(values) or any(c not in ALPHABET for s in strings for c in s):
        raise ValueError("Score values need unique ordinary decimal representations")
    children = {}
    paths = []
    for string in strings:
        path = []
        prefix = ()
        for character in string:
            token = ALPHABET.index(character)
            children.setdefault(prefix, set()).add(token)
            path.append((prefix, token))
            prefix += (token,)
        paths.append(path)
    branches = {p: tuple(sorted(v)) for p, v in children.items() if len(v) > 1}
    needed = {()}
    for p in branches:
        needed.update(p[:i] for i in range(len(p) + 1))
    return (
        tuple(strings),
        branches,
        tuple(sorted(needed, key=lambda p: (len(p), p))),
        tuple(tuple((p, t) for p, t in path if p in branches) for path in paths),
    )


class NumericTrieReadout(nn.Module):
    def __init__(self, output_rows, input_rows, query_size=256):
        super().__init__()
        if output_rows.shape != input_rows.shape or output_rows.shape[0] != len(ALPHABET):
            raise ValueError("One frozen input/output row per numeric character required")
        self.register_buffer("output_rows", output_rows.detach().float().clone())
        self.register_buffer("input_rows", input_rows.detach().float().clone())
        self.input_projection = nn.Linear(input_rows.shape[1], query_size, bias=False)
        self.context_norm = nn.LayerNorm(query_size)
        self.cell = nn.GRUCell(query_size, query_size)
        self.readout = nn.Linear(query_size, len(ALPHABET))
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _group(self, hidden, query, values):
        _, branches, needed, paths = numeric_tree(tuple(values))
        states = {(): self.context_norm(query)}
        embedded = self.input_projection(self.input_rows)
        # Batch all prefixes of each depth. Backbone is evaluated only once.
        for depth in range(1, max(map(len, needed)) + 1):
            prefixes = [p for p in needed if len(p) == depth]
            parent = torch.cat([states[p[:-1]] for p in prefixes], dim=0)
            inputs = torch.cat([embedded[p[-1]][None].expand(len(hidden), -1) for p in prefixes])
            next_states = self.cell(inputs, parent).split(len(hidden))
            states.update(zip(prefixes, next_states))
        root_prior = F.linear(hidden, self.output_rows)
        log_edges = {}
        for prefix, tokens in branches.items():
            logits = self.readout(states[prefix])
            if not prefix:
                logits = logits + root_prior
            logp = logits[:, list(tokens)].log_softmax(-1)
            log_edges.update({(prefix, token): logp[:, j] for j, token in enumerate(tokens)})
        return torch.stack(
            [torch.stack([log_edges[e] for e in path]).sum(0) for path in paths], dim=-1
        )

    def forward(self, hidden, query, supports):
        if len(hidden) != len(query) or len(hidden) != len(supports):
            raise ValueError("Paired hidden states, queries and supports required")
        groups = {}
        for i, values in enumerate(supports):
            groups.setdefault(tuple(values), []).append(i)
        outputs = [None] * len(hidden)
        # These small numeric computations stay FP32, including long-path logs.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for values, indices in groups.items():
                scores = self._group(hidden[indices].float(), query[indices].float(), values)
                for i, score in zip(indices, scores):
                    outputs[i] = score
        return outputs


@dataclass(frozen=True)
class TrieObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127
    cdf_weight: float = 0.5

    def __post_init__(self):
        if self.kind not in ("trie_ce", "trie_ce_cdf"):
            raise ValueError("Unknown numeric trie objective")
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale <= 0:
            raise ValueError("Positive finite ordinal scale required")
        if self.cdf_weight != (0.0 if self.kind == "trie_ce" else 0.5):
            raise ValueError("Frozen CE arm or equal-weight CE/CDF mixture required")

    def inventory(self):
        return asdict(self)


def components(logits, values, label, objective):
    ce = ordinal_loss(logits, values, label, SpecialistObjective("soft_ce"))
    cdf = ordinal_loss(
        logits, values, label, SpecialistObjective("soft_cdf", objective.ordinal_scale)
    )
    total = (1 - objective.cdf_weight) * ce + objective.cdf_weight * cdf
    return total, ce, cdf


def logit_diagnostics(logits, values, label, objective):
    with torch.enable_grad():
        z = logits.detach().float().requires_grad_(True)
        total, ce, _ = components(z, values, label, objective)
        grad = torch.autograd.grad(total, z, retain_graph=True)[0]
        ce_grad = torch.autograd.grad(ce, z)[0]
    p = z.detach().softmax(0)
    q = p.new_tensor(interpolation(values, label))
    return torch.stack(
        (
            p.new_ones(()),
            total.detach(),
            ce.detach(),
            grad.norm(),
            ce_grad.norm(),
            (p * q).sum(),
            p.new_tensor(len(p)),
        )
    ).double()
