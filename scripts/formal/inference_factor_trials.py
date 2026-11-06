"""Reversible inference-only factor experiments; never write model checkpoints."""
import types

import torch
from torch.nn import functional as F

from common import TASKS
from kev_ours_v5 import MatrixRoutedLinear


def routed_modules(model):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen eval model required')
    return [m for m in model.lm.modules() if isinstance(m, MatrixRoutedLinear)]


class PrecastFactors:
    """Use the same BF16 values autocast otherwise constructs on every call."""
    def __init__(self, model):
        self.parameters = []
        for module in routed_modules(model):
            adapter = module.active_adapters[0]
            self.parameters.extend([module.lora_A[adapter].weight, module.lora_B[adapter].weight])
        self.original = [p.data for p in self.parameters]
        self.converted = [p.detach().to(torch.bfloat16) for p in self.parameters]

    def __enter__(self):
        for p, value in zip(self.parameters, self.converted):
            p.data = value
        return self

    def __exit__(self, *exc):
        for p, value in zip(self.parameters, self.original):
            p.data = value
        return False


def fold_factor(a, b, r, side):
    """Column convention: B(I+R)A; compute a frozen factor in FP32."""
    if side == 'a':
        return a + r @ a
    if side == 'b':
        return b + b @ r
    raise ValueError(side)


class FoldedFactors:
    """Algebraic equivalence only: BF16 operation ordering changes."""
    def __init__(self, model, cache, side):
        self.model, self.cache, self.side = model, cache, side
        self.modules = routed_modules(model)
        self.originals = [(m, m.__dict__.get('forward')) for m in self.modules]
        self.task = None
        self.factors = {}
        matrices = {index: value[j] for indices, value in cache.constants.items()
                    for j, index in enumerate(indices)}
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            for module in self.modules:
                adapter = module.active_adapters[0]
                a = module.lora_A[adapter].weight.detach().to(torch.bfloat16)
                b = module.lora_B[adapter].weight.detach().to(torch.bfloat16)
                maps = matrices[module.route_index].to(torch.bfloat16).float()
                merged = torch.stack([fold_factor(a.float(), b.float(), r, side).to(torch.bfloat16) for r in maps])
                self.factors[module.route_index] = (merged, a, b)
        self.original_scores = None

    def forward(self, module, x, *args, **kwargs):
        if self.task is None or module.training or torch.is_grad_enabled():
            raise RuntimeError('Frozen single-task inference required')
        adapter = module.active_adapters[0]
        base = module.base_layer(x, *args, **kwargs)
        merged, a, b = self.factors[module.route_index]
        if self.side == 'a':
            a = merged[self.task]
        else:
            b = merged[self.task]
        low = F.linear(x.to(a.dtype), a)
        return (base + F.linear(low, b) * module.scaling[adapter]).to(base.dtype)

    def scores(self, model, rows, encs):
        tasks = {TASKS.index(r['task']) for r in rows}
        if len(tasks) != 1 or self.task is not None:
            raise RuntimeError('This small experiment requires single-task request batches')
        self.task = tasks.pop()
        try:
            return self.original_scores(rows, encs)
        finally:
            self.task = None

    def __enter__(self):
        self.original_scores = self.model.scores
        self.score_override = self.model.__dict__.get('scores')
        self.model.scores = types.MethodType(self.scores, self.model)
        for m in self.modules:
            m.forward = types.MethodType(self.forward, m)
        return self

    def __exit__(self, *exc):
        for m, original in self.originals:
            if original is None:
                del m.forward
            else:
                m.forward = original
        if self.score_override is None:
            del self.model.scores
        else:
            self.model.scores = self.score_override
        self.task = None
        return False
