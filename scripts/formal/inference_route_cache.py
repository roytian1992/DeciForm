"""Opt-in frozen D2 route cache; no matrix reassociation or changed head.

Only task-dependent constants are cached, never text, activations, or answers.
This diagnostic context is process-wide and deliberately not a threaded server.
Original routing is restored on exit. Training and stale route weights fail.
"""
import collections
from contextlib import contextmanager
import types

import torch

from common import TASKS
import independent_candidates as independent
from kev_ours_v5 import anchored_maps


class FrozenRouteCache:
    def __init__(self, model, max_contexts=32):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Freeze weights and set eval before constructing inference cache')
        self.model = model
        self.router = model.net.task_adapter
        self.max_contexts = max_contexts
        self.original_routing = independent.routing
        self.original_scores = model.scores
        self.score_override = model.__dict__.get('scores')
        self.signature = self.weight_signature()
        self.constants = {}
        self.contexts = collections.OrderedDict()
        self.current = None
        self.call = 0
        self.hits = self.misses = 0
        base = model.lm.get_base_model()
        with torch.no_grad():
            for layer in base.layers:
                indices = tuple(m.route_index for m in layer.modules() if hasattr(m, 'route_index'))
                # Use the exact original per-layer operation and shape.
                self.constants[indices] = anchored_maps(self.router.matrix_raw[list(indices)],
                                                       self.router.task_weights, self.router.radius)
        self.layers = len(base.layers)

    def weight_signature(self):
        return tuple((id(p), p._version, str(p.device), p.dtype) for p in
                     (self.router.matrix_raw, self.router.task_weights)) + (self.router.radius,)

    def bind(self, rows, encs):
        if self.model.training or torch.is_grad_enabled() or self.weight_signature() != self.signature:
            raise RuntimeError('Frozen inference cache is stale or used in training')
        if self.current is not None or self.router.maps is not None:
            raise RuntimeError('Concurrent inference contexts are unsupported')
        prefix = tuple(TASKS.index(r['task']) for r in rows)
        branches = tuple(t for t, enc in zip(prefix, encs) for _ in range(len(enc['branches'])+1))
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else next(self.model.parameters()).dtype
        key = (prefix, branches, dtype)
        if key in self.contexts:
            self.hits += 1
            self.contexts.move_to_end(key)
        else:
            self.misses += 1
            device = self.router.matrix_raw.device
            contexts = []
            for tasks in (prefix, branches):
                task_ids = torch.tensor(tasks, device=device)
                contexts.append({indices: {index: matrices[j, task_ids].to(dtype)
                                           for j, index in enumerate(indices)}
                                 for indices, matrices in self.constants.items()})
            self.contexts[key] = contexts
            while len(self.contexts) > self.max_contexts:
                self.contexts.popitem(last=False)
        self.current = self.contexts[key]
        self.call = 0

    @contextmanager
    def routing(self, router, tasks, dtype, indices=None):
        if router is not self.router or indices is None:
            with self.original_routing(router, tasks, dtype, indices):
                yield
            return
        if self.current is None or router.maps is not None:
            raise RuntimeError('No frozen per-request routing context')
        maps = self.current[self.call % 2][tuple(indices)]
        reference = next(iter(maps.values()))
        if len(reference) != len(tasks) or reference.dtype != dtype:
            raise RuntimeError('Cached task routing shape/dtype mismatch')
        router.maps, router.batch_size = maps, len(tasks)
        self.call += 1
        try:
            yield
        finally:
            router.clear()

    def scores(self, model, rows, encs):
        self.bind(rows, encs)
        try:
            result = self.original_scores(rows, encs)
            if self.call != self.layers * 2:
                raise RuntimeError('Unexpected D2 execution pattern')
            return result
        finally:
            self.current = None
            self.router.clear()

    def __enter__(self):
        if independent.routing is not self.original_routing:
            raise RuntimeError('Another routing override is active')
        independent.routing = self.routing
        self.model.scores = types.MethodType(self.scores, self.model)
        return self

    def __exit__(self, *exc):
        independent.routing = self.original_routing
        if self.score_override is None:
            del self.model.scores
        else:
            self.model.scores = self.score_override
        self.current = None
        self.router.clear()
        return False
