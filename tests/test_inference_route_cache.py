from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from inference_route_cache import FrozenRouteCache
import independent_candidates as independent
from kev_ours_v5 import anchored_maps


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        router = nn.Module()
        router.matrix_raw = nn.Parameter(torch.randn(2, 3, 2, 2))
        router.register_buffer('task_weights', torch.tensor([.25, .375, .375]))
        router.radius, router.maps, router.batch_size = .5, None, None
        def clear():
            router.maps = router.batch_size = None
        router.clear = clear
        self.net = nn.Module()
        self.net.task_adapter = router
        layer = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        for i, module in enumerate(layer):
            module.route_index = i
        self.lm = types.SimpleNamespace(get_base_model=lambda: types.SimpleNamespace(layers=[layer]))

    def scores(self, rows, encs):
        return None


def test_requires_frozen_model_and_rejects_stale_weights():
    model = Model()
    with pytest.raises(ValueError):
        FrozenRouteCache(model)
    model.eval().requires_grad_(False)
    cache = FrozenRouteCache(model)
    rows = [{'task':'choice'}]
    encs = [{'branches':[[1],[2]]}]
    with torch.no_grad():
        model.net.task_adapter.matrix_raw.add_(.1)
        with pytest.raises(RuntimeError, match='stale'):
            cache.bind(rows, encs)


def test_mixed_tasks_and_branch_counts_cache_exact_constants_and_restore():
    model = Model().eval().requires_grad_(False)
    cache = FrozenRouteCache(model)
    rows = [{'task':'choice'}, {'task':'score'}]
    encs = [{'branches':[[1],[2]]}, {'branches':[[3],[4],[5]]}]
    original = independent.routing
    with torch.no_grad(), cache:
        cache.bind(rows, encs)
        router = model.net.task_adapter
        matrices = anchored_maps(router.matrix_raw[[0,1]], router.task_weights, router.radius)
        for ids in ([0,2], [0,0,0,2,2,2,2]):
            tasks = torch.tensor(ids)
            with independent.routing(router, tasks, torch.float32, (0,1)):
                for i in (0,1):
                    assert torch.equal(router.maps[i], matrices[i,tasks])
            assert router.maps is None
        cache.current = None
        cache.bind(rows, encs)
        assert cache.hits == 1 and cache.misses == 1
    assert independent.routing is original
    assert 'scores' not in model.__dict__


def test_restores_original_routing_on_error():
    model = Model().eval().requires_grad_(False)
    original = independent.routing
    with pytest.raises(RuntimeError, match='simulated'):
        with FrozenRouteCache(model):
            raise RuntimeError('simulated')
    assert independent.routing is original
    assert 'scores' not in model.__dict__
