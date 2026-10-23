import copy
import torch
import pytest
from torch import nn

from deciform.adapter_transfer import initialize_adapter
from deciform.latent_program import initialize_latent, save_latent, modules, validate_genome, extend_genome


def block(op="tanh"):
    return {"nodes": [{"id": "n1", "op": op, "inputs": ["input"]}], "output": "n1"}


def test_graph_growth_and_reload_preserve_parent_then_can_change_predictions(tmp_path):
    torch.manual_seed(8)
    base = nn.Sequential()
    base.add_module("q_proj", nn.Linear(4, 3, bias=False))
    original = copy.deepcopy(base)
    linear, _ = initialize_adapter(base, "standard_lora_v1", 4, ["q_proj"])
    with torch.no_grad():
        for p in linear.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .1)
    path = tmp_path / "linear"
    linear.save_pretrained(path)
    x = torch.randn(5, 4)
    genome = extend_genome(None, block())
    nonlinear, _ = initialize_latent(copy.deepcopy(original), genome, 4, ["q_proj"], path)
    assert torch.allclose(linear(x), nonlinear(x), atol=2e-6, rtol=2e-5)
    nonlinear(x).sum().backward()
    assert any(p.grad.abs().sum() > 0 for m in modules(nonlinear).values() for p in m.gates)
    with torch.no_grad():
        for m in modules(nonlinear).values():
            m.gates[0].fill_(.3)
    assert not torch.allclose(linear(x), nonlinear(x))
    # A nonlinear map cannot be represented by a constant low-rank matrix.
    assert not torch.allclose(nonlinear(2 * x), 2 * nonlinear(x))
    path2 = tmp_path / "parent"
    save_latent(nonlinear, path2, genome, 4, ["q_proj"])
    grown = extend_genome(genome, block("silu"))
    child, audit = initialize_latent(copy.deepcopy(original), grown, 4, ["q_proj"], path2)
    assert audit["mode"] == "zero_gate_graph_extension"
    assert torch.allclose(nonlinear(x), child(x), atol=2e-6, rtol=2e-5)
    assert all(not p.requires_grad for p in child.q_proj.base.parameters())


def test_genome_rejects_cycles_dead_nodes_and_untrusted_operations():
    for bad in [
        {"nodes": [{"id": "n1", "op": "eval", "inputs": ["input"]}], "output": "n1"},
        {"nodes": [{"id": "n1", "op": "tanh", "inputs": ["n1"]}], "output": "n1"},
        {"nodes": [{"id": "n1", "op": "tanh", "inputs": ["input"]},
                   {"id": "n2", "op": "silu", "inputs": ["input"]}], "output": "n1"},
    ]:
        with pytest.raises(ValueError):
            validate_genome({"version": 1, "blocks": [bad]})
