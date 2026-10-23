"""Composable nonlinear programs inside a fixed-rank adapter bottleneck.

delta(x) = 2 B h(Ax). A new residual program starts with a zero gate, so
growing the computation graph preserves the complete parent function.
Graph specifications are data, never executable Python supplied by a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from deciform.adapter_program import STANDARD_PROGRAM
from deciform.adapter_transfer import initialize_adapter

LATENT_PROGRAM = "latent_graph_v1"
ARITY = {"tanh": 1, "silu": 1, "rms": 1, "add": 2, "multiply": 2, "subtract": 2}


def validate_genome(genome: dict) -> dict:
    if not isinstance(genome, dict) or set(genome) != {"version", "blocks"} or genome["version"] != 1:
        raise ValueError("expected a version-1 latent graph")
    if not isinstance(genome["blocks"], list) or not 1 <= len(genome["blocks"]) <= 4:
        raise ValueError("latent program allows one to four residual blocks")
    for block in genome["blocks"]:
        if not isinstance(block, dict) or set(block) != {"nodes", "output"}:
            raise ValueError("each residual block needs nodes and output")
        if not isinstance(block["nodes"], list) or not 1 <= len(block["nodes"]) <= 8:
            raise ValueError("each block allows one to eight graph nodes")
        known = {"input"}
        dependencies = {}
        for node in block["nodes"]:
            if not isinstance(node, dict) or set(node) != {"id", "op", "inputs"}:
                raise ValueError("graph node keys must be id/op/inputs")
            name, op, inputs = node["id"], node["op"], node["inputs"]
            if not isinstance(name, str) or not name.isidentifier() or name in known:
                raise ValueError("graph node IDs must be unique identifiers")
            if not isinstance(op, str) or op not in ARITY or not isinstance(inputs, list) or len(inputs) != ARITY[op]:
                raise ValueError("unknown operation or invalid arity")
            if any(not isinstance(i, str) or i not in known for i in inputs):
                raise ValueError("graph must be acyclic and topologically ordered")
            known.add(name)
            dependencies[name] = inputs
        if block["output"] not in dependencies:
            raise ValueError("block output must be a graph node")
        reachable = set()
        def visit(name):
            if name in reachable or name == "input":
                return
            reachable.add(name)
            for parent in dependencies[name]:
                visit(parent)
        visit(block["output"])
        if reachable != set(dependencies):
            raise ValueError("dead graph nodes are not allowed")
    return genome


def extend_genome(parent: dict | None, block: dict) -> dict:
    return validate_genome({"version": 1, "blocks": [*(parent or {"blocks": []})["blocks"], block]})


def execute_block(block, inputs):
    values = {"input": inputs}
    for node in block["nodes"]:
        arguments = [values[name] for name in node["inputs"]]
        op = node["op"]
        if op == "tanh":
            result = arguments[0].tanh()
        elif op == "silu":
            result = F.silu(arguments[0])
        elif op == "rms":
            result = arguments[0] * (arguments[0].square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        elif op == "add":
            result = arguments[0] + arguments[1]
        elif op == "subtract":
            result = arguments[0] - arguments[1]
        elif op == "multiply":
            result = arguments[0] * arguments[1]
        values[node["id"]] = result
    return values[block["output"]]


class LatentProgramLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, genome: dict):
        super().__init__()
        self.base = base
        self.genome = genome
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.gates = nn.ParameterList([nn.Parameter(self.A.new_zeros(rank)) for _ in genome["blocks"]])

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        latent = F.linear(x.to(self.A.dtype), self.A)
        for block, gate in zip(self.genome["blocks"], self.gates, strict=True):
            latent = latent + gate * execute_block(block, latent)
        result = self.base(x)
        return (result + 2.0 * F.linear(latent, self.B)).to(result.dtype)


def modules(model):
    return {name: layer for name, layer in model.named_modules() if isinstance(layer, LatentProgramLinear)}


def install_latent(model, genome, rank, targets):
    validate_genome(genome)
    if rank < 1:
        raise ValueError("rank must be positive")
    model.requires_grad_(False)
    names = [name for name, layer in model.named_modules()
             if name.rsplit(".", 1)[-1] in targets and isinstance(layer, nn.Linear)]
    if not names:
        raise ValueError("no latent adapter target modules")
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LatentProgramLinear(getattr(parent, leaf), rank, genome))
    return model


def latent_metadata(genome, rank, targets):
    return {"adapter_program": LATENT_PROGRAM, "adapter_program_version": 1,
            "genome": validate_genome(genome), "rank": rank, "target_modules": sorted(targets),
            "scale": 2.0, "new_block_initialization": "zero_residual_gate",
            "format": "jev_rsi_latent_graph_v1"}


def save_latent(model, destination, genome, rank, targets):
    from safetensors.torch import save_file
    destination.mkdir(parents=True, exist_ok=False)
    state = {}
    for name, module in modules(model).items():
        for key, value in module.named_parameters():
            if not key.startswith("base."):
                state[f"{name}.{key}"] = value.detach().cpu().contiguous()
    save_file(state, str(destination / "adapter_model.safetensors"))
    (destination / "adapter_config.json").write_text(json.dumps(latent_metadata(genome, rank, targets), indent=2) + "\n")


def initialize_latent(model, genome, rank, targets, parent_dir: Path):
    from safetensors.torch import load_file
    validate_genome(genome)
    config = json.loads((parent_dir / "adapter_config.json").read_text())
    if config.get("adapter_program") == LATENT_PROGRAM:
        parent_genome = validate_genome(config["genome"])
        if config["rank"] != rank or set(config["target_modules"]) != set(targets):
            raise ValueError("latent parent rank/targets differ")
        old_blocks = parent_genome["blocks"]
        if genome["blocks"][:len(old_blocks)] != old_blocks:
            raise ValueError("function-preserving growth must retain the complete parent graph")
        install_latent(model, genome, rank, targets)
        tensors = load_file(str(parent_dir / "adapter_model.safetensors"))
        expected = set()
        with torch.no_grad():
            for name, module in modules(model).items():
                parameters = dict(module.named_parameters())
                keys = ["A", "B", *[f"gates.{i}" for i in range(len(old_blocks))]]
                for key in keys:
                    full_key = f"{name}.{key}"
                    parameters[key].copy_(tensors[full_key])
                    expected.add(full_key)
        if set(tensors) != expected:
            raise ValueError("unexpected/missing latent parent tensors")
        mode = "continue" if genome == parent_genome else "zero_gate_graph_extension"
    else:
        loaded, _ = initialize_adapter(model, STANDARD_PROGRAM, rank, targets, parent_dir)
        factors = {name: (layer.lora_A.default.weight.detach().clone(),
                          layer.lora_B.default.weight.detach().clone() * layer.scaling["default"] / 2.0)
                   for name, layer in loaded.get_base_model().named_modules()
                   if hasattr(layer, "lora_A") and "default" in layer.lora_A}
        model = loaded.unload()
        install_latent(model, genome, rank, targets)
        if set(factors) != set(modules(model)):
            raise ValueError("latent conversion module names differ")
        with torch.no_grad():
            for name, layer in modules(model).items():
                layer.A.copy_(factors[name][0])
                layer.B.copy_(factors[name][1])
        mode = "linear_to_zero_gate_graph"
    return model, {"mode": mode, "parent": str(parent_dir), "target_program": LATENT_PROGRAM,
                   "genome": genome, "rank": rank, "preserves_parent_function_by_construction": True}
