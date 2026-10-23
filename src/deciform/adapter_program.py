"""Bounded adapter programs used by the recursive adapter search.

The registry deliberately keeps the rank and target modules outside the
program.  They are part of the fixed parameter budget; a program can only
change how that budget is factorised and combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


STANDARD_PROGRAM = "standard_lora_v1"
PARALLEL_PROGRAM = "parallel_lora_v1"
GATED_PARALLEL_PROGRAM = "gated_parallel_lora_v1"
ADAPTER_PROGRAMS = (STANDARD_PROGRAM, PARALLEL_PROGRAM, GATED_PARALLEL_PROGRAM)
SYNTH_PREFIX = "synth_v1"
_SYNTH_BRANCHES = (2, 4)
_SYNTH_MERGES = ("sum", "gated_sum")
_SYNTH_NORMALIZATIONS = ("none", "column_rms", "row_rms")


def synthesized_program_name(branch_count: int, merge: str, normalization: str) -> str:
    """Return the canonical name of a compiler-generated program genome."""
    return f"{SYNTH_PREFIX}:b{branch_count}:merge={merge}:norm={normalization}"


SYNTHESIZED_PROGRAMS = tuple(
    synthesized_program_name(branches, merge, normalization)
    for branches in _SYNTH_BRANCHES
    for merge in _SYNTH_MERGES
    for normalization in _SYNTH_NORMALIZATIONS
)
ADAPTER_PROGRAMS = ADAPTER_PROGRAMS + SYNTHESIZED_PROGRAMS


@dataclass(frozen=True)
class AdapterProgramSpec:
    name: str
    branch_count: int
    gated: bool
    normalized: bool
    implementation: str
    merge: str = "sum"
    normalization: str = "none"
    origin: str = "registry"


_REGISTRY = {
    STANDARD_PROGRAM: AdapterProgramSpec(
        STANDARD_PROGRAM, branch_count=1, gated=False, normalized=False,
        implementation="peft_lora",
    ),
    PARALLEL_PROGRAM: AdapterProgramSpec(
        PARALLEL_PROGRAM, branch_count=2, gated=False, normalized=False,
        implementation="parametrized_parallel_low_rank",
    ),
    GATED_PARALLEL_PROGRAM: AdapterProgramSpec(
        GATED_PARALLEL_PROGRAM, branch_count=2, gated=True, normalized=True,
        implementation="parametrized_gated_parallel_low_rank", merge="gated_sum",
        normalization="column_rms",
    ),
}


@lru_cache(maxsize=None)
def get_program(name: str) -> AdapterProgramSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        pass
    if not name.startswith(f"{SYNTH_PREFIX}:"):
        raise ValueError(f"unknown adapter program: {name}")
    fields = name.split(":")
    if len(fields) != 4 or not fields[1].startswith("b"):
        raise ValueError(f"malformed synthesized adapter program: {name}")
    try:
        branch_count = int(fields[1][1:])
        merge = fields[2].removeprefix("merge=")
        normalization = fields[3].removeprefix("norm=")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed synthesized adapter program: {name}") from exc
    canonical = synthesized_program_name(branch_count, merge, normalization)
    if name != canonical or branch_count not in _SYNTH_BRANCHES:
        raise ValueError(f"unsupported synthesized adapter program: {name}")
    if merge not in _SYNTH_MERGES or normalization not in _SYNTH_NORMALIZATIONS:
        raise ValueError(f"unsupported synthesized adapter program: {name}")
    return AdapterProgramSpec(
        name=name, branch_count=branch_count, gated=merge == "gated_sum",
        normalized=normalization != "none",
        implementation="parametrized_synthesized_low_rank",
        merge=merge, normalization=normalization, origin="synthesized",
    )


def validate_program(name: str, rank: int) -> AdapterProgramSpec:
    program = get_program(name)
    if rank < 1:
        raise ValueError("adapter rank must be positive")
    if program.branch_count > 1 and rank % program.branch_count:
        raise ValueError(
            f"rank {rank} cannot be split across {program.branch_count} adapter branches"
        )
    return program


def program_metadata(name: str, rank: int, target_modules: Iterable[str]) -> dict:
    program = validate_program(name, rank)
    return {
        "adapter_program": program.name,
        "adapter_program_version": 2,
        "scale": 2.0,
        "branch_aggregation": "sum_with_unit_initial_gates",
        "adapter_implementation": program.implementation,
        "rank": rank,
        "target_modules": sorted(set(target_modules)),
        "branch_count": program.branch_count,
        "learned_gates": program.gated,
        "normalized_branches": program.normalized,
        "merge": program.merge,
        "normalization": program.normalization,
        "program_origin": program.origin,
        "parameter_budget_rule": (
            "fixed_total_rank_and_target_modules_plus_explicit_structural_overhead"
        ),
    }


def _module_names(model, target_modules: Iterable[str]) -> list[str]:
    targets = set(target_modules)
    names = [
        name for name, child in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in targets
        and hasattr(child, "weight") and getattr(child.weight, "ndim", 0) == 2
    ]
    if not names:
        raise ValueError(f"no target modules found for {sorted(targets)}")
    return names


def install_program(model, name: str, rank: int,
                    target_modules: Iterable[str] = ("q_proj", "v_proj"), *,
                    semantics_version: int = 2) -> list[str]:
    """Install a non-PEFT program using weight parametrizations.

    The base weights remain frozen.  All branches together have the same total
    rank as the fixed LoRA budget, which makes structural comparisons possible.
    """
    import torch
    from torch import nn
    from torch.nn.utils import parametrize

    program = validate_program(name, rank)
    if program.name == STANDARD_PROGRAM:
        raise ValueError("standard_lora_v1 is installed through PEFT")
    if semantics_version not in (1, 2):
        raise ValueError("unknown structured adapter semantics version")
    # Freeze the entire backbone, including non-target layers. Optimizer
    # filtering alone does not prevent their backward graph/grad allocations.
    model.requires_grad_(False)

    class LowRankProgram(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            branch_rank = rank // program.branch_count
            self.branch_rank = branch_rank
            self.semantics_version = semantics_version
            self.scale = 2.0 / rank if semantics_version == 1 else 2.0
            self.A = nn.ParameterList()
            self.B = nn.ParameterList()
            for _ in range(program.branch_count):
                a = nn.Parameter(torch.empty(branch_rank, in_features))
                b = nn.Parameter(torch.zeros(out_features, branch_rank))
                nn.init.kaiming_uniform_(a, a=5**0.5)
                self.A.append(a)
                self.B.append(b)
            if program.gated:
                # Softmax keeps the two branches bounded and starts at an
                # equal mixture.  The gate is a structural primitive, not a
                # change in rank or parameter budget.
                self.gate_logits = nn.Parameter(torch.zeros(program.branch_count))

        def factors(self):
            gates = (
                torch.softmax(self.gate_logits, dim=0)
                if program.gated else
                self.A[0].new_full((program.branch_count,), 1.0 / program.branch_count)
            )
            if self.semantics_version == 2:
                gates = gates * program.branch_count
            normalization = program.normalization
            if self.semantics_version == 1 and name == GATED_PARALLEL_PROGRAM:
                normalization = "none"  # Exact replay of the old named program.
            effective_b = []
            for index, (a, b) in enumerate(zip(self.A, self.B, strict=True)):
                b_eff = b
                if normalization != "none":
                    dim = 0 if normalization == "column_rms" else 1
                    if self.semantics_version == 1:
                        denom = b.norm(dim=dim, keepdim=True).detach().clamp_min(1.0)
                    else:
                        # RMS clipping is the identity below unit RMS. Clamping
                        # before sqrt avoids an undefined gradient at B=0.
                        denom = b.square().mean(dim=dim, keepdim=True).clamp_min(1.0).sqrt()
                    b_eff = b / denom
                effective_b.append(gates[index] * b_eff * self.scale)
            return torch.cat(list(self.A), dim=0), torch.cat(effective_b, dim=1)

        def forward(self, base_weight):
            a, b = self.factors()
            return base_weight + b @ a

    names = _module_names(model, target_modules)
    for name_item in names:
        child = dict(model.named_modules())[name_item]
        for parameter in child.parameters():
            parameter.requires_grad_(False)
        parametrization = LowRankProgram(child.in_features, child.out_features).to(
            device=child.weight.device, dtype=child.weight.dtype,
        )
        parametrize.register_parametrization(
            child, "weight", parametrization,
        )
    return names


def iter_program_modules(model):
    for name, child in model.named_modules():
        if hasattr(child, "parametrizations") and hasattr(child.parametrizations, "weight"):
            parametrization = child.parametrizations.weight[0]
            if hasattr(parametrization, "A") and hasattr(parametrization, "B"):
                yield name, parametrization


def program_parameters(model):
    parameters = []
    for _, parametrization in iter_program_modules(model):
        parameters.extend(parametrization.parameters())
    return parameters


def save_program_adapter(model, adapter_dir: Path, name: str, rank: int,
                         target_modules: Iterable[str]) -> dict:
    from safetensors.torch import save_file

    import torch

    modules = list(iter_program_modules(model))
    if not modules:
        raise ValueError("model has no installed structured adapter program")
    adapter_dir.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for module_name, parametrization in modules:
        for key, value in parametrization.state_dict().items():
            tensors[f"{module_name}.{key}"] = value.detach().cpu().contiguous()
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    config = program_metadata(name, rank, target_modules)
    versions = {p.semantics_version for _, p in modules}
    if len(versions) != 1:
        raise ValueError("mixed adapter semantics")
    config["adapter_program_version"] = versions.pop()
    if config["adapter_program_version"] == 1:
        config["scale"] = 2.0 / rank
        config["branch_aggregation"] = "legacy_mean"
    config["format"] = "jev_rsi_structured_adapter_v1"
    (adapter_dir / "adapter_config.json").write_text(
        __import__("json").dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def load_program_adapter(model, adapter_dir: Path, name: str, rank: int,
                         target_modules: Iterable[str]) -> dict:
    from safetensors.torch import load_file

    import json

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    expected = program_metadata(name, rank, target_modules)
    for key in ("adapter_program", "rank", "target_modules", "branch_count"):
        if config.get(key) != expected[key]:
            raise ValueError(f"structured adapter {key} differs from the requested program")
    install_program(model, name, rank, target_modules,
                    semantics_version=int(config.get("adapter_program_version", 1)))
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    modules = dict(iter_program_modules(model))
    for module_name, parametrization in modules.items():
        state = {}
        prefix = f"{module_name}."
        for key, value in tensors.items():
            if key.startswith(prefix):
                state[key[len(prefix):]] = value.to(next(parametrization.parameters()).device)
        if not state:
            raise ValueError(f"missing structured adapter state for {module_name}")
        parametrization.load_state_dict(state, strict=True)
    return config
