from pathlib import Path

import torch
from torch import nn
import pytest
from deciform.adapter_transfer import initialize_adapter
from deciform.adapter_program import STANDARD_PROGRAM, iter_program_modules

from deciform.adapter_program import (
    ADAPTER_PROGRAMS,
    GATED_PARALLEL_PROGRAM,
    PARALLEL_PROGRAM,
    synthesized_program_name,
    install_program,
    load_program_adapter,
    program_parameters,
    save_program_adapter,
    validate_program,
)


class ToyDecisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 5)
        self.v_proj = nn.Linear(5, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.v_proj(torch.tanh(self.q_proj(inputs)))


def test_structural_programs_keep_total_rank_fixed() -> None:
    assert validate_program(PARALLEL_PROGRAM, 16).branch_count == 2
    with torch.no_grad():
        model = ToyDecisionBlock()
    install_program(model, GATED_PARALLEL_PROGRAM, 16, ("q_proj", "v_proj"))
    parameters = program_parameters(model)
    # Two rank-8 branches have the same total A/B element count as one rank-16
    # branch for each target module; gates are recorded separately as overhead.
    assert sum(parameter.numel() for parameter in parameters) == 2 * (
        (4 * 8 + 5 * 8) + (5 * 8 + 3 * 8)
    ) + 4


def test_synthesized_program_genome_is_canonical_and_fixed_budget() -> None:
    name = synthesized_program_name(4, "gated_sum", "row_rms")
    assert name in ADAPTER_PROGRAMS
    spec = validate_program(name, 16)
    assert spec.origin == "synthesized"
    assert spec.branch_count == 4
    assert spec.gated is True
    assert spec.normalization == "row_rms"
    with torch.no_grad():
        model = ToyDecisionBlock()
    install_program(model, name, 16, ("q_proj", "v_proj"))
    # Four rank-4 branches have exactly the rank-16 A/B budget. The four gate
    # scalars are explicit overhead and are charged in the run report.
    assert sum(parameter.numel() for parameter in program_parameters(model)) == 4 * (
        (4 * 4 + 5 * 4) + (5 * 4 + 3 * 4)
    ) + 8


def test_structured_adapter_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    first = ToyDecisionBlock()
    install_program(first, GATED_PARALLEL_PROGRAM, 4, ("q_proj", "v_proj"))
    with torch.no_grad():
        for parameter in program_parameters(first):
            parameter.add_(torch.randn_like(parameter) * 0.01)
    inputs = torch.randn(3, 4)
    expected = first(inputs).detach()
    save_program_adapter(first, tmp_path / "adapter", GATED_PARALLEL_PROGRAM, 4,
                         ("q_proj", "v_proj"))

    second = ToyDecisionBlock()
    # Match the frozen base weights before loading the adapter state.
    second.load_state_dict({
        "q_proj.weight": first.q_proj.parametrizations.weight.original.detach(),
        "q_proj.bias": first.q_proj.bias.detach(),
        "v_proj.weight": first.v_proj.parametrizations.weight.original.detach(),
        "v_proj.bias": first.v_proj.bias.detach(),
    })
    load_program_adapter(second, tmp_path / "adapter", GATED_PARALLEL_PROGRAM, 4,
                         ("q_proj", "v_proj"))
    assert torch.allclose(expected, second(inputs).detach(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("source,target", [
    (STANDARD_PROGRAM, synthesized_program_name(4, "gated_sum", "row_rms")),
    (synthesized_program_name(4, "gated_sum", "row_rms"), STANDARD_PROGRAM),
    (synthesized_program_name(2, "sum", "column_rms"),
     synthesized_program_name(4, "gated_sum", "row_rms")),
])
def test_structure_change_preserves_trained_predictions(tmp_path, source, target):
    torch.manual_seed(31)
    base = ToyDecisionBlock()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    adapter, _ = initialize_adapter(base, source, 8, ["q_proj", "v_proj"])
    with torch.no_grad():
        for parameter in adapter.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.4)
    inputs = torch.randn(9, 4)
    expected = adapter(inputs).detach()
    path = tmp_path / "adapter"
    if source == STANDARD_PROGRAM:
        adapter.save_pretrained(path)
    else:
        save_program_adapter(adapter, path, source, 8, ["q_proj", "v_proj"])
    new_base = ToyDecisionBlock()
    new_base.load_state_dict(base_state)
    converted, audit = initialize_adapter(new_base, target, 8, ["q_proj", "v_proj"], path)
    assert audit["mode"] == "exact_factor_conversion"
    assert torch.allclose(expected, converted(inputs), atol=3e-6, rtol=3e-5)


def test_freezes_non_target_backbone_and_has_finite_zero_init_gradients():
    model = ToyDecisionBlock()
    model.other = nn.Linear(3, 3)
    install_program(model, synthesized_program_name(2, "gated_sum", "column_rms"), 4)
    model(torch.randn(2, 4)).square().mean().backward()
    assert model.other.weight.requires_grad is False
    assert model.q_proj.bias.requires_grad is False
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in program_parameters(model))


def test_v2_parallel_scale_matches_standard_and_legacy_load_is_preserved(tmp_path):
    model = ToyDecisionBlock()
    install_program(model, PARALLEL_PROGRAM, 4, semantics_version=1)
    assert {p.scale for _, p in iter_program_modules(model)} == {0.5}
    path = tmp_path / "legacy"
    save_program_adapter(model, path, PARALLEL_PROGRAM, 4, ["q_proj", "v_proj"])
    second = ToyDecisionBlock()
    load_program_adapter(second, path, PARALLEL_PROGRAM, 4, ["q_proj", "v_proj"])
    assert {p.semantics_version for _, p in iter_program_modules(second)} == {1}
    third = ToyDecisionBlock()
    install_program(third, PARALLEL_PROGRAM, 4)
    assert {p.scale for _, p in iter_program_modules(third)} == {2.0}


def test_linear_reparameterization_matches_one_adam_step(tmp_path):
    torch.manual_seed(14)
    base = ToyDecisionBlock()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    first, _ = initialize_adapter(base, STANDARD_PROGRAM, 4, ["q_proj", "v_proj"])
    with torch.no_grad():
        for p in first.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .01)
    first.save_pretrained(tmp_path / "parent")
    second = ToyDecisionBlock()
    second.load_state_dict(base_state)
    second, _ = initialize_adapter(second, PARALLEL_PROGRAM, 4, ["q_proj", "v_proj"], tmp_path / "parent")
    inputs, labels = torch.randn(5, 4), torch.randn(5, 3)
    for model in (first, second):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0)
        (model(inputs) - labels).square().mean().backward()
        optimizer.step()
    assert torch.allclose(first(inputs), second(inputs), atol=2e-6, rtol=2e-5)
