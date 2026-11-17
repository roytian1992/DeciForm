import copy
import json
import torch
from deciform.core.model import DecisionModel
from deciform.core.checkpoint import save_export, load_export
from deciform.core.metrics import permutation_audit


def test_all_tasks_backward_roundtrip_and_permutation(tiny_config, rows, tmp_path):
    model = DecisionModel(tiny_config)
    losses = model.losses(rows)
    assert len(losses) == 3 and torch.isfinite(losses).all()
    losses.sum().backward()
    assert any(
        p.grad is not None and p.grad.norm() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n
    )
    assert any(
        p.grad is not None and p.grad.norm() > 0
        for n, p in model.named_parameters()
        if "numeric_score.readout" in n
    )
    opt = torch.optim.AdamW(model.optimizer_groups())
    opt.step()
    opt.zero_grad()
    expected = model.predict(rows)
    permuted = copy.deepcopy(rows)
    for k in ("options", "option_ids"):
        permuted[0]["question"][k].reverse()
    permuted[1]["candidate_values"] = [True, False]
    actual = model.predict(permuted)
    audit = permutation_audit(expected, actual)
    assert all(
        v["flips"] == 0 and v["max_probability_difference"] == 0 for v in audit["tasks"].values()
    )
    save_export(model, tmp_path / "export")
    restored = load_export(tmp_path / "export")
    assert restored.inventory() == model.inventory()
    output = restored.predict(rows)
    for a, b in zip(expected, output):
        torch.testing.assert_close(
            torch.tensor(a["probabilities"]), torch.tensor(b["probabilities"]), rtol=0, atol=0
        )
    # Labels and dataset/sample IDs cannot change the actual model encoding.
    changed = copy.deepcopy(rows[0])
    changed.update(id="new", label="s", source="different")
    assert model.encode(changed) == model.encode(rows[0])
    model.eval()
    z = model([rows[0]])[0]
    changed = copy.deepcopy(rows[0])
    changed["question"]["options"][1] = "a wholly different lengthy candidate"
    z2 = model([changed])[0]
    torch.testing.assert_close(z[[0, 2]], z2[[0, 2]], atol=1e-6, rtol=1e-5)
