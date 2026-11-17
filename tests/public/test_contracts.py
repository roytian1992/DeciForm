import copy
import json
import pytest
import torch
from deciform.core.schemas import normalize, support
from deciform.core.data import training_view, prepare, verify_release, fingerprint
from deciform.core.objectives import proper_loss, interpolation, data_loss, SpecialistObjective
from deciform.core.metrics import evaluate
from deciform.core.sources import convert


@pytest.mark.parametrize("k", [2, 4, 21, 77])
def test_proper_uniform_gradient(k):
    z = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(proper_loss(z, 0), z)[0]
    expected = torch.ones_like(z) / k
    expected[0] -= 1
    torch.testing.assert_close(g, expected, atol=1e-12, rtol=1e-12)


def test_numeric_cdf_formula_and_interpolation():
    values = [0.0, 1.0, 4.0, 10.0]
    y = 2.0
    z = torch.tensor([1.0, 2.0, -0.1, 0.2], dtype=torch.float64, requires_grad=True)
    q = z.new_tensor(interpolation(values, y))
    assert torch.dot(q, z.new_tensor(values)).item() == y
    pred = z.softmax(0).cumsum(0)[:-1]
    target = q.cumsum(0)[:-1]
    widths = z.new_tensor([1.0, 3.0, 6.0]) / 10
    direct = (
        -(widths * (target * pred.log() + (1 - target) * torch.log1p(-pred))).sum()
        * 3.0383850790577127
    )
    torch.testing.assert_close(data_loss(z, values, y, SpecialistObjective("soft_cdf")), direct)


def test_union_is_deterministic_label_preserving_and_unweighted(rows):
    row = copy.deepcopy(rows[0])
    row["question"]["options"] = [f"c{i}" for i in range(77)]
    row["question"]["option_ids"] = list(range(77))
    row["label"] = 31
    a = training_view(row, 3, 17, 2)
    b = training_view(row, 3, 17, 2)
    assert a == b and len(support(a)) == 7 and 31 in support(a) and "weight" not in a
    flipped = copy.deepcopy(row)
    for k in ("options", "option_ids"):
        flipped["question"][k].reverse()
    assert set(support(a)) == set(support(training_view(flipped, 3, 17, 2)))
    assert training_view(rows[2], 3, 17, 2) == normalize(rows[2], True)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_prepare_rejects_leakage_and_eval_sampling(rows, tmp_path):
    a = write(tmp_path / "a.jsonl", rows)
    b = write(tmp_path / "b.jsonl", rows)
    with pytest.raises(ValueError, match="crosses splits"):
        prepare({"train": a, "test": b}, tmp_path / "leak")
    b = write(b, [training_view(rows[0])])
    with pytest.raises(ValueError, match="sampled candidates"):
        prepare({"test": b}, tmp_path / "sampled")
    manifest = prepare({"train": a}, tmp_path / "ok")
    assert manifest["splits"]["train"]["rows"] == 3
    verify_release(tmp_path / "ok/manifest.json")
    with open(tmp_path / "ok/train.jsonl", "a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="Checksum"):
        verify_release(tmp_path / "ok/manifest.json")


def test_conversion_no_bool_string_inference(tmp_path):
    recipe = {
        "source": "example",
        "split": "train",
        "task": "noul",
        "label_field": "label",
        "text_template": "{text}",
        "instructions_template": "Is this true?",
        "label_mapping": {'"yes"': True, '"no"': False},
    }
    raw = write(
        tmp_path / "raw.jsonl", [{"text": "A", "label": "yes"}, {"text": "B", "label": "no"}]
    )
    out = tmp_path / "out.jsonl"
    result = convert(recipe, raw, out)
    assert result["rows"] == 2
    assert [json.loads(l)["label"] for l in out.read_text().splitlines()] == [True, False]


def test_ranking_metrics_full_support(rows):
    r = copy.deepcopy(rows[0])
    r["target_distribution"] = [0.5, 0, 0.5]
    p = {
        "id": "c",
        "task": "choice",
        "candidate_values": ["a", "s", "b"],
        "probabilities": [0.5, 0.1, 0.4],
        "prediction": "a",
    }
    m = evaluate([r], [p])["tasks"]["choice"]
    assert m["auc"] == 1 and m["mrr"] == 0.75 and m["ndcg@5"] == 1
    p["candidate_values"] = ["a", "s"]
    with pytest.raises(ValueError):
        evaluate([r], [p])


def test_fingerprint_ignores_label_and_normalizes_permutation(rows):
    a = normalize(rows[0])
    b = copy.deepcopy(a)
    for k in ("options", "option_ids"):
        b["question"][k].reverse()
    b.update(label="b", id="other", source="another")
    assert fingerprint(a) == fingerprint(b)
    a = normalize(rows[1])
    b = copy.deepcopy(a)
    b["candidate_values"].reverse()
    assert fingerprint(a) == fingerprint(b)


def test_score_without_numeric_label_and_repeated_sampling_rejected(rows):
    bad = copy.deepcopy(rows[2])
    bad.pop("label")
    bad["target_distribution"] = [1 / 11] * 11
    with pytest.raises(ValueError, match="raw numerical label"):
        normalize(bad, True)
    sampled = training_view(rows[0])
    with pytest.raises(ValueError, match="existing view"):
        training_view(sampled)
