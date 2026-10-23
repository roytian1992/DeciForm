import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from deciform.choice import ChoiceHead, choice_response, last_token_pool, render_choice_prompts
from deciform.sampling import candidate_subset, choose_candidates

ROOT = Path(__file__).resolve().parents[1]

def row():
    return {"id": "one", "state": {"text": "A payment question"}, "question": {
        "type": "choice", "instructions": "Choose", "options": ["a", "b", "c"],
        "option_ids": ["x", "y", "z"], "option_definitions": ["alpha", "beta", "gamma"]}, "label": "y"}

def test_prompt_permutation_preserves_candidate_identity():
    sample = row(); ids, prompts = render_choice_prompts(sample)
    reverse_ids, reverse = render_choice_prompts(sample, [2, 0, 1])
    assert dict(zip(ids, prompts)) == dict(zip(reverse_ids, reverse))
    p = choice_response(ids, torch.tensor([1., 3., 2.]))
    q = choice_response(reverse_ids, torch.tensor([2., 1., 3.]))
    assert p["prediction"] == q["prediction"] == "y"
    assert p["probabilities"] == pytest.approx(q["probabilities"])

def test_candidate_sampling_preserves_gold_and_alignment():
    sample = row(); chosen = choose_candidates(sample, 42, 1)
    subset = candidate_subset(sample, chosen)
    assert sample["label"] in subset["question"]["option_ids"]
    assert chosen == choose_candidates(sample, 42, 1)
    for key in ["options", "option_ids", "option_definitions"]:
        assert subset["question"][key] == [sample["question"][key][i] for i in chosen]

def test_last_token_pool_ignores_padding():
    hidden = torch.arange(24.).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert torch.equal(last_token_pool(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 2]]))

def test_release_tensors_and_config_are_intact():
    cp = ROOT / "checkpoints/DeciForm-9B-v0.1"
    hashes = json.loads((cp / "hashes.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((cp / name).read_bytes()).hexdigest() == digest
    release = json.loads((cp / "release.json").read_text())
    assert all(hashes[name] == digest for name, digest in release["published_weight_hashes"].items())
    state = torch.load(cp / "choice_head.pt", map_location="cpu", weights_only=True)
    head = ChoiceHead(state["hidden_size"]); head.load_state_dict(state["state_dict"])
    assert torch.isfinite(head(torch.zeros(2, state["hidden_size"]))).all()
    assert state["temperature"] == release["temperature"]

def test_trainer_imports_without_sibling_projects():
    spec = importlib.util.spec_from_file_location("deciform_train", ROOT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.STANDARD_PROGRAM == "standard_lora_v1"
    assert all(not name.startswith("../") for name in module.code_manifest())
