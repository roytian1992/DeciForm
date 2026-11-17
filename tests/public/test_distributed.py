import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from safetensors.torch import load_file
from deciform.core.training import train


def test_two_rank_training_matches_one_rank_with_unequal_tasks(tiny_config, rows, tmp_path):
    if tiny_config["backbone"] != "modernbert":
        pytest.skip("Collective normalization is independent of backbone")
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = {
        "device": "cpu",
        "seed": 91,
        "epochs": 1,
        "global_batch": 3,
        "micro_batch": 1,
        "schedule": "constant",
        "save_every": 1,
    }
    train(tiny_config, cfg, data, tmp_path / "single")
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(tiny_config))
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(cfg))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    source = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "deciform.core",
        "train",
        "--model-config",
        str(model_path),
        "--train-config",
        str(train_path),
        "--data",
        str(data),
        "--output",
        str(tmp_path / "multi"),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    a = load_file(str(tmp_path / "single/checkpoint/trainables.safetensors"))
    b = load_file(str(tmp_path / "multi/checkpoint/trainables.safetensors"))
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=2e-6)
