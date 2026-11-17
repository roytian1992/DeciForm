import json
import torch
from safetensors.torch import load_file
from deciform.core.training import train
import deciform.core.training as training


def test_exact_resume_matches_uninterrupted(tiny_config, rows, tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = {
        "device": "cpu",
        "seed": 91,
        "epochs": 2,
        "global_batch": 3,
        "micro_batch": 1,
        "schedule": "constant",
        "save_every": 1,
    }
    full = tmp_path / "full"
    paused = tmp_path / "paused"
    train(tiny_config, cfg, data, full)
    write = training.atomic_json

    def pause_after_step(path, value):
        write(path, value)
        if path == paused / "progress.json" and value["step"] == 1:
            (paused / "STOP").touch()

    monkeypatch.setattr(training, "atomic_json", pause_after_step)
    train(tiny_config, cfg, data, paused)
    assert json.loads((paused / "status.json").read_text())["phase"] == "paused"
    (paused / "STOP").unlink()
    monkeypatch.setattr(training, "atomic_json", write)
    train(tiny_config, cfg, data, paused, resume=True)
    a = load_file(str(full / "checkpoint/trainables.safetensors"))
    b = load_file(str(paused / "checkpoint/trainables.safetensors"))
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    assert json.loads((paused / "status.json").read_text())["sample_visits"] == 6
