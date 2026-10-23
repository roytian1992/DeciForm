import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def record(i, text):
    return {"id": i, "state": {"text": text}, "question": {
        "type": "choice", "instructions": "Select", "options": ["a", "b"]}, "label": "a"}


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_preparation_excludes_reserved_and_duplicate_text(tmp_path):
    train, reserved = tmp_path / "train.jsonl", tmp_path / "reserved.jsonl"
    write(train, [record("1", "keep\u2028text"), record("2", " KEEP text "), record("3", "reserved")])
    write(reserved, [record("test", "RESERVED")])
    out = tmp_path / "prepared"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"),
        "--source", f"example={train}", "--exclude-jsonl", str(reserved),
        "--base-model", str(tmp_path / "model"), "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    audit = json.loads((out / "audit.json").read_text())
    assert audit["sources"][0]["raw"] == 3
    assert audit["sources"][0]["excluded"] == 1
    assert audit["sources"][0]["duplicates"] == 1
    assert audit["retained_records"] == 1
    plan = json.loads((out / "plan.json").read_text())
    assert plan["sources"][0]["max_prompt_tokens"] is None
    assert Path(plan["parent_checkpoint"]).exists()


def test_evaluation_preparation_detects_overlap(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write(train, [record("train", "SAME text")]); write(test, [record("test", "same text")])
    out = tmp_path / "bad_eval"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_evaluation.py"),
        "--source", f"example={test}", "--training-jsonl", str(train), "--output", str(out)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads((out / "audit.json").read_text())["normalized_text_overlap"] == 1
