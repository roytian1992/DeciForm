"""Validate JSONL, exclude declared reserved/history rows, and build a training plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deciform.joint_data import atomic_json, file_sha256, index_source


def text_key(row):
    return hashlib.sha256(" ".join(row["state"]["text"].casefold().split()).encode()).hexdigest()


def validate_row(row):
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("each row needs a nonempty string ID")
    q = row["question"]
    if q["type"] != "choice" or not isinstance(q["instructions"], str):
        raise ValueError("expected typed choice instructions")
    if not isinstance(row["state"]["text"], str) or not row["state"]["text"].strip():
        raise ValueError("state.text must be a nonempty string")
    options = q["options"]; ids = q.get("option_ids", options)
    definitions = q.get("option_definitions", options)
    if len(options) < 2 or len(ids) != len(options) or len(definitions) != len(options):
        raise ValueError("choice arrays must align")
    if any(not isinstance(x, str) for x in options + ids + definitions):
        raise ValueError("choice arrays must contain strings")
    if len(set(ids)) != len(ids) or row["label"] not in ids:
        raise ValueError("unique IDs and a valid gold label are required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="NAME=/path/to/train.jsonl")
    parser.add_argument("--exclude-jsonl", action="append", type=Path, default=[],
                        help="reserved evaluation or previously trained JSONL; repeat as needed")
    parser.add_argument("--parent", type=Path, default=ROOT / "checkpoints/DeciForm-9B-v0.1")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / "indices").mkdir()
    db = sqlite3.connect(out / "dedup.sqlite")
    db.execute("CREATE TABLE excluded (kind TEXT, value TEXT, PRIMARY KEY(kind,value))")
    db.execute("CREATE TABLE retained (kind TEXT, value TEXT, PRIMARY KEY(kind,value))")
    exclusions = []
    for path in args.exclude_jsonl:
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():continue
                row = json.loads(line); count += 1
                db.executemany("INSERT OR IGNORE INTO excluded VALUES (?,?)", [("id", row["id"]), ("text", text_key(row))])
        db.commit()
        exclusions.append({"path": str(path.resolve()), "sha256": file_sha256(path), "records": count})
    sources, audits, names = [], [], set()
    for item in args.source:
        name, path = item.split("=", 1); path = Path(path).resolve()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("source names must be distinct simple identifiers")
        names.add(name); dest = out / "protocol" / name; dest.mkdir(parents=True)
        counts = Counter(); sizes = set()
        with path.open() as handle, (dest / "D_train.jsonl").open("x") as output:
            for line in handle:
                if not line.strip():continue
                row = json.loads(line); validate_row(row); counts["raw"] += 1
                keys = [("id", row["id"]), ("text", text_key(row))]
                if any(db.execute("SELECT 1 FROM excluded WHERE kind=? AND value=?", k).fetchone() for k in keys):
                    counts["excluded"] += 1; continue
                if any(db.execute("SELECT 1 FROM retained WHERE kind=? AND value=?", k).fetchone() for k in keys):
                    counts["duplicates"] += 1; continue
                db.executemany("INSERT INTO retained VALUES (?,?)", keys)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["retained"] += 1; sizes.add(len(row["question"]["options"]))
        db.commit()
        if not counts["retained"]:raise ValueError(f"no training rows retained for {name}")
        atomic_json(dest / "manifest.json", {"outputs": {"D_train": {"count": counts["retained"],
            "sha256": file_sha256(dest / "D_train.jsonl")}}, "max_prompt_tokens": None,
            "option_count": next(iter(sizes)) if len(sizes) == 1 else None})
        sources.append(index_source(name, dest, out / "indices"))
        audits.append({"name": name, "input": str(path), "input_sha256": file_sha256(path), **counts})
    db.close()
    parent = args.parent.resolve()
    parent_progress = json.loads((parent / "progress.json").read_text())
    if parent_progress["status"] != "complete":raise ValueError("parent checkpoint is not complete")
    validation = out / "validation_plan.json"
    atomic_json(validation, {"P_train": sorted(names), "excluded_inputs": exclusions})
    plan = {"scope": "registered_continuation", "status": "complete", "sources": sources,
            "total_rows": sum(s["count"] for s in sources), "validation_plan": str(validation),
            "validation_plan_sha256": file_sha256(validation), "parent_checkpoint": str(parent),
            "parent_plan_sha256": parent_progress["plan_sha256"]}
    atomic_json(out / "plan.json", plan)
    config = json.loads((ROOT / "configs/train_lora.json").read_text())
    config.update(joint_plan=str(out / "plan.json"), expected_train_rows=plan["total_rows"],
                  model_dir=str(args.base_model.resolve()), initial_checkpoint=str(parent))
    atomic_json(out / "training_config.json", config)
    atomic_json(out / "audit.json", {"sources": audits, "exclusions": exclusions,
        "history_and_evaluation_exclusion_scope": "only the JSONL files explicitly supplied",
        "normalization": "Unicode casefold and whitespace collapse of state.text",
        "retained_records": plan["total_rows"], "prompt_truncation": False})
    print(json.dumps({"plan": str(out / "plan.json"), "records": plan["total_rows"], "sources": audits}))


if __name__ == "__main__":
    main()
