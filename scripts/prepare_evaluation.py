"""Lock local evaluation JSONL and audit overlap against supplied training files."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deciform.evaluation import text_key
from deciform.joint_data import atomic_json, file_sha256
from prepare_data import validate_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="NAME=/path/to/test.jsonl")
    parser.add_argument("--training-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    seen_ids, seen_text, sources, names = set(), set(), [], set()
    for item in args.source:
        name, path = item.split("=", 1); path = Path(path).resolve()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("source names must be distinct simple identifiers")
        names.add(name); count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():continue
                row = json.loads(line); validate_row(row); key = text_key(row)
                if row["id"] in seen_ids or key in seen_text:
                    raise ValueError("duplicate evaluation ID or normalized text")
                seen_ids.add(row["id"]); seen_text.add(key); count += 1
        if not count:raise ValueError(f"empty source: {name}")
        sources.append({"name": name, "path": str(path), "sha256": file_sha256(path),
                        "count": count, "source_families": [name]})
    training, id_hits, text_hits = [], 0, 0
    for path in args.training_jsonl:
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():continue
                row = json.loads(line); count += 1
                id_hits += int(row["id"] in seen_ids)
                text_hits += int(text_key(row) in seen_text)
        training.append({"path": str(path.resolve()), "sha256": file_sha256(path), "records": count})
    manifest = {"status": "locked", "evaluation_only": True, "training_allowed": False,
                "checkpoint_selection_allowed": False, "streams": sources,
                "record_count": len(seen_ids), "source_families": sorted(names),
                "unique_source_family_count": len(names)}
    atomic_json(out / "manifest.json", manifest)
    status = "pass" if id_hits == text_hits == 0 else "fail"
    atomic_json(out / "audit.json", {"status": status, "test_manifest_sha256": file_sha256(out / "manifest.json"),
        "training_inputs": training, "id_overlap": id_hits, "normalized_text_overlap": text_hits,
        "scope": "Only supplied training inputs; no assertion about unlisted training/pretraining data."})
    if status != "pass":raise ValueError("train/test overlap detected; inspect audit.json")
    print(json.dumps({"status": status, "records": len(seen_ids), "sources": len(names)}))


if __name__ == "__main__":
    main()
