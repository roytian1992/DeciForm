"""Check exact input overlap without exposing sealed labels or scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from deciform.joint_data import atomic_json, file_sha256


def input_digest(row: dict) -> str:
    question = row["question"]
    payload = [row["source"], row["state"]["text"], question["instructions"],
               question.get("option_definitions", question["options"])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validation = json.loads(Path(plan["validation_plan"]).read_text())
    new_sources = [source for source in plan["sources"] if source.get("role") != "replay_prior_D_train"]
    names = {source["name"] for source in new_sources}
    sealed_hashes: dict[str, set[str]] = {name: set() for name in names}
    sealed_counts = {}
    for split in ("final_hidden", "external_transfer"):
        spec = validation[split]
        path = PROJECT.parent / spec["path"]
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"sealed {split} hash changed")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["source"] in names:
                    sealed_hashes[row["source"]].add(input_digest(row))
                count += 1
        if count != spec["record_count"]:
            raise ValueError(f"sealed {split} count changed")
        sealed_counts[split] = count
    reports = {}
    for source in new_sources:
        name = source["name"]
        count = 0
        overlap = 0
        with Path(source["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                overlap += input_digest({**row, "source": name}) in sealed_hashes[name]
                count += 1
        if count != source["count"] or file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"{name}: training source count or hash changed")
        reports[name] = {"train_rows": count, "exact_input_overlap_with_sealed": overlap}
    result = {"status": "pass" if all(not row["exact_input_overlap_with_sealed"] for row in reports.values()) else "fail",
              "plan_sha256": file_sha256(args.plan), "sealed_rows": sealed_counts,
              "note": "Only state, question and options are compared; sealed labels are not used.",
              "sources": reports}
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
