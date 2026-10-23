"""Remove exact sealed-input duplicates from a versioned training projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from deciform.joint_data import atomic_json, file_sha256
from audit_expanded_train import input_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.input_root / "manifest.json").read_text())
    validation = json.loads(args.validation_plan.read_text())
    hashes = {name: set() for name in original["sources"]}
    for split in ("final_hidden", "external_transfer"):
        spec = validation[split]
        path = PROJECT.parent / spec["path"]
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"{split} changed")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["source"] in hashes:
                    hashes[row["source"]].add(input_digest(row))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in original["sources"]:
        source_dir = args.input_root / name
        source_manifest = json.loads((source_dir / "manifest.json").read_text())
        source_path = source_dir / "D_train.jsonl"
        if file_sha256(source_path) != source_manifest["outputs"]["D_train"]["sha256"]:
            raise ValueError(f"{name} input changed")
        destination_dir = output / name
        destination_dir.mkdir()
        destination = destination_dir / "D_train.jsonl"
        count = 0
        removed = 0
        with source_path.open(encoding="utf-8") as incoming, destination.open("x", encoding="utf-8") as outgoing:
            for line in incoming:
                row = json.loads(line)
                if input_digest({**row, "source": name}) in hashes[name]:
                    removed += 1
                    continue
                outgoing.write(line)
                count += 1
        if count + removed != source_manifest["outputs"]["D_train"]["count"]:
            raise ValueError(f"{name} row count changed")
        source_manifest["outputs"]["D_train"] = {
            "path": str(destination), "count": count, "sha256": file_sha256(destination)}
        source_manifest["parent_manifest_sha256"] = file_sha256(source_dir / "manifest.json")
        source_manifest["filtering"]["exact_sealed_input_overlap_removed"] = removed
        atomic_json(destination_dir / "manifest.json", source_manifest)
        reports[name] = {"native_rows": original["sources"][name]["native_rows"],
                         "kept": count, "too_long": original["sources"][name]["too_long"],
                         "source_split": original["sources"][name]["source_split"],
                         "exact_sealed_input_overlap_removed": removed}
        print(json.dumps({"source": name, **reports[name]}), flush=True)
    atomic_json(output / "manifest.json", {
        "version": 2, "status": "complete", "sources": reports,
        "total_rows": sum(report["kept"] for report in reports.values()),
        "removed_total": sum(report["exact_sealed_input_overlap_removed"] for report in reports.values()),
        "parent_manifest_sha256": file_sha256(args.input_root / "manifest.json"),
        "validation_plan_sha256": file_sha256(args.validation_plan),
        "purpose": "Additional training splits with exact sealed-input overlaps removed; MMLU-Pro validation is 70-row training support.",
    })


if __name__ == "__main__":
    main()
