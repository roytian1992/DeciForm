"""Build a training protocol containing only rows absent from an older protocol.

The delta is defined by the stable per-source row id.  This makes a continuation
stage auditable: a row is either present in the completed parent pool or it is
eligible for this stage, and the resulting files are independently hashed and
indexed for the joint trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from deciform.joint_data import atomic_json, file_sha256, index_source


def _read_json(path: Path):
    return json.loads(path.read_text())


def _row_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_number}: missing stable string id")
            ids.add(row_id)
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-plan", type=Path, required=True)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-plan-dir", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    args = parser.parse_args()

    new_plan_path = args.new_plan.resolve()
    old_plan_path = args.old_plan.resolve()
    output_root = args.output_root.resolve()
    output_plan_dir = args.output_plan_dir.resolve()
    validation_path = args.validation_plan.resolve()
    if output_root.exists() or output_plan_dir.exists():
        raise FileExistsError("delta output already exists; use a new versioned path")
    new_plan = _read_json(new_plan_path)
    old_plan = _read_json(old_plan_path)
    output_root.mkdir(parents=True)
    output_plan_dir.mkdir(parents=True)
    (output_plan_dir / "indices").mkdir()

    old_sources = {entry["name"]: entry for entry in old_plan["sources"]}
    old_ids: dict[str, set[str]] = {}
    for source, entry in old_sources.items():
        print(json.dumps({"event": "index_old_ids", "source": source}), flush=True)
        old_ids[source] = _row_ids(Path(entry["path"]))
    parent_row_count = sum(entry["count"] for entry in old_plan["sources"])
    parent_unique_id_count = sum(len(old_ids[source]) for source in old_sources)
    parent_duplicate_rows = parent_row_count - parent_unique_id_count

    source_summaries: list[dict] = []
    total_new = 0
    overlap = 0
    old_only = 0
    zero_delta: list[str] = []
    for entry in new_plan["sources"]:
        source = entry["name"]
        input_path = Path(entry["path"])
        output_episode = output_root / source
        output_episode.mkdir()
        old_for_source = old_ids.get(source, set())
        kept = 0
        seen_new: set[str] = set()
        digest = hashlib.sha256()
        output_path = output_episode / "D_train.jsonl"
        with input_path.open() as source_handle, output_path.open("w") as output_handle:
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"{input_path}:{line_number}: missing stable string id")
                if row_id in seen_new:
                    raise ValueError(f"{source}: duplicate id in new plan: {row_id}")
                seen_new.add(row_id)
                total_new += 1
                if row_id in old_for_source:
                    overlap += 1
                    continue
                output_handle.write(line)
                digest.update(line.encode() if isinstance(line, str) else line)
                kept += 1
        if kept == 0:
            shutil.rmtree(output_episode)
            zero_delta.append(source)
            old_only += len(old_for_source - seen_new)
            continue
        input_manifest = _read_json(Path(entry["episode_manifest"]))
        source_manifest = {
            "version": 1,
            "status": "complete",
            "source": source,
            "purpose": "delta_choice_continuation",
            "parent_source_manifest": str(Path(entry["episode_manifest"]).resolve()),
            "parent_source_manifest_sha256": file_sha256(Path(entry["episode_manifest"])),
            "max_prompt_tokens": None,
            "option_count": input_manifest["option_count"],
            "outputs": {
                "D_train": {"path": str(output_path.resolve()), "count": kept,
                             "sha256": digest.hexdigest(), "role": "D_train"}
            },
            "excluded_parent_rows": len(old_for_source & seen_new),
            "parent_source_rows": old_sources.get(source, {}).get("count", 0),
            "parent_source_unique_ids": len(old_for_source),
            "parent_source_duplicate_rows": old_sources.get(source, {}).get("count", 0) - len(old_for_source),
            "new_source_rows": len(seen_new),
        }
        atomic_json(output_episode / "manifest.json", source_manifest)
        source_summaries.append({
            "name": source,
            "input_path": str(input_path.resolve()),
            "input_sha256": file_sha256(input_path),
            "new_source_rows": len(seen_new),
            "excluded_parent_rows": len(old_for_source & seen_new),
            "count": kept,
        })
        old_only += len(old_for_source - seen_new)
        print(json.dumps({"event": "delta_source", "source": source, "count": kept}), flush=True)

    for source, ids in old_ids.items():
        new_entry = next((entry for entry in new_plan["sources"] if entry["name"] == source), None)
        if new_entry is None:
            old_only += len(ids)
    kept_total = sum(item["count"] for item in source_summaries)
    if kept_total != total_new - overlap:
        raise AssertionError("delta accounting mismatch")
    # The expected count is part of the audit contract for this continuation.
    if kept_total != 915_569:
        raise AssertionError(f"unexpected delta count: {kept_total} (expected 915569)")

    root_manifest = {
        "version": 1,
        "status": "complete",
        "purpose": "delta_choice_continuation",
        "parent_plan": str(old_plan_path),
        "parent_plan_sha256": file_sha256(old_plan_path),
        "new_plan": str(new_plan_path),
        "new_plan_sha256": file_sha256(new_plan_path),
        "max_prompt_tokens": None,
        "artificial_prompt_cap": False,
        "source_count": len(source_summaries),
        "total_new_rows": total_new,
        "parent_row_count": parent_row_count,
        "parent_unique_id_count": parent_unique_id_count,
        "parent_duplicate_rows": parent_duplicate_rows,
        "parent_overlap_rows": overlap,
        "parent_only_unique_ids": old_only,
        "delta_rows": kept_total,
        "zero_delta_sources": sorted(zero_delta),
        "episodes": {item["name"]: {"path": str((output_root / item["name"]).resolve()),
                                      "manifest_sha256": file_sha256(output_root / item["name"] / "manifest.json"),
                                      "D_train_count": item["count"]}
                     for item in source_summaries},
    }
    atomic_json(output_root / "manifest.json", root_manifest)

    validation = _read_json(validation_path)
    validation["version"] = int(validation.get("version", 1)) + 1
    validation["name"] = "choice_delta_continuation_20261022_v1"
    validation["purpose"] = "Evaluate a continuation trained only on rows absent from the completed Static-Full parent."
    validation["P_train"] = sorted(item["name"] for item in source_summaries)
    validation["P_dev"] = []
    validation["P_assess_locked"] = []
    validation["delta_parent_plan_sha256"] = file_sha256(old_plan_path)
    validation["delta_rows"] = kept_total
    atomic_json(validation_path, validation)

    def index_one(item: dict) -> dict:
        return index_source(item["name"], output_root / item["name"], output_plan_dir / "indices")

    with ThreadPoolExecutor(max_workers=4) as pool:
        indexed = list(pool.map(index_one, source_summaries))
    plan = {
        "version": 1,
        "status": "complete",
        "scope": "registered_continuation",
        "protocol_root": str(output_root),
        "protocol_manifest_sha256": file_sha256(output_root / "manifest.json"),
        "validation_plan": str(validation_path),
        "validation_plan_sha256": file_sha256(validation_path),
        "source_count": len(indexed),
        "total_rows": sum(item["count"] for item in indexed),
        "sources": indexed,
        "sampling": "global_permutation_without_replacement",
        "training_role": "D_train",
        "excluded_roles": ["P_dev", "P_assess_locked"],
        "parent_checkpoint": str((WORKSPACE / "JEV_RSI/runs/joint_static_full_4gpu_v6_seed20261015/checkpoints/step_00038022").resolve()),
        "parent_plan_sha256": file_sha256(old_plan_path),
        "parent_run_manifest_sha256": file_sha256(WORKSPACE / "JEV_RSI/runs/joint_static_full_4gpu_v6_seed20261015/run_manifest.json"),
        "delta_definition": "new row id absent from parent source's D_train id set",
    }
    atomic_json(output_plan_dir / "plan.json", plan)
    print(json.dumps({"event": "complete", "delta_rows": kept_total,
                      "sources": len(indexed), "plan": str(output_plan_dir / "plan.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
