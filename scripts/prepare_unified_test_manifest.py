"""Freeze one unified test inventory for all local and external baselines.

The inventory has one record pool and no natural/balanced panels. It combines
unseen-source assessments, the official DBpedia14 test holdout, and the two
already sealed cross-source holdouts. Sealed files are referenced in place and
are never copied into a development directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from deciform.joint_data import atomic_json, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"test manifest already exists: {output}")

    protocol_root = WORKSPACE / "RSI_Research/data/typed_adapt_bench/protocols/full_choice_20261015_v6"
    streams = [
        {
            "name": "banking77_unseen_source",
            "source_families": ["banking77"],
            "path": protocol_root / "banking77/D_assess_visible.jsonl",
            "role": "unseen_source",
            "max_prompt_tokens": 1024,
            "training_source_family": False,
        },
        {
            "name": "f2_priority_unseen_source",
            "source_families": ["f2_priority"],
            "path": protocol_root / "f2_priority/D_assess_visible.jsonl",
            "role": "unseen_source",
            "max_prompt_tokens": 1024,
            "training_source_family": False,
        },
        {
            "name": "sst5_unseen_source",
            "source_families": ["sst5"],
            "path": protocol_root / "sst5/D_assess_visible.jsonl",
            "role": "unseen_source",
            "max_prompt_tokens": 1024,
            "training_source_family": False,
        },
        {
            "name": "trec_unseen_source",
            "source_families": ["trec"],
            "path": protocol_root / "trec/D_assess_visible.jsonl",
            "role": "unseen_source",
            "max_prompt_tokens": 1024,
            "training_source_family": False,
        },
        {
            "name": "dbpedia14_official_test",
            "source_families": ["dbpedia14"],
            "path": WORKSPACE / "RSI_Research/data/typed_adapt_bench/sealed/dbpedia14_final_hidden_20261021_unbounded_v1/final_hidden.jsonl",
            "role": "official_test_holdout",
            "max_prompt_tokens": None,
            "training_source_family": True,
            "sealed": True,
        },
        {
            "name": "sealed_choice_final_hidden",
            "source_families": ["mmlu", "mmlu_pro", "sciq", "qnli", "paws", "emotion", "tweet_offensive"],
            "path": WORKSPACE / "RSI_Research/data/typed_adapt_bench/sealed/choice_sealed_20261016_v1/final_hidden.jsonl",
            "role": "sealed_in_source_holdout",
            "max_prompt_tokens": 1024,
            "training_source_family": True,
            "sealed": True,
        },
        {
            "name": "sealed_choice_external_transfer",
            "source_families": ["mmlu", "mmlu_pro", "sciq", "qnli", "paws", "emotion", "tweet_offensive"],
            "path": WORKSPACE / "RSI_Research/data/typed_adapt_bench/sealed/choice_sealed_20261016_v1/external_transfer.jsonl",
            "role": "sealed_transfer_holdout",
            "max_prompt_tokens": 1024,
            "training_source_family": True,
            "sealed": True,
        },
    ]

    seen_ids: set[str] = set()
    source_families: set[str] = set()
    total = 0
    sealed_count = 0
    entries = []
    for stream in streams:
        path = stream["path"].resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        local_ids: set[str] = set()
        local_families: set[str] = set()
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"{path}:{line_number}: missing id")
                if row_id in local_ids:
                    raise ValueError(f"{stream['name']}: duplicate id {row_id}")
                if row_id in seen_ids:
                    raise ValueError(f"cross-stream duplicate id {row_id}")
                local_ids.add(row_id)
                seen_ids.add(row_id)
                family = row.get("source")
                if isinstance(family, str):
                    local_families.add(family)
                count += 1
        source_families.update(stream["source_families"])
        total += count
        if stream.get("sealed"):
            sealed_count += count
        entries.append({
            "name": stream["name"],
            "path": str(path),
            "sha256": file_sha256(path),
            "count": count,
            "source_families": sorted(stream["source_families"]),
            "role": stream["role"],
            "max_prompt_tokens": stream["max_prompt_tokens"],
            "training_source_family": stream["training_source_family"],
            "sealed": bool(stream.get("sealed", False)),
        })

    output.mkdir(parents=True)
    parent_plan_path = WORKSPACE / "JEV_RSI/runs/joint_pool_20261016_v1/plan.json"
    delta_plan_path = WORKSPACE / "JEV_RSI/runs/delta_choice_pool_20261022_v1/plan.json"
    parent_plan = json.loads(parent_plan_path.read_text())
    delta_plan = json.loads(delta_plan_path.read_text())
    training_source_families = sorted({entry["name"] for entry in parent_plan["sources"]}
                                      | {entry["name"] for entry in delta_plan["sources"]})
    manifest = {
        "version": 1,
        "status": "locked",
        "name": "unified_choice_test_20261022_v1",
        "purpose": "One common test inventory for our model, Open-Jev and Kev; no panel split.",
        "evaluation_only": True,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "record_count": total,
        "stream_count": len(entries),
        "unique_source_family_count": len(source_families),
        "sealed_record_count": sealed_count,
        "unseen_source_record_count": sum(e["count"] for e in entries if e["role"] == "unseen_source"),
        "official_dbpedia_test_record_count": sum(e["count"] for e in entries if e["name"] == "dbpedia14_official_test"),
        "source_families": sorted(source_families),
        "streams": entries,
        "selection_policy": {
            "unseen_sources": "P_dev assessment records from Banking77, F2 priority, SST-5 and TREC; none occurs in the 19-source training plan",
            "dbpedia14": "official test split remainder after training/development exclusion",
            "sealed_sources": "pre-existing sealed final-hidden and external-transfer records; referenced without copying",
            "reporting": "report per source family and one aggregate; do not report natural/balanced panels",
        },
        "training_plans": {
            "parent": {"path": str(parent_plan_path.resolve()), "sha256": file_sha256(parent_plan_path)},
            "delta": {"path": str(delta_plan_path.resolve()), "sha256": file_sha256(delta_plan_path)},
        },
        "training_source_families": training_source_families,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "test_plan.json", {"version": 1, "status": "locked", "streams": entries,
        "manifest": str((output / "manifest.json").resolve()),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "record_count": total, "unique_source_family_count": len(source_families),
        "evaluation_only": True, "training_allowed": False})
    (output / "README.md").write_text(
        "# Unified Choice Test Set\n\n"
        "This is one locked test inventory. It has no natural/balanced panels and must be used identically for our model, Open-Jev, and Kev.\n\n"
        f"Records: {total}; unique source families: {len(source_families)}; sealed records: {sealed_count}.\n\n"
        "The manifest references sealed files in place. Do not copy or load sealed streams during training or checkpoint selection.\n"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), "record_count": total,
                      "unique_source_families": len(source_families), "sealed_records": sealed_count}, indent=2))


if __name__ == "__main__":
    main()
