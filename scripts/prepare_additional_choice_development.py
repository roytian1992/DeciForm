"""Build visible development panels for the seven newly trained sources."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path[:0] = [str(PROJECT / "src"), str(WORKSPACE / "RSI_Research"), str(PROJECT / "scripts")]

from deciform.data_protocol import text_sha256
from deciform.development_validation import sample_key
from deciform.joint_data import atomic_json, file_sha256
from deciform.choice import render_choice_prompts
from prepare_sealed_choice_evaluations import SOURCES, normalize


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def scan_training(plan_path: Path, include, candidates: set[str], blocked: set[str]) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    reports = []
    seen_paths = set()
    for source in plan["sources"]:
        if not include(source):
            continue
        path = Path(source["path"])
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if file_sha256(path) != source["sha256"]:
            raise ValueError(f"training input changed: {path}")
        rows = matches = 0
        for row in read_jsonl(path):
            rows += 1
            key = text_sha256(row)
            if key in candidates:
                blocked.add(key)
                matches += 1
        reports.append({"path": str(path), "sha256": source["sha256"],
                        "rows_scanned": rows, "candidate_matches": matches})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--expanded-plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--target-per-source", type=int, default=1000)
    args = parser.parse_args()
    if args.target_per_source < 1:
        raise ValueError("--target-per-source must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sealed_manifest = json.loads(args.sealed_manifest.read_text())
    sealed_indices = {source: set() for source in SOURCES}
    sealed_text = set()
    sealed_reports = []
    for role in ("final_hidden", "external_transfer"):
        spec = sealed_manifest["outputs"][role]
        path = Path(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"sealed input changed: {path}")
        count = 0
        for row in read_jsonl(path):
            count += 1
            sealed_indices[row["source"]].add(row["provenance"]["raw_row_index"])
            sealed_text.add(text_sha256(row))
        if count != spec["count"]:
            raise ValueError(f"sealed count changed: {path}")
        sealed_reports.append({"role": role, "path": str(path), "sha256": spec["sha256"],
                               "count": count})

    from datasets import load_from_disk

    pools = {}
    origins = {}
    all_candidate_hashes = set()
    for source in SOURCES:
        dataset_path = args.holdout_root / source
        dataset = load_from_disk(str(dataset_path))
        rows = []
        for index, native in enumerate(dataset):
            if index in sealed_indices[source]:
                continue
            row, _ = normalize(source, native, index, dataset)
            row["source_split"] = "pinned_test_visible_remainder"
            row["split"] = "additional_development_natural"
            rows.append(row)
            all_candidate_hashes.add(text_sha256(row))
        pools[source] = rows
        arrow_files = sorted(dataset_path.glob("*.arrow"))
        origins[source] = {
            "native_count": len(dataset),
            "sealed_raw_indices_excluded": len(sealed_indices[source]),
            "candidate_count": len(rows),
            "raw_files": {str(path): file_sha256(path) for path in arrow_files},
        }

    blocked = set(sealed_text)
    scans = []
    scans.extend(scan_training(args.parent_plan, lambda source: True, all_candidate_hashes, blocked))
    scans.extend(scan_training(args.expanded_plan,
                               lambda source: source.get("role") != "replay_prior_D_train",
                               all_candidate_hashes, blocked))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sources = []
    selected_text = set()
    for source in SOURCES:
        reasons = Counter()
        eligible = []
        source_seen = set()
        for row in sorted(pools[source], key=lambda item: sample_key(item, args.seed)):
            key = text_sha256(row)
            if not row["state"]["text"]:
                reasons["empty"] += 1
            elif key in blocked:
                reasons["overlap_training_or_sealed"] += 1
            elif key in selected_text or key in source_seen:
                reasons["duplicate_visible_text"] += 1
            else:
                source_seen.add(key)
                eligible.append(row)
        valid = []
        for start in range(0, len(eligible), 64):
            batch = eligible[start:start + 64]
            prompts = [prompt for row in batch for prompt in render_choice_prompts(row)[1]]
            lengths = [len(ids) for ids in tokenizer(
                prompts, truncation=False, add_special_tokens=True
            )["input_ids"]]
            cursor = 0
            for row in batch:
                size = len(row["question"]["options"])
                if max(lengths[cursor:cursor + size]) <= 1024:
                    valid.append(row)
                else:
                    reasons["over_1024_tokens"] += 1
                cursor += size
        records = valid[:args.target_per_source]
        if not records:
            raise ValueError(f"no eligible development rows for {source}")
        path = output / f"{source}_natural.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                        encoding="utf-8")
        for row in records:
            selected_text.add(text_sha256(row))
        sources.append({
            "name": f"{source}__new_source_natural",
            "dataset": source,
            "panel": "new_source_natural",
            "path": str(path),
            "sha256": file_sha256(path),
            "count": len(records),
            "max_prompt_tokens": 1024,
            "label_counts": dict(Counter(row["label"] for row in records)),
        })
        origins[source].update({
            "eligible_after_exclusion_and_length": len(valid),
            "selected_count": len(records),
            "exclusions": dict(reasons),
        })
        print(json.dumps({"source": source, "selected": len(records),
                          "eligible": len(valid), "exclusions": dict(reasons)}), flush=True)

    selected_count = sum(source["count"] for source in sources)
    if selected_text & blocked or len(selected_text) != selected_count:
        raise ValueError("materialized development data is duplicated or overlaps training/sealed inputs")
    manifest = {
        "version": 1,
        "status": "complete",
        "purpose": "development_monitoring_only",
        "seed": args.seed,
        "target_per_source": args.target_per_source,
        "sources": sources,
        "source_count": len(sources),
        "record_count": selected_count,
        "unique_text_count": len(selected_text),
        "cross_panel_text_overlap": 0,
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": file_sha256(args.parent_plan),
        "joint_plan": str(args.expanded_plan.resolve()),
        "joint_plan_sha256": file_sha256(args.expanded_plan),
        "sealed_manifest": str(args.sealed_manifest.resolve()),
        "sealed_manifest_sha256": file_sha256(args.sealed_manifest),
        "sealed_inputs_excluded": sealed_reports,
        "sealed_labels_or_scores_used_for_selection": False,
        "training_exclusion_audit": scans,
        "origins": origins,
        "selection": "stable SHA256 order by row ID after excluding all sealed raw indices, exact training/sealed text overlaps, duplicates, and prompts over 1024 tokens",
        "temperature_policy": "inherited checkpoint temperature; monitoring only, no fitting",
        "code": {str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve())},
        "tokenizer_config_sha256": file_sha256(args.model_dir / "tokenizer_config.json"),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "output": str(output),
                      "record_count": manifest["record_count"]}), flush=True)


if __name__ == "__main__":
    main()
