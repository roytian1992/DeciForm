"""Audit supplemental train splits against locked test and reserved data.

Writes metadata only; historical training overlap remains a separate audit.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from deciform.evaluation import load_inventory, text_key
from deciform.joint_data import file_sha256

SOURCES = {"banking77": "v3", "sst5": "v3", "trec": "v3", "f2_priority": "v6"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    test_path = WORKSPACE / "RSI_Research/data/typed_adapt_bench/sealed/unified_choice_test_20261022_v2/manifest.json"
    test, test_rows, _ = load_inventory(test_path)
    test_ids = {r["id"] for r in test_rows}
    test_texts = {text_key(r) for r in test_rows}
    reserved_ids, reserved_texts, reserved_files = set(), set(), []
    manifests = {}
    for source, version in SOURCES.items():
        path = WORKSPACE / f"RSI_Research/data/typed_adapt_bench/protocols/full_choice_20261015_{version}/{source}/manifest.json"
        manifest = json.loads(path.read_text())
        manifests[source] = (path, manifest)
        for role in ("D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"):
            spec = manifest["outputs"][role]
            p = Path(spec["path"])
            if file_sha256(p) != spec["sha256"]:
                raise ValueError(f"changed reserved split: {p}")
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            if len(rows) != spec["count"]:
                raise ValueError(f"changed reserved count: {p}")
            reserved_ids.update(r["id"] for r in rows)
            reserved_texts.update(text_key(r) for r in rows)
            reserved_files.append({"source": source, "role": role, **spec})
    sources = []
    for source, (manifest_path, manifest) in manifests.items():
        spec = manifest["outputs"]["D_train"]
        p = Path(spec["path"])
        if file_sha256(p) != spec["sha256"]:
            raise ValueError(f"changed training split: {p}")
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if len(rows) != spec["count"]:
            raise ValueError(f"changed training count: {p}")
        seen_ids, seen_texts, exclusions = set(), set(), []
        for row in rows:
            key = text_key(row)
            reasons = []
            for hit, reason in [(row["id"] in test_ids, "locked_test_id"),
                                (key in test_texts, "locked_test_text"),
                                (row["id"] in reserved_ids, "reserved_role_id"),
                                (key in reserved_texts, "reserved_role_text"),
                                (row["id"] in seen_ids, "duplicate_train_id"),
                                (key in seen_texts, "duplicate_train_text")]:
                if hit:
                    reasons.append(reason)
            if reasons:
                exclusions.append({"id": row["id"], "reasons": reasons})
            seen_ids.add(row["id"])
            seen_texts.add(key)
        sources.append({"source": source, "train": spec,
                        "protocol_manifest": str(manifest_path),
                        "protocol_manifest_sha256": file_sha256(manifest_path),
                        "raw_rows": len(rows), "proposed_exclusion_rows": len(exclusions),
                        "eligible_before_history_audit": len(rows)-len(exclusions),
                        "exclusions": exclusions,
                        "upstream_filtering": manifest.get("filtering"),
                        "legacy_initialization_overlap_flag": manifest.get("initialization_overlap")})
    report = {
        "status": "prepared_not_launched",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": file_sha256(Path(__file__)),
        "test_manifest": str(test_path), "test_manifest_sha256": file_sha256(test_path),
        "test_records": len(test_rows), "sources": sources,
        "raw_candidate_train_rows": sum(s["raw_rows"] for s in sources),
        "eligible_before_history_audit": sum(s["eligible_before_history_audit"] for s in sources),
        "reserved_files": reserved_files, "historical_training_plans": test["training_plans"],
        "historical_training_record_overlap_checked": False,
        "baseline_training_record_overlap_checked": False,
        "normalization": "Unicode casefold plus whitespace collapse of state.text",
        "next_step": "Audit candidate IDs/text against completed training and ancestor checkpoints; exclude repeats from new training, keep test fixed; create continuation config and schedule.",
        "jobs_changed": False, "training_launched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write(chr(10))
    print(json.dumps({k: report[k] for k in ("status", "test_records", "raw_candidate_train_rows", "eligible_before_history_audit", "training_launched")}, ensure_ascii=False))
    for source in sources:
        print(json.dumps({k: source[k] for k in ("source", "raw_rows", "proposed_exclusion_rows", "eligible_before_history_audit")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
