"""Version a locked test inventory using label-independent overlap exclusions."""
from pathlib import Path
import argparse
import datetime
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from deciform.evaluation import load_inventory, text_key
from deciform.joint_data import file_sha256, atomic_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old, audit_path = args.manifest.resolve(), args.audit.resolve()
    audit = json.loads(audit_path.read_text())
    manifest, rows, _ = load_inventory(old)
    if audit["test_manifest_sha256"] != file_sha256(old):
        raise ValueError("audit does not describe this inventory")
    excluded = {rid: "training normalized-text overlap" for rid in audit["affected_test_ids"]}
    seen = {}
    for row in rows:
        if row["id"] in excluded:
            continue
        key = json.dumps({"source_family": row["source_family"], "text": text_key(row),
                          "question": row["question"]}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            if row["label"] != seen[key]["label"]:
                raise ValueError("conflicting duplicate labels")
            excluded[row["id"]] = "duplicate decision input; keep first manifest occurrence"
        else:
            seen[key] = row
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    for stream in manifest["streams"]:
        ids = [r["id"] for r in rows if r["test_stream"] == stream["name"] and r["id"] in excluded]
        stream["raw_count"] = stream["count"]
        stream["count"] -= len(ids)
        stream["exclude_ids"] = ids
        stream["legacy_max_prompt_tokens"] = stream["max_prompt_tokens"]
        stream["max_prompt_tokens"] = None
    manifest.update({"version": 2, "name": output.name,
        "record_count": sum(s["count"] for s in manifest["streams"]),
        "sealed_record_count": sum(s["count"] for s in manifest["streams"] if s["sealed"]),
        "parent_manifest": {"path": str(old), "sha256": file_sha256(old), "record_count": len(rows)},
        "exclusion_audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
        "excluded_records": [{"id": k, "reason": v} for k,v in excluded.items()],
        "inference_policy": {"truncation": False, "max_prompt_tokens": None, "skip_overlength": False},
        "frozen_at": datetime.datetime.now().astimezone().isoformat()})
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "test_plan.json", {"version": 2, "status": "locked",
        "manifest": str(output/"manifest.json"), "manifest_sha256": file_sha256(output/"manifest.json"),
        "record_count": manifest["record_count"],
        "unique_source_family_count": manifest["unique_source_family_count"],
        "streams": manifest["streams"], "evaluation_only": True, "training_allowed": False})
    _, clean, counts = load_inventory(output / "manifest.json")
    print(json.dumps({"record_count": len(clean), "counts": counts, "excluded": excluded,
                      "manifest": str(output/"manifest.json")}, indent=2))

if __name__ == "__main__":
    main()
