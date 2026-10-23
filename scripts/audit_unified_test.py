"""Verify locked test IDs/text against every parent and delta training record."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from deciform.joint_data import atomic_json, file_sha256
from deciform.evaluation import load_inventory, text_key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    manifest, rows, counts = load_inventory(args.manifest)
    ids = {r["id"] for r in rows}
    texts = {text_key(r) for r in rows}
    entries, overlap_ids, overlap_texts = [], set(), set()
    for stage, reference in manifest["training_plans"].items():
        plan_path = Path(reference["path"])
        if file_sha256(plan_path) != reference["sha256"]:
            raise ValueError(f"training plan changed: {stage}")
        plan = json.loads(plan_path.read_text())
        for source in plan["sources"]:
            path = Path(source["path"])
            digest = hashlib.sha256()
            count = id_hits = text_hits = 0
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    if row["id"] in ids:
                        id_hits += 1
                        overlap_ids.add(row["id"])
                    key = text_key(row)
                    if key in texts:
                        text_hits += 1
                        overlap_texts.add(key)
            if count != source["count"] or digest.hexdigest() != source["sha256"]:
                raise ValueError(f"training content changed: {stage}/{source['name']}")
            entry = {"stage": stage, "source": source["name"], "training_rows": count,
                     "id_overlap_rows": id_hits, "normalized_text_overlap_rows": text_hits,
                     "path": str(path), "sha256": digest.hexdigest()}
            entries.append(entry)
            print(json.dumps(entry), flush=True)
    affected = [r["id"] for r in rows if r["id"] in overlap_ids or text_key(r) in overlap_texts]
    report = {"status": "pass" if not affected else "overlap_found",
              "test_manifest": str(args.manifest.resolve()),
              "test_manifest_sha256": file_sha256(args.manifest),
              "test_records": len(rows), "source_counts": counts,
              "training_rows_scanned": sum(e["training_rows"] for e in entries),
              "unique_overlapping_test_ids": len(overlap_ids),
              "unique_overlapping_test_texts": len(overlap_texts),
              "affected_test_ids": affected, "sources": entries,
              "text_normalization": "Unicode casefold plus whitespace collapse; state.text only",
              "test_internal_text_duplicate_count": len(rows)-len(texts),
              "elapsed_seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({k:report[k] for k in ("status","test_records","training_rows_scanned","unique_overlapping_test_ids","unique_overlapping_test_texts")}), flush=True)
    if affected:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
