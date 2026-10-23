"""Independently verify generated episode roles and their source manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROLES = ("D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_key(row: dict) -> str:
    text = " ".join(row["state"]["text"].casefold().split())
    question = " ".join(row["question"]["instructions"].casefold().split())
    return hashlib.sha256(f"{text}\n{question}".encode("utf-8")).hexdigest()


def audit_episode(path: Path, expected_hash: str) -> dict:
    manifest_path = path / "manifest.json"
    if sha256(manifest_path) != expected_hash:
        raise ValueError(f"episode manifest hash changed: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["version"] != 2:
        raise ValueError(f"episode incomplete: {path}")
    for pool in manifest["inputs"].values():
        for filename, expected in pool["files"].items():
            if sha256(Path(filename)) != expected:
                raise ValueError(f"source file hash changed: {filename}")
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    verified = {}
    for role in ROLES:
        output = manifest["outputs"][role]
        filename = path / f"{role}.jsonl"
        if sha256(filename) != output["sha256"]:
            raise ValueError(f"episode role hash changed: {filename}")
        labels = Counter()
        with filename.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                option_ids = row["question"].get("option_ids", row["question"]["options"])
                if (row["split"] != role or row["source"] != manifest["source"]
                        or row["question"]["type"] != "choice"
                        or row["label"] not in option_ids
                        or len(option_ids) != manifest["option_count"]):
                    raise ValueError(f"invalid choice record: {filename}:{row['id']}")
                key = input_key(row)
                if row["id"] in seen_ids or key in seen_inputs:
                    raise ValueError(f"cross-role duplicate: {filename}:{row['id']}")
                seen_ids.add(row["id"])
                seen_inputs.add(key)
                labels[row["label"]] += 1
        if sum(labels.values()) != output["count"] or dict(labels) != output["labels"]:
            raise ValueError(f"episode role count changed: {filename}")
        verified[role] = output["count"]
    train_eligible = manifest["filtering"]["train"]["eligible"]
    used_from_train = (sum(verified.values())
                       if manifest["source_kind"] in {"hf", "raw_parquet", "aegis_json"}
                       else verified["D_train"])
    if used_from_train != train_eligible:
        raise ValueError(f"eligible train count not fully allocated: {path}")
    return {"status": "verified", "source": manifest["source"],
            "raw_train": manifest["inputs"]["train"]["count"],
            "eligible_train": train_eligible,
            "filtering": manifest["filtering"], "roles": verified,
            "initialization_overlap": manifest["initialization_overlap"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.protocol_root.resolve()
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if index["status"] != "complete":
        raise ValueError("protocol root is incomplete")
    if set(index["requested_sources"]) != set(index["episodes"]):
        raise ValueError("protocol root is missing requested sources")
    episodes = {
        name: audit_episode(root / name, item["manifest_sha256"])
        for name, item in index["episodes"].items()
    }
    report = {"status": "verified", "protocol_root": str(root),
              "root_manifest_sha256": sha256(root / "manifest.json"),
              "source_count": len(episodes), "episodes": episodes}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_count": len(episodes),
                      "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
