"""Build a pinned new-source stage with balanced replay of the completed pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from deciform.joint_data import atomic_json, file_sha256, index_source


ADDITIONAL = ("mmlu", "mmlu_pro", "sciq", "qnli", "paws", "emotion", "tweet_offensive")


def replay_quotas(sources: list[dict], target: int) -> dict[str, int]:
    quotas = {source["name"]: 0 for source in sources}
    capacities = {source["name"]: source["count"] for source in sources}
    if target < 0 or target > sum(capacities.values()):
        raise ValueError("replay target exceeds the prior training pool")
    remaining = target
    while remaining:
        active = [name for name in sorted(quotas) if quotas[name] < capacities[name]]
        share = max(1, remaining // len(active))
        for name in active:
            count = min(share, capacities[name] - quotas[name], remaining)
            quotas[name] += count
            remaining -= count
            if not remaining:
                break
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path, required=True)
    parser.add_argument("--boolq-episode", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    old = json.loads(args.prior_plan.read_text())
    validation = json.loads(args.validation_plan.read_text())
    if old["scope"] != "all_P_train_joint" or old["status"] != "complete":
        raise ValueError("prior pool must be a completed full training plan")
    parent = args.parent_checkpoint.resolve()
    parent_progress = json.loads((parent / "progress.json").read_text())
    if parent_progress["status"] != "complete" or parent_progress["plan_sha256"] != file_sha256(args.prior_plan):
        raise ValueError("parent checkpoint is not complete for the prior pool")
    old_names = {source["name"] for source in old["sources"]}
    if old_names | set(ADDITIONAL) | {"boolq"} != set(validation["P_train"]):
        raise ValueError("expanded P_train must contain precisely the prior and eight new sources")
    if set(validation["P_train"]) & (set(validation["P_dev"]) | set(validation["P_assess_locked"])):
        raise ValueError("expanded source roles overlap")
    extra = json.loads((args.additional_root / "manifest.json").read_text())
    if extra["status"] != "complete" or set(extra["sources"]) != set(ADDITIONAL):
        raise ValueError("additional train sources are incomplete")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    index_dir = output / "indices"
    index_dir.mkdir()
    new_sources = [index_source("boolq", args.boolq_episode, index_dir)]
    for name in ADDITIONAL:
        new_sources.append(index_source(name, args.additional_root / name, index_dir))
    added_count = sum(source["count"] for source in new_sources)
    quotas = replay_quotas(old["sources"], added_count)
    replay_sources = []
    for position, source in enumerate(sorted(old["sources"], key=lambda row: row["name"])):
        old_offsets = np.load(source["index"], mmap_mode="r")
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, position]))
        selected = np.sort(rng.choice(source["count"], size=quotas[source["name"]], replace=False))
        index_path = index_dir / f"{source['name']}.npy"
        np.save(index_path, old_offsets[selected])
        replay_sources.append({**source, "count": quotas[source["name"]],
                               "index": str(index_path), "index_sha256": file_sha256(index_path),
                               "role": "replay_prior_D_train", "prior_source_count": source["count"]})
    sources = sorted(new_sources + replay_sources, key=lambda row: row["name"])
    plan = {
        "version": 2, "status": "complete", "scope": "registered_continuation",
        "validation_plan": str(args.validation_plan.resolve()),
        "validation_plan_sha256": file_sha256(args.validation_plan),
        "parent_checkpoint": str(parent), "parent_plan_sha256": file_sha256(args.prior_plan),
        "additional_manifest_sha256": file_sha256(args.additional_root / "manifest.json"),
        "seed": args.seed, "source_count": len(sources), "total_rows": sum(s["count"] for s in sources),
        "new_source_rows": added_count, "replay_rows": sum(quotas.values()),
        "replay_quotas": quotas, "sources": sources,
        "sampling": "one global permutation without replacement over all new rows and balanced old-source replay",
        "training_role": "D_train", "excluded_roles": ["P_dev", "P_assess_locked", "sealed"],
    }
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"plan": str(output / "plan.json"), "total_rows": plan["total_rows"],
                      "new_source_rows": added_count, "replay_rows": plan["replay_rows"],
                      "replay_quotas": quotas}), flush=True)


if __name__ == "__main__":
    main()
