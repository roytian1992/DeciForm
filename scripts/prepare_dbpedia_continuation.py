"""Prepare a full DBpedia14 continuation stage with replay from prior sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from deciform.joint_data import atomic_json, file_sha256, index_source


def replay_quotas(sources: list[dict], target: int) -> dict[str, int]:
    quotas = {source["name"]: 0 for source in sources}
    capacities = {source["name"]: int(source["count"]) for source in sources}
    if target < 0 or target > sum(capacities.values()):
        raise ValueError("replay target exceeds the prior training pool")
    remaining = target
    while remaining:
        active = [name for name in sorted(quotas) if quotas[name] < capacities[name]]
        share = max(1, remaining // len(active))
        for name in active:
            amount = min(share, capacities[name] - quotas[name], remaining)
            quotas[name] += amount
            remaining -= amount
            if not remaining:
                break
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-plan", type=Path, required=True)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--completed-plan", type=Path, required=True)
    parser.add_argument("--completed-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--dbpedia-episode", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()

    initial = json.loads(args.initial_plan.read_text())
    completed = json.loads(args.completed_plan.read_text())
    initial_report = json.loads(args.initial_report.read_text())
    completed_report = json.loads(args.completed_report.read_text())
    validation = json.loads(args.validation_plan.read_text())
    parent = args.parent_checkpoint.resolve()
    parent_progress = json.loads((parent / "progress.json").read_text())
    parent_manifest = json.loads((parent.parent.parent / "run_manifest.json").read_text())

    if initial["status"] != "complete" or initial_report["status"] != "complete":
        raise ValueError("the initial Static-Full stage must be complete")
    if completed["status"] != "complete" or completed_report["status"] != "complete":
        raise ValueError("the 18-source Static-Full stage must be complete")
    if parent_progress["status"] != "complete":
        raise ValueError("continuation parent checkpoint is incomplete")
    completed_sha = file_sha256(args.completed_plan)
    if parent_progress["plan_sha256"] != completed_sha or parent_manifest["plan_sha256"] != completed_sha:
        raise ValueError("parent checkpoint does not belong to the completed 18-source plan")
    if parent_manifest["config"]["joint_plan"] != str(args.completed_plan.resolve()).replace(
            str(WORKSPACE) + "/", ""):
        raise ValueError("parent run manifest points to a different completed plan")

    initial_sources = {source["name"]: source for source in initial["sources"]}
    completed_sources = {source["name"]: source for source in completed["sources"]}
    if set(initial_sources) - set(completed_sources):
        raise ValueError("expanded training plan is missing an initial source")
    new_names = set(completed_sources) - set(initial_sources)
    if len(initial_sources) != 10 or len(new_names) != 8 or len(completed_sources) != 18:
        raise ValueError("expected the completed ten-source stage plus eight added sources")

    initial_seen = initial_report["source_rows_seen"]
    for name, source in initial_sources.items():
        if initial_seen.get(name) != source["count"]:
            raise ValueError(f"initial Static-Full did not visit every planned row for {name}")
    expanded_seen = completed_report["source_rows_seen"]
    for name in new_names:
        source = completed_sources[name]
        if expanded_seen.get(name) != source["count"]:
            raise ValueError(f"expanded Static-Full did not visit every row for new source {name}")

    episode_dir = args.dbpedia_episode.resolve().parent
    episode_manifest = json.loads(args.dbpedia_episode.read_text())
    dbpedia_count = int(episode_manifest["outputs"]["D_train"]["count"])
    if episode_manifest["source"] != "fancyzhx/dbpedia_14" or dbpedia_count != 559477:
        raise ValueError("DBpedia14 D_train identity or count differs from the audited 559,477 rows")
    if set(validation["P_train"]) != set(completed_sources) | {"dbpedia14"}:
        raise ValueError("validation plan must add DBpedia14 to all existing P_train sources")
    if set(validation["P_train"]) & (set(validation["P_dev"]) | set(validation["P_assess_locked"])):
        raise ValueError("training and held-out source roles overlap")

    prior_sources = [initial_sources[name] for name in sorted(initial_sources)]
    prior_sources.extend(completed_sources[name] for name in sorted(new_names))
    quotas = replay_quotas(prior_sources, dbpedia_count)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    index_dir = output / "indices"
    index_dir.mkdir()

    dbpedia = index_source("dbpedia14", episode_dir, index_dir)
    if dbpedia["count"] != dbpedia_count:
        raise ValueError("indexed DBpedia14 row count differs from its manifest")

    replay_sources = []
    for position, source in enumerate(prior_sources):
        if file_sha256(Path(source["index"])) != source["index_sha256"]:
            raise ValueError(f"prior row index changed: {source['name']}")
        offsets = np.load(source["index"], mmap_mode="r")
        if len(offsets) != source["count"]:
            raise ValueError(f"prior row index count mismatch: {source['name']}")
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, position]))
        selected = np.sort(rng.choice(source["count"], size=quotas[source["name"]], replace=False))
        index_path = index_dir / f"{source['name']}.npy"
        np.save(index_path, offsets[selected])
        replay_sources.append({
            **source,
            "count": quotas[source["name"]],
            "index": str(index_path.resolve()),
            "index_sha256": file_sha256(index_path),
            "role": "static_full_replay",
            "prior_source_count": source["count"],
        })

    sources = sorted([dbpedia, *replay_sources], key=lambda row: row["name"])
    total_rows = sum(source["count"] for source in sources)
    plan = {
        "version": 1,
        "status": "complete",
        "scope": "registered_continuation",
        "validation_plan": str(args.validation_plan.resolve()),
        "validation_plan_sha256": file_sha256(args.validation_plan),
        "parent_checkpoint": str(parent),
        "parent_plan_sha256": completed_sha,
        "parent_report_sha256": file_sha256(args.completed_report),
        "initial_plan_sha256": file_sha256(args.initial_plan),
        "initial_report_sha256": file_sha256(args.initial_report),
        "dbpedia_episode_manifest": str(args.dbpedia_episode.resolve()),
        "dbpedia_episode_manifest_sha256": file_sha256(args.dbpedia_episode),
        "seed": args.seed,
        "source_count": len(sources),
        "total_rows": total_rows,
        "dbpedia_new_rows": dbpedia_count,
        "replay_rows": sum(quotas.values()),
        "replay_quotas": quotas,
        "sources": sources,
        "sampling": "All DBpedia14 D_train rows plus same-sized, source-balanced replay sampled without replacement from the unique rows seen by the completed 18-source Static-Full model.",
        "training_role": "D_train",
        "excluded_roles": ["P_dev", "P_assess_locked", "final_hidden", "external_transfer"],
    }
    if total_rows != 2 * dbpedia_count or plan["replay_rows"] != dbpedia_count:
        raise ValueError("DBpedia continuation must contain equal new-data and replay budgets")
    atomic_json(output / "plan.json", plan)
    print(json.dumps({
        "plan": str(output / "plan.json"),
        "source_count": len(sources),
        "dbpedia_new_rows": dbpedia_count,
        "replay_rows": plan["replay_rows"],
        "total_rows": total_rows,
        "replay_quotas": quotas,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
