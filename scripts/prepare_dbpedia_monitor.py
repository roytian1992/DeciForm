"""Add DBpedia14 official-test development panels and seal the remainder."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path[:0] = [str(PROJECT / "src"), str(WORKSPACE / "RSI_Research")]

from datasets import load_from_disk
from deciform.data_protocol import raw_choice, text_sha256
from deciform.development_validation import select_panels
from deciform.joint_data import atomic_json, file_sha256


def add_input(inputs: dict[Path, tuple[str, int, str]], path: Path,
              digest: str, count: int, role: str) -> None:
    path = path.resolve()
    value = (digest, int(count), role)
    if path in inputs:
        previous = inputs[path]
        if previous[0] != digest:
            raise ValueError(f"conflicting content hashes for {path}")
        inputs[path] = (digest, max(previous[1], value[1]),
                        f"{previous[2]}|{role}" if role not in previous[2].split("|") else previous[2])
    else:
        inputs[path] = value


def collect_exclusion_inputs(plan_paths: list[Path], episode_manifest: Path,
                             existing_development: Path) -> dict[Path, tuple[str, int, str]]:
    inputs: dict[Path, tuple[str, int, str]] = {}
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text())
        for source in plan["sources"]:
            add_input(inputs, Path(source["path"]), source["sha256"], source["count"],
                      f"{plan_path.stem}:D_train")
            manifest_path = Path(source["episode_manifest"])
            if file_sha256(manifest_path) != source["episode_manifest_sha256"]:
                raise ValueError(f"episode manifest changed: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            for role, output in manifest["outputs"].items():
                if role not in {"D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"}:
                    continue
                path = Path(output.get("path", manifest_path.parent / f"{role}.jsonl"))
                add_input(inputs, path, output["sha256"], output["count"],
                          f"{source['name']}:{role}")

    dbpedia_manifest = json.loads(episode_manifest.read_text())
    if dbpedia_manifest["source"] != "fancyzhx/dbpedia_14":
        raise ValueError("the supplied episode manifest is not DBpedia14")
    for role, output in dbpedia_manifest["outputs"].items():
        if role in {"D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"}:
            add_input(inputs, Path(output["path"]), output["sha256"], output["count"],
                      f"dbpedia14:{role}")

    development = json.loads(existing_development.read_text())
    if development.get("status") != "complete" or development.get("purpose") != "development_monitoring_only":
        raise ValueError("existing development manifest is not a completed monitoring release")
    for source in development["sources"]:
        add_input(inputs, Path(source["path"]), source["sha256"], source["count"],
                  f"existing_development:{source['name']}")
    return inputs


def scan_exclusions(inputs: dict[Path, tuple[str, int, str]],
                    candidates: set[str]) -> tuple[set[str], list[dict]]:
    blocked: set[str] = set()
    audit = []
    for path, (expected_hash, expected_count, role) in inputs.items():
        observed = hashlib.sha256()
        count = matches = 0
        with path.open("rb") as handle:
            for line in handle:
                observed.update(line)
                if not line.strip():
                    continue
                count += 1
                key = text_sha256(json.loads(line))
                if key in candidates:
                    blocked.add(key)
                    matches += 1
        if count != expected_count or observed.hexdigest() != expected_hash:
            raise ValueError(f"exclusion input changed: {path}")
        audit.append({"path": str(path), "sha256": expected_hash, "count": count,
                      "role": role, "candidate_text_matches": matches})
        print(json.dumps({"event": "exclusion_scan", "role": role, "count": count,
                          "candidate_matches": matches}), flush=True)
    return blocked, audit


def write_jsonl(path: Path, rows: list[dict], split: str) -> dict:
    digest = hashlib.sha256()
    labels = Counter()
    with path.open("wb") as handle:
        for row in rows:
            row["split"] = split
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            labels[row["label"]] += 1
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "count": len(rows), "label_counts": dict(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-plan", type=Path, action="append", required=True,
                        help="Training plan whose rows and episode roles must be excluded")
    parser.add_argument("--dbpedia-episode-manifest", type=Path, required=True)
    parser.add_argument("--existing-development-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=WORKSPACE / "RSI_Research/data")
    parser.add_argument("--model-dir", type=Path,
                        default=WORKSPACE / "models/Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()
    output, sealed = args.output_dir.resolve(), args.sealed_dir.resolve()
    if output.exists() or sealed.exists():
        raise FileExistsError("output and sealed directories must be new")
    output.mkdir(parents=True)
    sealed.mkdir(parents=True)

    root = args.data_root.resolve()
    source_manifest_path = root / "kev_upstream/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_info = next(item for item in source_manifest["datasets"]
                       if item["repo"] == "fancyzhx/dbpedia_14")
    revision = source_info["revision"]
    split_path = Path(source_info["output_path"]) / "test"
    raw_files = sorted(split_path.glob("*.arrow"))
    if not raw_files:
        raise FileNotFoundError(f"no DBpedia14 test Arrow files in {split_path}")
    raw_hashes = {str(path.resolve()): file_sha256(path) for path in raw_files}
    dataset = load_from_disk(str(split_path))
    if len(dataset) != source_info["splits"]["test"] or len(dataset) != 70000:
        raise ValueError(f"unexpected DBpedia14 test count: {len(dataset)}")
    labels = list(dataset.features["label"].names)
    episode_path = args.dbpedia_episode_manifest.resolve()
    expected_labels = json.loads(episode_path.read_text())["source_info"]["labels"]
    if labels != expected_labels:
        raise ValueError("DBpedia14 label order differs from the existing typed-choice protocol")

    candidates = []
    for index, native in enumerate(dataset):
        row = raw_choice(native, source="fancyzhx/dbpedia_14", name="dbpedia14",
                         index=index, labels=labels, revision=revision)
        if row is None:
            continue
        row.update(id=f"dbpedia14/test/{index:07d}", source_split="official_test",
                   split="unassigned")
        candidates.append(row)
    candidate_hashes = {text_sha256(row) for row in candidates}
    if len(candidate_hashes) != len(candidates):
        raise ValueError("duplicate DBpedia14 test texts before exclusion filtering")

    inputs = collect_exclusion_inputs(args.joint_plan, episode_path,
                                      args.existing_development_manifest)
    blocked, exclusion_audit = scan_exclusions(inputs, candidate_hashes)
    eligible = []
    rejected = Counter()
    seen = set()
    for row in candidates:
        key = text_sha256(row)
        if not row["state"]["text"]:
            rejected["empty"] += 1
        elif key in blocked:
            rejected["overlap_training_or_existing_roles"] += 1
        elif key in seen:
            rejected["duplicate_text"] += 1
        else:
            seen.add(key)
            eligible.append(row)
    print(json.dumps({"event": "eligibility", "raw_test_count": len(dataset),
                      "candidate_count": len(candidates), "eligible_count": len(eligible),
                      "rejected": dict(rejected)}), flush=True)

    panels = select_panels(eligible, seed=args.seed, target=args.target)
    selected_ids = {row["id"] for rows in panels.values() for row in rows}
    hidden = [row for row in eligible if row["id"] not in selected_ids]
    if not panels["natural"] or not panels["balanced"] or not hidden:
        raise ValueError("DBpedia14 split produced an empty development panel or hidden set")
    selected_keys = [text_sha256(row) for rows in panels.values() for row in rows]
    if len(selected_keys) != len(set(selected_keys)) or set(selected_keys) & blocked:
        raise ValueError("development panel overlap check failed")

    dev_sources = []
    for panel, rows in panels.items():
        path = output / f"dbpedia14_{panel}.jsonl"
        details = write_jsonl(path, rows, f"expanded_development_{panel}")
        dev_sources.append({"name": f"dbpedia14__{panel}", "dataset": "dbpedia14",
                            "panel": panel, **details, "max_prompt_tokens": None})
    hidden_path = sealed / "final_hidden.jsonl"
    hidden_details = write_jsonl(hidden_path, hidden, "sealed_final_hidden")

    episode = json.loads(episode_path.read_text())
    train_output = episode["outputs"]["D_train"]
    eval_plan = {
        "version": 1,
        "status": "complete",
        "purpose": "evaluation_only_dbpedia14_development_panels",
        "training_disabled": True,
        "sources": [{"name": "dbpedia14", "path": train_output["path"],
                     "count": train_output["count"], "sha256": train_output["sha256"],
                     "episode_manifest": str(episode_path),
                     "episode_manifest_sha256": file_sha256(episode_path),
                     "max_prompt_tokens": None,
                     "option_count": episode["option_count"]}],
    }
    eval_plan_path = output / "evaluation_plan.json"
    atomic_json(eval_plan_path, eval_plan)
    dbpedia_role_inputs = {}
    for role, item in episode["outputs"].items():
        if role in {"D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"}:
            dbpedia_role_inputs[role] = {"path": item["path"], "sha256": item["sha256"],
                                         "count": item["count"]}
    monitor_manifest = {
        "version": 1,
        "status": "complete",
        "purpose": "development_monitoring_only",
        "dataset_family_count": 1,
        "seed": args.seed,
        "target_per_source_per_panel": args.target,
        "source": "fancyzhx/dbpedia_14",
        "source_revision": revision,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_split": "official test",
        "source_split_count": len(dataset),
        "raw_files": raw_hashes,
        "evaluation_plan": str(eval_plan_path.resolve()),
        "joint_plan_sha256": file_sha256(args.joint_plan[0]),
        "evaluation_plan_sha256": file_sha256(eval_plan_path),
        "dbpedia_training_episode_roles": dbpedia_role_inputs,
        "sources": dev_sources,
        "record_count": sum(item["count"] for item in dev_sources),
        "unique_text_count": len(selected_keys),
        "cross_panel_text_overlap": 0,
        "eligible_count": len(eligible),
        "exclusions": dict(rejected),
        "exclusion_audit": exclusion_audit,
        "sealed_examples_evaluated": False,
        "sealed_labels_loaded_by_deterministic_split_builder": True,
        "sealed_rows_used_for_model_or_method_selection": False,
        "sealed_final_hidden_manifest": str((sealed / "manifest.json").resolve()),
        "selection": {
            "natural": "deterministic sample from eligible official-test rows without label conditioning",
            "balanced": "equal per-label quota from eligible rows disjoint from natural",
            "final_hidden": "all remaining eligible official-test rows after development selection",
        },
        "token_limit": None,
        "code": {str(path.resolve()): file_sha256(path) for path in (
            Path(__file__), PROJECT / "src/deciform/development_validation.py",
            PROJECT / "src/deciform/data_protocol.py")},
        "tokenizer_config_sha256": file_sha256(args.model_dir / "tokenizer_config.json"),
    }
    atomic_json(output / "manifest.json", monitor_manifest)
    atomic_json(sealed / "manifest.json", {
        "version": 1,
        "status": "sealed",
        "purpose": "final_source_holdout_never_used_for_development_or_checkpoint_selection",
        "source": "fancyzhx/dbpedia_14",
        "source_revision": revision,
        "native_split": "official test",
        "raw_split_count": len(dataset),
        "eligible_after_exclusion_filter": len(eligible),
        "final_hidden": hidden_details,
        "label_counts": dict(Counter(row["label"] for row in hidden)),
        "disjoint_from_development_ids": True,
        "disjoint_from_training_and_existing_visible_text": True,
        "do_not_load_until_method_and_checkpoints_are_frozen": True,
        "labels_loaded_by_deterministic_split_builder": True,
        "model_evaluation_performed": False,
        "development_manifest_sha256": file_sha256(output / "manifest.json"),
    })
    print(json.dumps({"status": "complete", "development_manifest": str(output / "manifest.json"),
                      "development_counts": {key: len(value) for key, value in panels.items()},
                      "sealed_final_hidden_count": len(hidden),
                      "sealed_manifest": str(sealed / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
