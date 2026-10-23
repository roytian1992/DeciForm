"""Materialize full-pool, visible Choice episodes from the local registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "DeciForm" / "src"))
sys.path.insert(0, str(WORKSPACE / "RSI_Research"))

from deciform.data_protocol import build_episode, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path,
                        default=WORKSPACE / "DeciForm/configs/local/data_registry_v3.json")
    parser.add_argument("--data-root", type=Path,
                        default=WORKSPACE / "RSI_Research/data")
    parser.add_argument("--model-dir", type=Path,
                        default=WORKSPACE / "models/Qwen3.5-9B")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", action="append", help="registered source name; default: all ready")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=0,
        help="explicit prompt cap; 0 keeps every model-context-valid row",
    )
    parser.add_argument("--evaluation-target-per-role", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-from", type=Path,
                        help="reuse matching, audited episode directories from an earlier protocol")
    parser.add_argument("--rebuild-source", action="append", default=[],
                        help="source that must be rebuilt even when --reuse-from matches")
    args = parser.parse_args()
    if args.max_prompt_tokens < 0 or args.evaluation_target_per_role < 1:
        raise ValueError("prompt cap must be nonnegative and evaluation budget must be positive")
    max_prompt_tokens = args.max_prompt_tokens or None
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["version"] != 1:
        raise ValueError("unknown data registry version")
    entries = {entry["name"]: entry for entry in registry["ready"]}
    names = args.source or list(entries)
    if len(names) != len(set(names)) or any(name not in entries for name in names):
        raise ValueError("source names must be distinct and registered as ready")
    if len(args.rebuild_source) != len(set(args.rebuild_source)) or any(
            name not in names for name in args.rebuild_source):
        raise ValueError("rebuild sources must be distinct requested sources")
    output_root = args.output_root.resolve()
    reuse_root = args.reuse_from.resolve() if args.reuse_from else None
    reuse_index = None
    if reuse_root is not None:
        reuse_index = json.loads((reuse_root / "manifest.json").read_text(encoding="utf-8"))
        if reuse_index["status"] != "complete":
            raise ValueError("reuse source protocol is incomplete")
    model_dir = args.model_dir.resolve()
    config = {
        "version": 1, "status": "partial", "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "data_root": str(args.data_root.resolve()), "model_dir": str(model_dir),
        "tokenizer_sha256": sha256(model_dir / "tokenizer.json"),
        "seed": args.seed, "max_prompt_tokens": max_prompt_tokens,
        "evaluation_target_per_role": args.evaluation_target_per_role,
        "requested_sources": names, "episodes": {},
        "reuse_from": str(reuse_root) if reuse_root else None,
        "reuse_manifest_sha256": sha256(reuse_root / "manifest.json") if reuse_root else None,
    }
    if args.rebuild_source:
        config["rebuild_sources"] = sorted(args.rebuild_source)
    root_manifest_path = output_root / "manifest.json"
    if args.resume:
        previous = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        for key in config:
            if key not in {"status", "episodes"} and previous[key] != config[key]:
                raise ValueError(f"resume configuration changed: {key}")
        config = previous
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    for name in names:
        if name in config["episodes"]:
            episode_path = output_root / name / "manifest.json"
            if sha256(episode_path) != config["episodes"][name]["manifest_sha256"]:
                raise ValueError(f"completed episode manifest changed: {name}")
            for role, info in json.loads(episode_path.read_text(encoding="utf-8"))["outputs"].items():
                if sha256(output_root / name / f"{role}.jsonl") != info["sha256"]:
                    raise ValueError(f"completed episode data changed: {name}/{role}")
            print(f"verified complete: {name}", flush=True)
            continue
        if (reuse_index is not None and name in reuse_index["episodes"]
                and name not in args.rebuild_source):
            old_path = reuse_root / name
            old_manifest_path = old_path / "manifest.json"
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            if (old_manifest["source"] == entries[name]["source"]
                    and old_manifest["source_kind"] == entries[name]["kind"]
                    and old_manifest["seed"] == args.seed
                    and old_manifest["max_prompt_tokens"] == max_prompt_tokens
                    and old_manifest["evaluation_target_per_role"] == args.evaluation_target_per_role):
                if sha256(old_manifest_path) != reuse_index["episodes"][name]["manifest_sha256"]:
                    raise ValueError(f"reused episode manifest hash changed: {name}")
                for role, info in old_manifest["outputs"].items():
                    if sha256(old_path / f"{role}.jsonl") != info["sha256"]:
                        raise ValueError(f"reused episode role hash changed: {name}/{role}")
                (output_root / name).symlink_to(old_path, target_is_directory=True)
                config["episodes"][name] = {
                    "path": str(output_root / name),
                    "manifest_sha256": sha256(old_manifest_path),
                    "D_train_count": old_manifest["outputs"]["D_train"]["count"],
                    "D_mv_count": old_manifest["outputs"]["D_mv"]["count"],
                    "reused_from": str(old_path),
                }
                root_manifest_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                print(f"reused verified episode: {name}", flush=True)
                continue
        print(f"building: {name}", flush=True)
        episode = build_episode(
            entries[name], args.data_root.resolve(), output_root / name,
            tokenizer, seed=args.seed, max_prompt_tokens=max_prompt_tokens,
            target_per_role=args.evaluation_target_per_role,
        )
        config["episodes"][name] = {
            "path": str(output_root / name),
            "manifest_sha256": sha256(output_root / name / "manifest.json"),
            "D_train_count": episode["outputs"]["D_train"]["count"],
            "D_mv_count": episode["outputs"]["D_mv"]["count"],
        }
        root_manifest_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"complete: {name} train={episode['outputs']['D_train']['count']}", flush=True)
    config["status"] = "complete"
    root_manifest_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": config["status"], "episodes": config["episodes"]}, indent=2))


if __name__ == "__main__":
    main()
