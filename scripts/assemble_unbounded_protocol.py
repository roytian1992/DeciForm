"""Assemble the audited no-artificial-cap Choice episodes into one protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from deciform.joint_data import atomic_json, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path, required=True)
    parser.add_argument("--dbpedia-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = {}
    for root in (args.full_root.resolve(), args.additional_root.resolve(), args.dbpedia_root.resolve()):
        root_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if root_manifest.get("status") != "complete":
            raise ValueError(f"incomplete source protocol: {root}")
        for name in root_manifest["episodes"] if "episodes" in root_manifest else root_manifest["sources"]:
            if name in sources:
                raise ValueError(f"duplicate source episode: {name}")
            episode = root / name
            manifest_path = episode / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError(f"incomplete episode: {episode}")
            if manifest.get("max_prompt_tokens") is not None:
                raise ValueError(f"episode still has a prompt cap: {episode}")
            link = output / name
            link.symlink_to(episode, target_is_directory=True)
            sources[name] = {
                "path": str(link.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "D_train_count": manifest["outputs"]["D_train"]["count"],
                "D_mv_count": manifest.get("outputs", {}).get("D_mv", {}).get("count", 0),
                "source": manifest["source"],
            }
    manifest = {
        "version": 1,
        "status": "complete",
        "purpose": "unbounded_model_context_choice_protocol",
        "max_prompt_tokens": None,
        "artificial_prompt_cap": False,
        "source_count": len(sources),
        "episodes": dict(sorted(sources.items())),
        "inputs": {
            "full_root": str(args.full_root.resolve()),
            "additional_root": str(args.additional_root.resolve()),
            "dbpedia_root": str(args.dbpedia_root.resolve()),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({
        "protocol_root": str(output),
        "source_count": len(sources),
        "total_train_rows": sum(item["D_train_count"] for item in sources.values()),
        "sources": {name: item["D_train_count"] for name, item in sorted(sources.items())},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
