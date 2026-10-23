"""Pin and index ALL P_train sources; never materialize a sampled replacement."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from deciform.joint_data import atomic_json, file_sha256, index_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = args.protocol_root.resolve()
    validation = json.loads(args.validation_plan.read_text())
    root_manifest = json.loads((protocol / "manifest.json").read_text())
    names = sorted(validation["P_train"])
    if set(names) & (set(validation["P_dev"]) | set(validation["P_assess_locked"])):
        raise ValueError("source roles overlap")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "indices").mkdir()
    def work(name):
        episode = protocol / name
        if file_sha256(episode / "manifest.json") != root_manifest["episodes"][name]["manifest_sha256"]:
            raise ValueError(f"collection manifest mismatch: {name}")
        result = index_source(name, episode, output / "indices")
        print(json.dumps({"source": name, "verified_rows": result["count"]}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sources = list(pool.map(work, names))
    plan = {"version": 1, "status": "complete", "scope": "all_P_train_joint",
            "protocol_root": str(protocol),
            "protocol_manifest_sha256": file_sha256(protocol / "manifest.json"),
            "validation_plan": str(args.validation_plan.resolve()),
            "validation_plan_sha256": file_sha256(args.validation_plan),
            "source_count": len(sources), "total_rows": sum(s["count"] for s in sources),
            "sources": sources, "sampling": "global_permutation_without_replacement",
            "training_role": "D_train", "excluded_roles": ["P_dev", "P_assess_locked"]}
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"plan": str(output / "plan.json"), "total_rows": plan["total_rows"]}), flush=True)


if __name__ == "__main__":
    main()
