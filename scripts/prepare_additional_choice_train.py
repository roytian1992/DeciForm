"""Convert cached native train splits into auditable Choice-only training rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path[:0] = [str(PROJECT / "src"), str(WORKSPACE / "RSI_Research")]

from datasets import Dataset
from transformers import AutoTokenizer

from deciform.joint_data import atomic_json, file_sha256
from prepare_sealed_choice_evaluations import normalize
from deciform.choice import render_choice_prompts


SPLITS = {
    "mmlu": ("cais___mmlu", "mmlu-auxiliary_train.arrow", "auxiliary_train"),
    "mmlu_pro": ("TIGER-Lab___mmlu-pro", "mmlu-pro-validation.arrow", "validation"),
    "sciq": ("allenai___sciq", "sciq-train.arrow", "train"),
    "qnli": ("nyu-mll___glue", "glue-train.arrow", "train"),
    "paws": ("google-research-datasets___paws", "paws-train.arrow", "train"),
    "emotion": ("dair-ai___emotion", "emotion-train.arrow", "train"),
    "tweet_offensive": ("cardiffnlp___tweet_eval", "tweet_eval-train.arrow", "train"),
}


def cached_split(cache: Path, source: str) -> Path:
    directory, filename, _ = SPLITS[source]
    files = sorted((cache / directory).rglob(filename))
    if len(files) != 1:
        raise ValueError(f"expected one cached {source} split, found {len(files)}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=WORKSPACE / "RSI_Research/data/kev_upstream/hf_cache_holdout")
    parser.add_argument("--holdout-manifest", type=Path,
                        default=WORKSPACE / "RSI_Research/data/kev_upstream/holdout_manifest.json")
    parser.add_argument("--model-dir", type=Path, default=WORKSPACE / "models/Qwen3.5-9B")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=0,
        help="explicit prompt cap; 0 keeps every model-context-valid row",
    )
    args = parser.parse_args()
    if args.max_prompt_tokens < 0:
        raise ValueError("max prompt tokens must be nonnegative; use 0 for no artificial cap")
    max_prompt_tokens = args.max_prompt_tokens or None
    cache = args.cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    holdouts = json.loads(args.holdout_manifest.read_text())
    source_info = {item["id"]: item for item in holdouts["sources"]}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    output.mkdir(parents=True)
    reports = {}
    for source, (_, _, native_split) in SPLITS.items():
        native_file = cached_split(cache, source)
        dataset = Dataset.from_file(str(native_file))
        expected = json.loads((native_file.parent / "dataset_info.json").read_text())["splits"][native_split]["num_examples"]
        if len(dataset) != expected:
            raise ValueError(f"{source}: native split count changed")
        source_dir = output / source
        source_dir.mkdir()
        destination = source_dir / "D_train.jsonl"
        kept = 0
        too_long = 0
        max_options = 0
        with destination.open("x", encoding="utf-8") as handle:
            for index, native in enumerate(dataset):
                row, _ = normalize(source, native, index, dataset)
                row["id"] = f"{source}/{native_split}/{index:07d}"
                row["source"] = source_info[source]["repo"]
                row["source_split"] = native_split
                row["split"] = "D_train"
                row["provenance"] = {"label_source": "native_label", "raw_row_index": index,
                                     "source_revision": source_info[source]["revision"]}
                _, prompts = render_choice_prompts(row)
                lengths = ([len(ids) for ids in tokenizer(prompts, add_special_tokens=True)["input_ids"]]
                           if max_prompt_tokens is not None else [])
                if max_prompt_tokens is not None and max(lengths) > max_prompt_tokens:
                    too_long += 1
                    continue
                max_options = max(max_options, len(prompts))
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                kept += 1
        if kept == 0:
            raise ValueError(f"{source}: no eligible training rows")
        manifest = {
            "version": 1, "status": "complete", "episode": f"{source}_additional_choice_train_v1",
            "source": source_info[source]["repo"], "source_revision": source_info[source]["revision"],
            "source_split": native_split, "native_file": str(native_file),
            "native_file_sha256": file_sha256(native_file), "native_rows": expected,
            "holdout_manifest_sha256": file_sha256(args.holdout_manifest),
            "max_prompt_tokens": max_prompt_tokens, "option_count": max_options,
            "filtering": {"kept": kept, "too_long": too_long},
            "outputs": {"D_train": {"path": str(destination), "count": kept,
                                    "sha256": file_sha256(destination)}},
        }
        atomic_json(source_dir / "manifest.json", manifest)
        reports[source] = {"native_rows": expected, "kept": kept, "too_long": too_long,
                           "source_split": native_split}
        print(json.dumps({"source": source, **reports[source]}), flush=True)
    atomic_json(output / "manifest.json", {
        "version": 1, "status": "complete", "sources": reports,
        "total_rows": sum(report["kept"] for report in reports.values()),
        "purpose": "Additional training only; native tests and QNLI validation are excluded. MMLU-Pro has no native train split, so its separate 70-row validation split is training support.",
    })


if __name__ == "__main__":
    main()
