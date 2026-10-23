"""Materialize reproducible, non-training Choice evaluations from pinned holdouts.

The generated records are deliberately outside the adaptation protocol.  The
script records source revisions and every output hash so a later evaluation can
be audited without making the holdout visible to the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SOURCES = ("mmlu", "mmlu_pro", "sciq", "qnli", "paws", "emotion", "tweet_offensive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, source: str, row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}:{source}:{payload}".encode("utf-8")).hexdigest()


def class_names(dataset, field: str) -> list[str] | None:
    feature = dataset.features.get(field)
    names = getattr(feature, "names", None)
    return list(names) if names else None


def normalize(source: str, row: dict, index: int, dataset) -> tuple[dict, str]:
    if source == "mmlu":
        labels = ["A", "B", "C", "D"]
        definitions = [str(value).strip() for value in row["choices"]]
        state = row["question"].strip()
        instructions = "Which answer is correct for this question?"
        label = labels[int(row["answer"])]
    elif source == "mmlu_pro":
        definitions = [str(value).strip() for value in row["options"]]
        labels = [chr(ord("A") + offset) for offset in range(len(definitions))]
        state = row["question"].strip()
        instructions = "Which answer is correct for this question?"
        answer = str(row["answer"]).strip().upper()
        label = answer if answer in labels else labels[int(row["answer_index"])]
    elif source == "sciq":
        definitions = [str(row[key]).strip() for key in
                       ("correct_answer", "distractor1", "distractor2", "distractor3")]
        labels = ["A", "B", "C", "D"]
        state = row["support"].strip() + "\nQuestion: " + row["question"].strip()
        instructions = "Which answer correctly completes the question?"
        label = "A"
    elif source == "qnli":
        labels = class_names(dataset, "label") or ["entailment", "not_entailment"]
        definitions = labels[:]
        state = f"Question: {row['question'].strip()}\nSentence: {row['sentence'].strip()}"
        instructions = "Does the sentence entail the question?"
        label = labels[int(row["label"])]
    elif source == "paws":
        labels = class_names(dataset, "label") or ["0", "1"]
        definitions = ["not equivalent", "equivalent"]
        state = f"Sentence 1: {row['sentence1'].strip()}\nSentence 2: {row['sentence2'].strip()}"
        instructions = "Are the two sentences semantically equivalent?"
        label = labels[int(row["label"])]
    elif source in {"emotion", "tweet_offensive"}:
        labels = class_names(dataset, "label")
        if not labels:
            raise ValueError(f"missing class names for {source}")
        definitions = labels[:]
        state = row["text"].strip()
        instructions = ("What emotion is expressed by this text?"
                        if source == "emotion" else
                        "Is this tweet offensive?")
        label = labels[int(row["label"])]
    else:
        raise ValueError(f"unsupported holdout source: {source}")
    if len(definitions) < 2 or any(not value for value in definitions):
        raise ValueError(f"invalid option definitions: {source}/{index}")
    return {
        "id": f"holdout/{source}/{index:07d}",
        "source": source,
        "source_split": "pinned_test",
        "split": "sealed",
        "state": {"text": state},
        "question": {
            "type": "choice",
            "instructions": instructions,
            "options": definitions,
            "option_ids": labels,
            "option_definitions": definitions,
        },
        "label": label,
        "label_index": labels.index(label),
        "provenance": {"label_source": "pinned_native_label", "raw_row_index": index},
    }, label


def choose_rows(rows: list[tuple[dict, str, str]], per_label: int, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for row, label, key in rows:
        grouped[label].append((key, row))
    hidden: list[dict] = []
    transfer: list[dict] = []
    for label, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: item[0])
        needed = per_label * 2
        if len(candidates) < needed:
            raise ValueError(f"label {label!r} has only {len(candidates)} rows, need {needed}")
        hidden.extend(row for _, row in candidates[:per_label])
        transfer.extend(row for _, row in candidates[per_label:needed])
    return hidden, transfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-root", type=Path,
                        default=Path("RSI_Research/data/kev_upstream/holdout_hf"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--per-label", type=int, default=32)
    parser.add_argument("--source", action="append", choices=SOURCES)
    args = parser.parse_args()
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")
    from datasets import load_from_disk

    root = args.holdout_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    names = args.source or list(SOURCES)
    root_manifest = root.parent / "holdout_manifest.json"
    if not root_manifest.exists():
        raise FileNotFoundError(root_manifest)
    source_manifest = json.loads(root_manifest.read_text(encoding="utf-8"))
    manifest_by_id = {item["id"]: item for item in source_manifest["sources"]}
    records = {"final_hidden": [], "external_transfer": []}
    source_reports = {}
    for source in names:
        dataset = load_from_disk(str(root / source))
        info = manifest_by_id[source]
        if len(dataset) != info["rows"]:
            raise ValueError(f"row count changed for {source}: {len(dataset)} != {info['rows']}")
        rows = []
        for index, native in enumerate(dataset):
            row, label = normalize(source, native, index, dataset)
            rows.append((row, label, stable_key(args.seed, source, native)))
        hidden, transfer = choose_rows(rows, args.per_label, args.seed)
        for split, selected in (("final_hidden", hidden), ("external_transfer", transfer)):
            for row in selected:
                row = {**row, "split": split}
                records[split].append(row)
        source_reports[source] = {
            "repo": info["repo"], "revision": info["revision"], "config": info.get("config"),
            "native_split": info["split"], "native_rows": len(dataset),
            "source_path": str(root / source), "source_path_sha256": sha256(root / source / "data-00000-of-00001.arrow"),
            "final_hidden_count": len(hidden), "external_transfer_count": len(transfer),
        }
    output_files = {}
    for split, selected in records.items():
        selected.sort(key=lambda row: row["id"])
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_files[split] = {"path": str(path), "count": len(selected), "sha256": sha256(path)}
    manifest = {
        "version": 1, "status": "sealed", "protocol": "choice_sealed_evaluations_20261016_v1",
        "purpose": "frozen final-hidden and external-transfer Choice evaluations; never adaptation inputs",
        "seed": args.seed, "per_label": args.per_label, "sources": names,
        "holdout_manifest": str(root_manifest), "holdout_manifest_sha256": sha256(root_manifest),
        "source_reports": source_reports, "outputs": output_files,
        "selection": "stable SHA256 ordering within each native label; first per_label rows hidden, next per_label transfer",
        "controller_access": "forbidden",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "outputs": output_files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
