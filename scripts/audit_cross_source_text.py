"""Report exact text reuse across registered training and visible roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


VISIBLE = ("D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible")


def text_key(row: dict) -> int:
    normalized = " ".join(row["state"]["text"].casefold().split())
    return int.from_bytes(hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest(), "big")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.protocol_root.resolve()
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if index["status"] != "complete":
        raise ValueError("protocol root is incomplete")
    owner: dict[int, str] = {}
    training_pairs = Counter()
    train_records = 0
    for source in index["episodes"]:
        for row in rows(root / source / "D_train.jsonl"):
            train_records += 1
            key = text_key(row)
            previous = owner.setdefault(key, source)
            if previous != source:
                training_pairs[tuple(sorted((previous, source)))] += 1
    visible_pairs = Counter()
    visible_records = 0
    for source in index["episodes"]:
        for role in VISIBLE:
            for row in rows(root / source / f"{role}.jsonl"):
                visible_records += 1
                previous = owner.get(text_key(row))
                if previous is not None and previous != source:
                    visible_pairs[(previous, source, role)] += 1
    report = {
        "status": "complete", "protocol_root": str(root),
        "matching_rule": "casefolded whitespace-normalized state.text; 128-bit BLAKE2b",
        "training_records": train_records, "visible_records": visible_records,
        "cross_source_train_duplicate_count": sum(training_pairs.values()),
        "cross_source_train_pairs": [
            {"source_a": left, "source_b": right, "count": count}
            for (left, right), count in sorted(training_pairs.items())
        ],
        "visible_text_in_other_source_train_count": sum(visible_pairs.values()),
        "visible_pairs": [
            {"train_source": train, "visible_source": visible,
             "visible_role": role, "count": count}
            for (train, visible, role), count in sorted(visible_pairs.items())
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
