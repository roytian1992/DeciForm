"""Locked test inventory loading and source-aware probability metrics."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from deciform.metrics import summarize
from deciform.joint_data import file_sha256


def text_key(row):
    text = " ".join(row["state"]["text"].casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inventory(path):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("status") != "locked"
            or manifest.get("evaluation_only") is not True
            or manifest.get("training_allowed") is not False
            or manifest.get("checkpoint_selection_allowed") is not False):
        raise ValueError("test inventory must be locked and evaluation-only")
    rows, seen, counts = [], set(), Counter()
    for stream in manifest["streams"]:
        source_path = Path(stream["path"])
        if file_sha256(source_path) != stream["sha256"]:
            raise ValueError(f"test input hash mismatch: {source_path}")
        local_count = raw_count = 0
        excluded = set(stream.get("exclude_ids", []))
        excluded_seen = set()
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_count += 1
                if row["id"] in excluded:
                    excluded_seen.add(row["id"])
                    continue
                if row["id"] in seen:
                    raise ValueError(f"duplicate test ID: {row['id']}")
                seen.add(row["id"])
                families = stream["source_families"]
                family = families[0] if len(families) == 1 else row["source"]
                if family not in families:
                    raise ValueError(f"unknown source family: {family}")
                options = row["question"].get("option_ids", row["question"]["options"])
                if row["question"]["type"] != "choice" or row["label"] not in options:
                    raise ValueError("invalid choice test record")
                row = {**row, "source_family": family, "test_stream": stream["name"]}
                rows.append(row)
                counts[family] += 1
                local_count += 1
        if local_count != stream["count"]:
            raise ValueError(f"test stream count mismatch: {stream['name']}")
        if excluded_seen != excluded or raw_count != stream.get("raw_count", stream["count"]):
            raise ValueError(f"test stream exclusion mismatch: {stream['name']}")
    if len(rows) != manifest["record_count"] or len(counts) != manifest["unique_source_family_count"]:
        raise ValueError("test inventory count mismatch")
    if set(counts) != set(manifest["source_families"]):
        raise ValueError("test inventory source mismatch")
    return manifest, rows, dict(sorted(counts.items()))


def summarize_unified(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[row["source_family"]].append(row)
    sources = {name: summarize(rows) for name, rows in sorted(groups.items())}
    # Source-qualified labels prevent unrelated class names being conflated.
    pooled = []
    for row in predictions:
        prefix = row["source_family"] + "::"
        pooled.append({**row, "label": prefix + row["label"],
                       "prediction": prefix + row["prediction"],
                       "probabilities": {prefix+k: v for k,v in row["probabilities"].items()}})
    summary = summarize(pooled)
    aggregate = {k: summary[k] for k in ("record_count", "accuracy", "brier", "ece", "nll",
                                             "accuracy_wilson_95", "reliability_bins", "fixed_threshold_risk")}
    aggregate["macro_f1"] = sum(s["macro_f1"] for s in sources.values()) / len(sources)
    aggregate["macro_f1_definition"] = "unweighted mean of per-source macro F1"
    aggregate["probability_metric_weighting"] = "one vote per test record; ECE recomputed on pooled records"
    return {"aggregate": aggregate, "source_metrics": sources,
            "record_count": len(predictions), "unique_source_family_count": len(sources)}
