"""Versioned JSONL preparation; original splits and full evaluation supports persist."""

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from collections import Counter
from .schemas import normalize, public_input, support, semantic_key
from .objectives import interpolation


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_jsonl(path, require_label=False):
    path = Path(path)
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"{path}:{i}: blank records are not silently skipped")
            try:
                yield normalize(json.loads(line), require_label)
            except Exception as e:
                raise ValueError(f"{path}:{i}: {e}") from e


def fingerprint(row):
    value = public_input(row)
    if row["task"] == "choice":
        q = value["question"]
        ids = support(row)
        order = sorted(range(len(ids)), key=lambda i: semantic_key(ids[i]))
        if "json_keys" in value:
            value["json_keys"] = [value["json_keys"][i] for i in order]
        for field in ("options", "option_ids", "option_definitions"):
            if field in q:
                q[field] = [q[field][i] for i in order]
    if row["task"] == "noul":
        value["candidate_values"] = [False, True]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_view(record, negative_count=3, seed=20261106, views=1, epoch=0):
    row = normalize(record, True)
    if negative_count < 1 or views < 1:
        raise ValueError("Positive negative_count and positive views required")
    if row["task"] != "choice":
        return row
    if "training_view" in row or row.get("native_candidate_count", len(support(row))) != len(
        support(row)
    ):
        raise ValueError("Prepare sampling from complete native candidates, not an existing view")
    values = support(row)
    positive = (
        [i for i, p in enumerate(row["target_distribution"]) if p > 0]
        if "target_distribution" in row
        else [list(map(semantic_key, values)).index(semantic_key(row["label"]))]
    )
    negatives = [i for i in range(len(values)) if i not in positive]
    # Union of disjoint views: each candidate is encoded once, each question has weight one.
    key = json.dumps([seed, row.get("id", fingerprint(row)), epoch], sort_keys=True).encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest(), 16))
    negatives.sort(key=lambda i: semantic_key(values[i]))
    rng.shuffle(negatives)
    selected = set(positive + negatives[: negative_count * views])
    indices = [i for i in range(len(values)) if i in selected]
    if len(indices) < 2:
        raise ValueError("Training view must retain at least two candidates")
    q = row["question"]
    for k in ("options", "option_ids", "option_definitions"):
        if k in q:
            q[k] = [q[k][i] for i in indices]
    for k in ("target_distribution", "json_keys", "click_labels"):
        if k in row:
            row[k] = [row[k][i] for i in indices]
    row["native_candidate_count"] = record.get("native_candidate_count", len(values))
    row["training_view"] = {
        "negative_count": negative_count,
        "views": views,
        "aggregation": "union",
        "seed": seed,
        "epoch": epoch,
    }
    return normalize(row, True)


def prepare(sources, out, negative_count=3, views=1, seed=20261106):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "splits": {},
        "evaluation_candidates": "complete_native",
        "sampling": {"negative_count": negative_count, "views": views, "kind": "union"},
        "truncated": 0,
        "dropped": 0,
    }
    occupied = {}
    groups = {}
    for split, path in sources.items():
        if split not in ("train", "dev", "test"):
            raise ValueError("Use train/dev/test; preserve source split names in manifest")
        target = out / f"{split}.jsonl"
        sample = out / "train.views.jsonl"
        ids = set()
        tasks = Counter()
        sources_count = Counter()
        n = 0
        with target.open("x") as f:
            view_file = sample.open("x") if split == "train" else None
            try:
                for row in read_jsonl(path, True):
                    row.setdefault("id", f"{split}:{n}")
                    if semantic_key(row["id"]) in ids:
                        raise ValueError("Duplicate record ID within split")
                    ids.add(semantic_key(row["id"]))
                    fp = fingerprint(row)
                    if fp in occupied and occupied[fp] != split:
                        raise ValueError("Identical model input crosses splits")
                    occupied[fp] = split
                    if "group_id" in row:
                        key = (
                            row.get("source_family", row.get("source", "")),
                            semantic_key(row["group_id"]),
                        )
                        if key in groups and groups[key] != split:
                            raise ValueError("Source group crosses splits")
                        groups[key] = split
                    # Official native evaluation input must not already be a sampled view.
                    if (
                        split != "train"
                        and row["task"] == "choice"
                        and (
                            row.get("native_candidate_count", len(support(row)))
                            != len(support(row))
                            or "training_view" in row
                        )
                    ):
                        raise ValueError("Evaluation input contains sampled candidates")
                    if row["task"] == "score":
                        row["score_target_distribution"] = interpolation(support(row), row["label"])
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    if view_file:
                        view_file.write(
                            json.dumps(
                                training_view(row, negative_count, seed, views),
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    tasks[row["task"]] += 1
                    sources_count[row.get("source_family", row.get("source", "unspecified"))] += 1
                    n += 1
            finally:
                if view_file:
                    view_file.close()
        if not n:
            raise ValueError("Empty split")
        desc = {
            "path": target.name,
            "rows": n,
            "sha256": sha256(target),
            "tasks": dict(tasks),
            "sources": dict(sources_count),
            "input_sha256": sha256(path),
            "input_name": Path(path).name,
        }
        if split == "train":
            desc["training_view"] = {"path": sample.name, "rows": n, "sha256": sha256(sample)}
        manifest["splits"][split] = desc
    atomic_json(out / "manifest.json", manifest)
    return manifest


def verify_release(path):
    path = Path(path)
    m = json.loads(path.read_text())
    for split, d in m["splits"].items():
        for desc in [d] + ([d["training_view"]] if "training_view" in d else []):
            f = path.parent / desc["path"]
            if sha256(f) != desc["sha256"]:
                raise ValueError("Checksum mismatch: " + str(f))
            rows = list(read_jsonl(f, True))
            if len(rows) != desc["rows"]:
                raise ValueError("Row count mismatch")
    return m
