"""Task-native metrics with exact full-candidate coverage checks."""

import json
import math
from .schemas import normalize, support, semantic_key


def evaluate(rows, predictions):
    predictions = list(predictions)
    rows = list(rows)
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Prediction count mismatch or empty population")
    records = {}
    for p in predictions:
        k = semantic_key(p["id"])
        if k in records:
            raise ValueError("Duplicate prediction ID")
        records[k] = p
    totals = {}
    seen = set()
    for raw in rows:
        row = normalize(raw, True)
        key = semantic_key(row["id"])
        if key in seen:
            raise ValueError("Duplicate evaluation ID")
        seen.add(key)
        p = records.pop(key)
        values = support(row)
        if p["candidate_values"] != values:
            raise ValueError("Incomplete or misaligned candidate support")
        probabilities = p["probabilities"]
        if (
            len(probabilities) != len(values)
            or any(not math.isfinite(x) or x < 0 for x in probabilities)
            or abs(math.fsum(probabilities) - 1) > 1e-5
        ):
            raise ValueError("Invalid predictive distribution")
        task = row["task"]
        kind = "ranking" if "target_distribution" in row and task != "score" else task
        if row.get("native_candidate_count", len(values)) != len(values) or "training_view" in row:
            raise ValueError("Cannot evaluate sampled candidates")
        if p["task"] != task or semantic_key(p["prediction"]) not in list(
            map(semantic_key, values)
        ):
            raise ValueError("Invalid prediction type or value")
        t = totals.setdefault(task, {"rows": 0, "sum": 0.0, "_kind": kind})
        if t["_kind"] != kind:
            raise ValueError("Evaluate single-label and multi-positive Choice separately")
        t["rows"] += 1
        if task == "score":
            t["sum"] += abs(p["prediction"] - row["label"])
        elif "target_distribution" in row:
            labels = [float(v > 0) for v in row["target_distribution"]]
            order = sorted(
                range(len(values)), key=lambda i: (-probabilities[i], semantic_key(values[i]))
            )
            ranked = [labels[i] for i in order]
            npos = sum(labels)
            neg = [i for i, v in enumerate(labels) if not v]
            pos = [i for i, v in enumerate(labels) if v]
            if not pos or not neg:
                raise ValueError("Ranking metrics need positive and negative candidates")
            metrics = {
                "auc": sum(
                    (probabilities[i] > probabilities[j])
                    + 0.5 * (probabilities[i] == probabilities[j])
                    for i in pos
                    for j in neg
                )
                / (len(pos) * len(neg)),
                "mrr": sum(v / (i + 1) for i, v in enumerate(ranked)) / npos,
            }
            for k in (5, 10):
                metrics[f"ndcg@{k}"] = sum(
                    v / math.log2(i + 2) for i, v in enumerate(ranked[:k])
                ) / sum(1 / math.log2(i + 2) for i in range(min(k, len(pos))))
            for k, v in metrics.items():
                t[k] = t.get(k, 0.0) + v
        else:
            t["sum"] += float(semantic_key(p["prediction"]) == semantic_key(row["label"]))
    if records:
        raise ValueError("Extra predictions")
    for task, t in totals.items():
        n = t["rows"]
        t.pop("_kind")
        value = t.pop("sum")
        t["mae" if task == "score" else "accuracy"] = value / n
        if "auc" in t:
            t.pop("accuracy")
            for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
                t[k] /= n
    return {"rows": len(rows), "candidate_policy": "complete_native", "tasks": totals}


def permutation_audit(original, permuted):
    def index(rows):
        out = {}
        for r in rows:
            k = semantic_key(r["id"])
            if k in out:
                raise ValueError("Duplicate audit ID")
            out[k] = r
        return out

    for row in original + permuted:
        values = row["candidate_values"]
        prob = row["probabilities"]
        keys = list(map(semantic_key, values))
        if len(keys) != len(set(keys)) or len(keys) != len(prob) or not keys:
            raise ValueError("Invalid audit support")
        if any(not math.isfinite(p) or p < 0 for p in prob) or abs(math.fsum(prob) - 1) > 1e-5:
            raise ValueError("Invalid audit probabilities")
        if semantic_key(row["prediction"]) not in keys:
            raise ValueError("Audit prediction outside support")
    a, b = index(original), index(permuted)
    if a.keys() != b.keys() or not a:
        raise ValueError("Audit populations differ")
    results = {}
    for key, x in a.items():
        y = b[key]
        if x["task"] != y["task"]:
            raise ValueError("Task changed")
        pa = {semantic_key(v): p for v, p in zip(x["candidate_values"], x["probabilities"])}
        pb = {semantic_key(v): p for v, p in zip(y["candidate_values"], y["probabilities"])}
        if pa.keys() != pb.keys():
            raise ValueError("Candidate identities changed")
        t = results.setdefault(
            x["task"], {"rows": 0, "flips": 0, "max_probability_difference": 0.0}
        )
        t["rows"] += 1
        t["flips"] += semantic_key(x["prediction"]) != semantic_key(y["prediction"])
        t["max_probability_difference"] = max(
            t["max_probability_difference"], max(abs(pa[v] - pb[v]) for v in pa)
        )
    for t in results.values():
        t["flip_rate"] = t["flips"] / t["rows"]
    return {"tasks": results}
