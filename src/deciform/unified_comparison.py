"""Source-relationship reporting on the existing, immutable test population."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from deciform.joint_data import file_sha256
from deciform.evaluation import summarize_unified

GROUP_LABELS = {"unseen_source": "未见来源（相对我们的微调）",
                "trained_source_holdout": "已训练来源的 holdout"}


def relationship_inventory(manifest, rows):
    """Classify using actual training plans; never infer baseline exposure."""
    training = Counter()
    for reference in manifest["training_plans"].values():
        path = Path(reference["path"])
        if file_sha256(path) != reference["sha256"]:
            raise ValueError("training plan hash changed")
        for source in json.loads(path.read_text())["sources"]:
            training[source["name"]] += source["count"]
    source_counts = Counter(row["source_family"] for row in rows)
    source_groups = {source: "trained_source_holdout" if training[source] else "unseen_source"
                     for source in source_counts}
    for stream in manifest["streams"]:
        for source in stream["source_families"]:
            if bool(training[source]) != stream["training_source_family"]:
                raise ValueError(f"training relationship disagrees: {source}")
    groups = {}
    for group, label in GROUP_LABELS.items():
        counts = {s:n for s,n in sorted(source_counts.items()) if source_groups[s] == group}
        groups[group] = {"label": label, "record_count": sum(counts.values()),
                         "source_count": len(counts), "source_counts": counts}
    return {"classification_reference": "our completed parent + delta finetuning only",
            "record_count": len(rows), "groups": groups,
            "source_relationships": {s:{"group":source_groups[s],
                "test_records":source_counts[s], "our_training_records":training[s]}
                for s in sorted(source_counts)},
            "baseline_exposure": {"open_jev": "not established by our training audit",
                                    "kev": "not established by our training audit"}}


def validate_prediction(prediction, row):
    if prediction["id"] != row["id"] or prediction["label"] != row["label"]:
        raise ValueError("prediction ID/label mismatch")
    if prediction["source_family"] != row["source_family"]:
        raise ValueError("prediction source mismatch")
    options = row["question"].get("option_ids", row["question"]["options"])
    probs = prediction["probabilities"]
    if set(probs) != set(options) or prediction["prediction"] not in probs:
        raise ValueError("prediction option IDs mismatch")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values())-1)>1e-5:
        raise ValueError("invalid probability distribution")
    if abs(prediction["confidence"]-max(probs.values()))>1e-5:
        raise ValueError("confidence differs from maximum probability")
    if max(probs.values())-probs[prediction["prediction"]]>1e-5:
        raise ValueError("prediction differs from probability argmax")


def read_run_predictions(output, rows, manifest_sha256, *, require_complete=False, checkpoint=None):
    output = Path(output)
    expected = {r["id"]:r for r in rows}
    predictions, seen, workers = [], set(), []
    statuses = sorted(output.glob("rank_*_progress.json"))
    for path in statuses:
        status = json.loads(path.read_text())
        rank = status["rank"]
        provenance_path = output/f"rank_{rank}_manifest.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        count = status["evaluated_records"]
        if count and provenance is None:
            raise ValueError("evaluated rows without provenance")
        if provenance:
            if provenance["test_manifest_sha256"] != manifest_sha256:
                raise ValueError("model used a different test population")
            if checkpoint and provenance["checkpoint"] != str(Path(checkpoint).resolve()):
                raise ValueError("model checkpoint differs from comparison configuration")
        if count:
            prediction_path = output/f"rank_{rank}_predictions.jsonl"
            with prediction_path.open() as handle:
                for _ in range(count):
                    line = handle.readline()
                    if not line:
                        raise ValueError("reported progress exceeds saved predictions")
                    prediction = json.loads(line)
                    row = expected.get(prediction["id"])
                    if row is None or prediction["id"] in seen:
                        raise ValueError("unknown or duplicate prediction")
                    validate_prediction(prediction, row)
                    seen.add(prediction["id"])
                    predictions.append(prediction)
                if status["status"] == "complete" and any(line.strip() for line in handle):
                    raise ValueError("completed prediction file has uncounted rows")
            if status["status"] == "complete" and status.get("prediction_sha256") != file_sha256(prediction_path):
                raise ValueError("completed prediction file hash mismatch")
        workers.append({"progress": status, "provenance": provenance})
    complete = bool(workers) and all(w["progress"]["status"]=="complete" for w in workers) and seen==set(expected)
    if require_complete and not complete:
        raise ValueError("test prediction coverage is incomplete")
    failed = any(w["progress"]["status"]=="failed" for w in workers)
    state = "complete" if complete else "failed" if failed else "running" if workers else "pending"
    return predictions, {"status":state, "evaluated_records":len(predictions),
                         "expected_records":len(rows), "workers":workers}


def grouped_results(predictions, inventory):
    source_relationships = inventory["source_relationships"]
    groups = defaultdict(list)
    sources = defaultdict(list)
    for row in predictions:
        groups[source_relationships[row["source_family"]]["group"]].append(row)
        sources[row["source_family"]].append(row)
    def result(values, count):
        complete = len(values)==count
        return {"evaluated_records":len(values), "expected_records":count, "complete":complete,
                "metrics":summarize_unified(values)["aggregate"] if complete and values else None}
    return {"overall":result(predictions,inventory["record_count"]),
            "groups":{g:result(groups[g],v["record_count"]) for g,v in inventory["groups"].items()},
            "sources":{s:{"group":v["group"], **result(sources[s],v["test_records"])}
                       for s,v in source_relationships.items()}}
