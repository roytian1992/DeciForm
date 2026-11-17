"""Public request schema. Labels and metadata never enter model encodings."""

import copy
import json
import math
from decimal import Decimal

TASKS = ("choice", "noul", "score")


def semantic_key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def support(row):
    task = row["task"]
    q = row["question"]
    if task == "score":
        return list(row["score_support"])
    if task == "noul":
        return list(row.get("candidate_values", [False, True]))
    return list(q.get("option_ids", q["options"]))


def normalize(record, require_label=False):
    row = copy.deepcopy(record)
    q = row.get("question", {})
    if not isinstance(q, dict):
        raise ValueError("question must be an object")
    row["task"] = row.get("task", q.get("type"))
    if row["task"] not in TASKS:
        raise ValueError("Explicit task must be choice, noul, or score")
    if "type" in q and q["type"] != row["task"]:
        raise ValueError("Conflicting public task types")
    q["type"] = row["task"]
    if isinstance(row.get("state"), str):
        row["state"] = {"text": row["state"]}
    if not isinstance(row.get("state", {}).get("text"), str):
        raise ValueError("state.text must be a string")
    if not isinstance(q.get("instructions"), str):
        raise ValueError("question.instructions must be a string")
    if row["task"] == "score":
        if "score_support" not in row:
            spec = row.get("score_range", q.get("score_range"))
            if not isinstance(spec, dict) or set(spec) != {"min", "max", "step"}:
                raise ValueError("Declare full score_support or score_range with min/max/step")
            lo, hi, step = (Decimal(str(spec[k])) for k in ("min", "max", "step"))
            if not all(v.is_finite() for v in (lo, hi, step)) or step <= 0 or hi <= lo:
                raise ValueError("Invalid score range")
            count = (hi - lo) / step
            if count != count.to_integral_value() or count > 100000:
                raise ValueError("Grid must end at max and have at most 100001 values")
            row["score_support"] = [float(lo + i * step) for i in range(int(count) + 1)]
        values = row["score_support"]
        if not isinstance(values, list):
            raise ValueError("score_support must be a list")
        if len(values) < 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        ):
            raise ValueError("Decimal-prefix readout supports finite nonnegative numeric values")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("score_support must strictly increase")
        row.setdefault(
            "label_kind",
            "native_integer" if all(float(v).is_integer() for v in values) else "continuous",
        )
        if row["label_kind"] not in ("native_integer", "continuous"):
            raise ValueError("Unknown label_kind")
    elif row["task"] == "noul":
        values = row.get("candidate_values", [False, True])
        if (
            len(values) != 2
            or any(type(v) is not bool for v in values)
            or set(values) != {False, True}
        ):
            raise ValueError("Noul support must contain both boolean values")
        row["candidate_values"] = list(values)
    else:
        options = q.get("options")
        if (
            not isinstance(options, list)
            or len(options) < 2
            or any(not isinstance(v, str) for v in options)
        ):
            raise ValueError("Choice requires at least two option strings")
        for key in ("option_ids", "option_definitions"):
            if key in q and (not isinstance(q[key], list) or len(q[key]) != len(options)):
                raise ValueError("Misaligned " + key)
        if "option_definitions" in q and any(
            not isinstance(t, str) for t in q["option_definitions"]
        ):
            raise ValueError("Definitions must be text")
    values = support(row)
    keys = list(map(semantic_key, values))
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate semantic identities must be unique")
    if require_label and row["task"] == "score" and row.get("label") is None:
        raise ValueError("Score requires the raw numerical label")
    if require_label and "label" not in row and "target_distribution" not in row:
        raise ValueError("Training requires a label or explicit target_distribution")
    if "target_distribution" in row:
        target = row["target_distribution"]
        if (
            len(target) != len(values)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in target)
            or abs(math.fsum(target) - 1) > 1e-7
        ):
            raise ValueError("target_distribution must align with support and sum to one")
    if "label" in row and row["label"] is not None:
        y = row["label"]
        if row["task"] == "score":
            if (
                type(y) not in (int, float)
                or not math.isfinite(y)
                or not values[0] <= y <= values[-1]
            ):
                raise ValueError("Score label outside declared range")
        elif semantic_key(y) not in keys:
            raise ValueError("Label must be a candidate identity")
    elif require_label and "target_distribution" not in row:
        raise ValueError("Missing target")
    return row


def public_input(record):
    row = normalize(record)
    question = {
        k: v
        for k, v in row["question"].items()
        if k in ("type", "instructions", "options", "option_ids", "option_definitions")
    }
    result = {"task": row["task"], "state": {"text": row["state"]["text"]}, "question": question}
    if "json_keys" in row:
        result["json_keys"] = row["json_keys"]
    for k in ("candidate_values", "score_support", "label_kind"):
        if k in row:
            result[k] = row[k]
    return result


def canonical_order(row):
    values = support(row)
    return (
        list(range(len(values)))
        if row["task"] == "score"
        else sorted(range(len(values)), key=lambda i: semantic_key(values[i]))
    )
