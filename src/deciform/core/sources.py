"""Explicit raw adapters; no silent exclusions or label inference."""

import csv
import gzip
import json
from pathlib import Path
from .schemas import normalize, semantic_key
from .data import sha256, atomic_json


def raw_rows(path):
    path = Path(path)
    suffixes = path.suffixes
    op = gzip.open if suffixes[-1:] == [".gz"] else open
    suffix = suffixes[-2] if suffixes[-1:] == [".gz"] else path.suffix
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        for b in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from b.to_pylist()
    elif suffix in (".csv", ".tsv"):
        with op(path, "rt", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")
    else:
        with op(path, "rt") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    raise ValueError(f"Blank raw record {i}")
                yield json.loads(line)


def field(row, key):
    for part in key.split("."):
        row = row[part]
    return row


def adapt(raw, recipe, index):
    name = recipe["source"]
    split = recipe["split"]
    task = recipe["task"]
    label = field(raw, recipe["label_field"])
    if "label_mapping" in recipe:
        key = semantic_key(label)
        if key not in recipe["label_mapping"]:
            raise ValueError(f"Unmapped raw label: {key}")
        label = recipe["label_mapping"][key]
    elif task == "score":
        label = float(label)
    text = recipe["text_template"].format_map(raw)
    question = {"type": task, "instructions": recipe["instructions_template"].format_map(raw)}
    identity = field(raw, recipe["id_field"]) if recipe.get("id_field") else index
    row = {
        "id": f"{name}/{split}/{identity}",
        "source": name,
        "source_split": split,
        "task": task,
        "state": {"text": text},
        "question": question,
        "label": label,
    }
    if "group_field" in recipe:
        row["group_id"] = field(raw, recipe["group_field"])
    if task == "choice":
        question["options"] = (
            field(raw, recipe["options_field"]) if "options_field" in recipe else recipe["options"]
        )
        if "option_ids" in recipe:
            question["option_ids"] = recipe["option_ids"]
        if "option_ids_field" in recipe:
            question["option_ids"] = field(raw, recipe["option_ids_field"])
        if recipe.get("label_is_index"):
            i = int(label)
            if i < 0 or i >= len(question["options"]):
                raise ValueError("Label index outside options")
            row["label"] = question.get("option_ids", question["options"])[i]
    elif task == "score":
        for k in ("score_support", "score_range", "label_kind"):
            if k in recipe:
                row[k] = recipe[k]
    return normalize(row, True)


def _write(rows, recipe, output, origin):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_name(output.name + ".partial")
    n = 0
    with temp.open("x") as f:
        for i, raw in enumerate(rows):
            try:
                row = adapt(raw, recipe, i)
            except Exception as e:
                raise ValueError(f"Raw row {i}: {e}") from e
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
    if not n:
        raise ValueError("Empty source split")
    temp.replace(output)
    manifest = {
        "schema_version": 1,
        "rows": n,
        "sha256": sha256(output),
        "path": output.name,
        "recipe": recipe,
        "origin": origin,
        "dropped": 0,
        "truncated": 0,
    }
    atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def convert(recipe, input, output):
    return _write(
        raw_rows(input), recipe, output, {"name": Path(input).name, "sha256": sha256(input)}
    )


def fetch(recipe, output):
    from datasets import load_dataset

    hf = recipe["huggingface"]
    revision = hf.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin dataset revision to a 40-character commit hash")
    data = load_dataset(hf["path"], name=hf.get("name"), split=hf["split"], revision=revision)
    return _write(data, recipe, output, {"huggingface": hf, "fingerprint": data._fingerprint})
