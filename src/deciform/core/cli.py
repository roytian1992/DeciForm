"""Public command line: python -m deciform.core --help."""

import argparse
import json
from pathlib import Path
from .data import read_jsonl, prepare, verify_release, atomic_json


def read_config(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="deciform",
        description="DeciForm data, training, prediction and full-candidate evaluation",
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("validate")
    a.add_argument("input")
    a.add_argument("--require-label", action="store_true")
    a = sub.add_parser("convert")
    a.add_argument("--recipe", required=True)
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("fetch")
    a.add_argument("--recipe", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("prepare")
    for key in ("train", "dev", "test"):
        a.add_argument("--" + key, required=key == "train")
    a.add_argument("--output", required=True)
    a.add_argument("--negative-count", type=int, default=3)
    a.add_argument("--views", type=int, default=1)
    a.add_argument("--seed", type=int, default=20261106)
    a = sub.add_parser("prepare-mind")
    for key in ("news", "behaviors", "split", "output"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--negative-count", type=int, default=3)
    a.add_argument("--seed", type=int, default=20261114)
    a = sub.add_parser("verify")
    a.add_argument("manifest")
    a = sub.add_parser("train")
    for key in ("model-config", "train-config", "data", "output"):
        a.add_argument("--" + key, required=True)
    group = a.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--initialize-from")
    a = sub.add_parser("predict")
    for key in ("checkpoint", "input", "output"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--base-model")
    a.add_argument("--device", default="cuda")
    a.add_argument("--batch-size", type=int, default=1)
    a = sub.add_parser("evaluate")
    for key in ("data", "predictions", "output"):
        a.add_argument("--" + key, required=True)
    a = sub.add_parser("permutation-audit")
    for key in ("original", "permuted", "output"):
        a.add_argument("--" + key, required=True)
    args = p.parse_args(argv)
    if args.command == "validate":
        from collections import Counter

        counts = Counter(r["task"] for r in read_jsonl(args.input, args.require_label))
        result = {"rows": sum(counts.values()), "tasks": dict(counts)}
        if not result["rows"]:
            raise ValueError("Empty input")
    elif args.command in ("convert", "fetch"):
        from .sources import convert, fetch

        result = (
            convert(read_config(args.recipe), args.input, args.output)
            if args.command == "convert"
            else fetch(read_config(args.recipe), args.output)
        )
    elif args.command == "prepare":
        result = prepare(
            {s: getattr(args, s) for s in ("train", "dev", "test") if getattr(args, s)},
            args.output,
            args.negative_count,
            args.views,
            args.seed,
        )
    elif args.command == "prepare-mind":
        from .mind import prepare_mind

        result = prepare_mind(
            args.news, args.behaviors, args.split, args.output, args.negative_count, args.seed
        )
    elif args.command == "verify":
        result = verify_release(args.manifest)
    elif args.command == "train":
        from .training import train

        train(
            read_config(args.model_config),
            read_config(args.train_config),
            args.data,
            args.output,
            args.resume,
            args.initialize_from,
        )
        return
    elif args.command == "predict":
        import torch
        from .checkpoint import load_export

        if args.batch_size < 1:
            raise ValueError("Positive batch size required")
        model = load_export(args.checkpoint, args.device, args.base_model)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(out)
        temp = out.with_name(out.name + ".partial")
        count = 0
        batch = []
        with temp.open("x") as f:

            def flush():
                nonlocal count
                with torch.autocast(
                    device_type=model.device.type,
                    dtype=torch.bfloat16,
                    enabled=model.device.type == "cuda" and model.config.get("bf16", True),
                ):
                    predictions = model.predict(batch)
                for pred in predictions:
                    f.write(json.dumps(pred, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
                batch.clear()

            for i, row in enumerate(read_jsonl(args.input)):
                row.setdefault("id", str(i))
                batch.append(row)
                if len(batch) == args.batch_size:
                    flush()
            if batch:
                flush()
        temp.replace(out)
        result = {"predictions": count, "output": str(out)}
    elif args.command == "evaluate":
        from .metrics import evaluate

        with open(args.predictions) as f:
            result = evaluate(read_jsonl(args.data, True), (json.loads(line) for line in f))
        atomic_json(args.output, result)
    else:
        from .metrics import permutation_audit

        with open(args.original) as a, open(args.permuted) as b:
            result = permutation_audit([json.loads(x) for x in a], [json.loads(x) for x in b])
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
