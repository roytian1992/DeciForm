"""Evaluate a trained candidate scorer on a visible normalized choice split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deciform.choice import (  # noqa: E402
    ChoiceHead,
    choice_response,
    encode_prompts,
    render_choice_prompts,
)
from deciform.adapter_program import (  # noqa: E402
    STANDARD_PROGRAM, load_program_adapter,
)
from deciform.latent_program import LATENT_PROGRAM, initialize_latent  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=0,
        help="explicit prompt cap; 0 uses the model context without an artificial protocol cap",
    )
    parser.add_argument("--backbone-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--evaluation-scope", default="visible_dev",
        help="provenance label for this evaluation, e.g. visible_dev, final_hidden, or external_transfer",
    )
    args = parser.parse_args()
    if args.max_prompt_tokens < 0:
        raise ValueError("max prompt tokens must be nonnegative; use 0 for no artificial cap")
    max_prompt_tokens = args.max_prompt_tokens or None
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run on exactly one visible CUDA device")
    torch.set_num_threads(4)
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = args.model_dir.resolve()
    checkpoint_path = args.head_checkpoint.resolve()
    input_path = args.input.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("backbone_dtype", "bf16") != args.backbone_dtype:
        raise ValueError("checkpoint feature dtype and requested backbone dtype differ")
    if checkpoint["model_config_sha256"] != sha256(model_dir / "config.json"):
        raise ValueError("checkpoint and backbone config do not match")
    with input_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("input must have unique, nonempty record IDs")
    prompts = []
    sizes = []
    for row in rows:
        option_ids, row_prompts = render_choice_prompts(row)
        if row["label"] not in option_ids:
            raise ValueError(f"label is not an option: {row['id']}")
        prompts.extend(row_prompts)
        sizes.append(len(option_ids))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = checkpoint.get("padding_side", "left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32 if args.backbone_dtype == "fp32" else torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir else None
    if adapter_dir is not None:
        adapter_config = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        adapter_program = adapter_config.get("adapter_program", STANDARD_PROGRAM)
        if adapter_program == STANDARD_PROGRAM:
            model.model = PeftModel.from_pretrained(
                model.model, adapter_dir, is_trainable=False,
            )
        elif adapter_program == LATENT_PROGRAM:
            model.model, _ = initialize_latent(
                model.model, adapter_config["genome"], int(adapter_config["rank"]),
                adapter_config["target_modules"], adapter_dir,
            )
        else:
            load_program_adapter(
                model.model, adapter_dir, adapter_program,
                int(adapter_config["rank"]), adapter_config["target_modules"],
            )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("backbone did not load on the visible GPU")
    head = ChoiceHead(checkpoint["hidden_size"]).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    forward_started = time.time()
    features = encode_prompts(
        model, tokenizer, prompts, device=device,
        batch_size=args.candidate_batch_size,
        max_prompt_tokens=max_prompt_tokens,
        bucket_by_length=max_prompt_tokens is None,
    )
    forward_seconds = time.time() - forward_started
    temperature = float(checkpoint["temperature"])
    prediction_rows = []
    cursor = 0
    for row, size in zip(rows, sizes, strict=True):
        option_ids, _ = render_choice_prompts(row)
        with torch.inference_mode():
            logits = head(features[cursor : cursor + size].to(device))
        response = choice_response(option_ids, logits, temperature=temperature)
        prediction_rows.append({"id": row["id"], "label": row["label"], **response})
        cursor += size

    first_ids, first_prompts = render_choice_prompts(rows[0])
    reverse = list(reversed(range(len(first_ids))))
    reverse_ids, reverse_prompts = render_choice_prompts(rows[0], reverse)
    reverse_features = encode_prompts(
        model, tokenizer, reverse_prompts, device=device,
        batch_size=args.candidate_batch_size,
        max_prompt_tokens=max_prompt_tokens,
        bucket_by_length=max_prompt_tokens is None,
    )
    with torch.inference_mode():
        reverse_response = choice_response(
            reverse_ids, head(reverse_features.to(device)), temperature=temperature
        )
    permutation_error = max(
        abs(prediction_rows[0]["probabilities"][option] - reverse_response["probabilities"][option])
        for option in first_ids
    )
    if permutation_error > 1e-3:
        raise RuntimeError(f"end-to-end option permutation failed: {permutation_error}")

    labels = [row["label"] for row in prediction_rows]
    predicted = [row["prediction"] for row in prediction_rows]
    correct = [left == right for left, right in zip(labels, predicted, strict=True)]
    all_labels = sorted(set(labels))
    gold_count = Counter(labels)
    predicted_count = Counter(predicted)
    true_positive = Counter(label for label, passed in zip(labels, correct, strict=True) if passed)
    f1 = []
    for label in all_labels:
        precision = true_positive[label] / predicted_count[label] if predicted_count[label] else 0.0
        recall = true_positive[label] / gold_count[label]
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    confidences = [row["confidence"] for row in prediction_rows]
    ece = 0.0
    for bin_index in range(10):
        indices = [
            index for index, value in enumerate(confidences)
            if min(int(value * 10), 9) == bin_index
        ]
        if indices:
            ece += len(indices) / len(rows) * abs(
                sum(confidences[index] for index in indices) / len(indices)
                - sum(correct[index] for index in indices) / len(indices)
            )
    brier = sum(
        sum((probability - float(option == row["label"])) ** 2
            for option, probability in row["probabilities"].items())
        for row in prediction_rows
    ) / len(rows)
    nll = -sum(
        torch.log(torch.tensor(max(row["probabilities"][row["label"]], 1e-30))).item()
        for row in prediction_rows
    ) / len(rows)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "complete",
        "scope": f"{args.evaluation_scope} choice checkpoint evaluation",
        "model_dir": str(model_dir),
        "model_config_sha256": sha256(model_dir / "config.json"),
        "head_checkpoint": str(checkpoint_path),
        "head_checkpoint_sha256": sha256(checkpoint_path),
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "adapter_model_sha256": sha256(adapter_dir / "adapter_model.safetensors") if adapter_dir else None,
        "adapter_program": (
            json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            .get("adapter_program", STANDARD_PROGRAM)
            if adapter_dir else None
        ),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "predictions": str(predictions_path),
        "record_count": len(rows),
        "candidate_count": len(prompts),
        "class_count": len(all_labels),
        "temperature": temperature,
        "backbone_dtype": args.backbone_dtype,
        "max_prompt_tokens": max_prompt_tokens,
        "padding_side": tokenizer.padding_side,
        "accuracy": sum(correct) / len(rows),
        "macro_f1": sum(f1) / len(f1),
        "brier_multiclass_sum": brier,
        "ece_10_equal_width": ece,
        "nll": nll,
        "end_to_end_permutation_max_probability_error": permutation_error,
        "candidate_forward_seconds": forward_seconds,
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
