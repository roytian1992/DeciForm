"""Run a fixed-budget, visible Choice LoRA adaptation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import heapq
import random
from collections import deque
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deciform.choice import (  # noqa: E402
    ChoiceHead, last_token_pool, render_choice_prompts,
)
from deciform.adapter_program import (  # noqa: E402
    ADAPTER_PROGRAMS, STANDARD_PROGRAM, install_program, load_program_adapter,
    program_metadata, program_parameters, save_program_adapter, validate_program,
)
from deciform.adapter_transfer import initialize_adapter  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_examples(rows: list[dict], seed: int, rows_per_label: int | None) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    selected = []
    for label in sorted(groups):
        candidates = sorted(
            groups[label],
            key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
        )
        if rows_per_label is None:
            selected.extend(candidates)
        else:
            if len(candidates) < rows_per_label:
                raise ValueError(f"too few support examples for {label}")
            selected.extend(candidates[:rows_per_label])
    return sorted(
        selected,
        key=lambda row: hashlib.sha256(f"train:{seed}:{row['id']}".encode()).hexdigest(),
    )


def select_examples_streaming(path: Path, seed: int, rows_per_label: int) -> list[dict]:
    """Select the same lowest-hash examples as select_examples with bounded memory."""
    if rows_per_label < 1:
        raise ValueError("rows_per_label must be positive")
    heaps: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score = int(hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), 16)
            heap = heaps[row["label"]]
            candidate = (-score, row)
            if len(heap) < rows_per_label:
                heapq.heappush(heap, candidate)
            elif candidate[0] > heap[0][0]:
                heapq.heapreplace(heap, candidate)
    selected = []
    for label, heap in sorted(heaps.items()):
        if len(heap) < rows_per_label:
            raise ValueError(f"too few support examples for {label}")
        selected.extend(row for _, row in heap)
    return sorted(selected, key=lambda row: hashlib.sha256(
        f"train:{seed}:{row['id']}".encode()).hexdigest())


def iter_full_training_rows(path: Path, seed: int, epoch: int,
                            chunk_size: int = 4096):
    """Visit every training row once per epoch with bounded-memory chunk shuffling."""
    rng = random.Random(f"{seed}:{epoch}")
    chunk = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                rng.shuffle(chunk)
                yield from chunk
                chunk.clear()
    if chunk:
        rng.shuffle(chunk)
        yield from chunk


def choose_candidates(row: dict, seed: int, negative_count: int) -> list[int]:
    option_ids, _ = render_choice_prompts(row)
    positive = option_ids.index(row["label"])
    negatives = sorted(
        (index for index in range(len(option_ids)) if index != positive),
        key=lambda index: hashlib.sha256(
            f"negative:{seed}:{row['id']}:{option_ids[index]}".encode()
        ).hexdigest(),
    )[:negative_count]
    if len(negatives) != negative_count:
        raise ValueError("candidate budget exceeds available negatives")
    return sorted(
        [positive, *negatives],
        key=lambda index: hashlib.sha256(
            f"order:{seed}:{row['id']}:{option_ids[index]}".encode()
        ).hexdigest(),
    )


def candidate_subset(row: dict, indices: list[int]) -> dict:
    question = row["question"].copy()
    for key in ("options", "option_ids", "option_definitions"):
        if key in question:
            question[key] = [question[key][index] for index in indices]
    return {**row, "question": question}


def fit_choice_prompt(row: dict, candidate_index: int, tokenizer,
                      max_prompt_tokens: int) -> tuple[str, bool]:
    """Fit a prompt by trimming state text while preserving the decision suffix."""
    question = row["question"]
    definitions = question.get("option_definitions", question["options"])

    def render(state_text: str) -> str:
        return (
            f"State: {state_text}\n"
            f"Question: {question['instructions']}\n"
            f"Candidate: {definitions[candidate_index]}"
        )

    prompt = render(row["state"]["text"])
    if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) <= max_prompt_tokens:
        return prompt, False
    state_tokens = tokenizer(row["state"]["text"], add_special_tokens=False)["input_ids"]
    if len(tokenizer(render(""), add_special_tokens=True)["input_ids"]) > max_prompt_tokens:
        raise ValueError("question and candidate exceed the training token budget")
    low, high = 0, len(state_tokens)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        state_text = tokenizer.decode(state_tokens[:middle], skip_special_tokens=True)
        candidate_prompt = render(state_text)
        length = len(tokenizer(candidate_prompt, add_special_tokens=True)["input_ids"])
        if length <= max_prompt_tokens:
            best = candidate_prompt
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ValueError("no state text fits the training token budget")
    return best, True


def base_adapter_from_report(
    report_path: Path, head_path: Path, checkpoint: dict, model_dir: Path, rank: int,
    adapter_program: str,
) -> tuple[Path, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] not in {"base_lora_candidate_complete", "pilot_train_complete", "full_train_complete"}:
        raise ValueError("expected a completed LoRA run")
    if report["model_config_sha256"] != sha256(model_dir / "config.json"):
        raise ValueError("base adapter and backbone config differ")
    if report["rank"] != rank:
        raise ValueError("continued adapter rank differs from the requested rank")
    source_program = report.get("adapter_program", STANDARD_PROGRAM)
    if sha256(head_path) != report["head_sha256"] and (
        checkpoint.get("parent_head_sha256") != report["head_sha256"]
    ):
        raise ValueError("base adapter and Choice head have different parents")
    adapter_dir = Path(report["adapter_path"]).resolve()
    adapter_hash = sha256(adapter_dir / "adapter_model.safetensors")
    if adapter_hash != report["adapter_model_sha256"]:
        raise ValueError("base adapter hash differs from its run report")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if source_program == STANDARD_PROGRAM:
        if config["r"] != rank or set(config["target_modules"]) != {"q_proj", "v_proj"}:
            raise ValueError("base adapter LoRA configuration differs from the pilot")
    else:
        if config.get("adapter_program") != source_program:
            raise ValueError("base structured adapter configuration differs from the pilot")
    return adapter_dir, adapter_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--base-run-report", type=Path)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--rows-per-label", type=int, default=1)
    parser.add_argument("--all-training-rows", action="store_true")
    parser.add_argument("--negative-count", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--adapter-program", choices=ADAPTER_PROGRAMS,
        default=STANDARD_PROGRAM,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("run on exactly one visible GPU")
    if (not args.all_training_rows and args.rows_per_label < 1) or args.negative_count < 1 or args.rank < 1 or not 1 <= args.epochs <= 8 or args.micro_batch_size < 1:
        raise ValueError("adaptation budget and rank must be positive")
    validate_program(args.adapter_program, args.rank)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.time()
    model_dir = args.model_dir.resolve()
    head_path = args.head_checkpoint.resolve()
    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = episode_dir / "manifest.json"
    train_path = episode_dir / "D_train.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilot_episode = manifest["episode"] in (
        "banking77_visible_adapt_pilot_20261015",
        "trec_visible_adapt_pilot_20261015",
        "sst5_visible_adapt_pilot_20261015",
    )
    full_episode = manifest.get("version") == 2 and manifest.get("status") == "complete"
    if not pilot_episode and not full_episode:
        raise ValueError("unexpected episode")
    if full_episode:
        for role in ("D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"):
            role_path = episode_dir / f"{role}.jsonl"
            if sha256(role_path) != manifest["outputs"][role]["sha256"]:
                raise ValueError(f"episode role hash mismatch: {role}")
    protocol_max_prompt_tokens = int(manifest.get("max_prompt_tokens", args.max_prompt_tokens))
    if not 1 <= args.max_prompt_tokens <= protocol_max_prompt_tokens:
        raise ValueError("training token cap must not exceed the episode protocol cap")
    if sha256(train_path) != manifest["outputs"]["D_train"]["sha256"]:
        raise ValueError("episode train hash mismatch")
    if args.negative_count >= manifest["option_count"]:
        raise ValueError("negative count exceeds the episode's available options")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    if checkpoint["model_config_sha256"] != sha256(model_dir / "config.json"):
        raise ValueError("base head and backbone config differ")
    if checkpoint["backbone_dtype"] != "fp32":
        raise ValueError("pilot requires the validated FP32 backbone")
    base_report_path = args.base_run_report.resolve() if args.base_run_report else None
    base_adapter_dir = None
    base_adapter_hash = None
    if base_report_path is not None:
        base_adapter_dir, base_adapter_hash = base_adapter_from_report(
            base_report_path, head_path, checkpoint, model_dir, args.rank,
            args.adapter_program,
        )
    class_count = manifest.get("class_count", manifest["option_count"])
    train_label_count = len(manifest["outputs"]["D_train"].get("labels", {})) or class_count
    selected = (None if args.all_training_rows else
                select_examples_streaming(train_path, args.seed, args.rows_per_label))
    selected_count = manifest["outputs"]["D_train"]["count"] if selected is None else len(selected)
    if selected is not None and len(selected) != train_label_count * args.rows_per_label:
        raise ValueError("expected equal support from every episode class")
    total_steps = math.ceil(selected_count / args.micro_batch_size) * args.epochs
    if output_dir.exists():
        raise FileExistsError(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True,
    )
    target_modules = ["q_proj", "v_proj"]
    adapter, inheritance = initialize_adapter(
        model.model, args.adapter_program, args.rank, target_modules, base_adapter_dir,
    )
    model.model = adapter
    adapter.train()
    device = next(adapter.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("backbone did not load on the visible GPU")
    head = ChoiceHead(checkpoint["hidden_size"]).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.train()
    adapter_parameters = (
        program_parameters(model.model)
        if args.adapter_program != STANDARD_PROGRAM
        else [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    )
    parameters = list(adapter_parameters)
    parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    # Qwen3.5 FP32 training can exceed 80 GB even for one long row. Keep the
    # logical optimizer batch and step budget fixed, but accumulate gradients
    # from one row at a time for all practical training caps.
    physical_micro_batch_size = (
        1 if args.max_prompt_tokens >= 256 else args.micro_batch_size
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = output_dir / "selected_examples.jsonl"
    first_losses = []
    last_losses = deque(maxlen=10)
    truncated_prompt_count = 0
    step = 0
    training_started = time.time()

    def forward_loss(rows: list[dict], epoch: int, selected_handle,
                     step_number: int) -> torch.Tensor:
        nonlocal truncated_prompt_count
        prompts: list[str] = []
        row_sizes: list[int] = []
        targets: list[int] = []
        for row in rows:
            indices = choose_candidates(row, args.seed, args.negative_count)
            subset = candidate_subset(row, indices)
            option_ids, _ = render_choice_prompts(subset)
            if epoch == 0:
                selected_handle.write(json.dumps({
                    "id": row["id"], "label": row["label"], "candidate_ids": option_ids,
                }, ensure_ascii=False) + "\n")
            for candidate_index in range(len(option_ids)):
                prompt, truncated = fit_choice_prompt(
                    subset, candidate_index, tokenizer, args.max_prompt_tokens,
                )
                prompts.append(prompt)
                truncated_prompt_count += int(truncated)
            row_sizes.append(len(option_ids))
            targets.append(option_ids.index(row["label"]))
        tokens = tokenizer(
            prompts, padding=True, truncation=False,
            return_tensors="pt", add_special_tokens=True,
        )
        longest = int(tokens["attention_mask"].sum(dim=1).max())
        if longest > args.max_prompt_tokens:
            raise ValueError(f"prompt batch exceeds token budget after fitting: {longest}")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = adapter(**tokens, use_cache=False, return_dict=True).last_hidden_state
        scores = head(last_token_pool(hidden, tokens["attention_mask"]))
        losses = []
        cursor = 0
        for size, target in zip(row_sizes, targets, strict=True):
            losses.append(F.cross_entropy(
                scores[cursor:cursor + size].unsqueeze(0),
                torch.tensor([target], device=device),
            ))
            cursor += size
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite training loss at step {step_number}")
        return loss

    def train_micro_batch(batch_rows: list[dict], epoch: int,
                          selected_handle, step_number: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        row_count = len(batch_rows)
        for start in range(0, row_count, physical_micro_batch_size):
            physical_rows = batch_rows[start:start + physical_micro_batch_size]
            loss = forward_loss(physical_rows, epoch, selected_handle, step_number)
            weight = len(physical_rows) / row_count
            (loss * weight).backward()
            weighted_loss += float(loss.detach().cpu()) * len(physical_rows)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        return weighted_loss / row_count

    with selected_path.open("x", encoding="utf-8") as selected_handle:
        for epoch in range(args.epochs):
            epoch_rows = (iter_full_training_rows(train_path, args.seed, epoch)
                          if selected is None else iter(selected))
            epoch_count = 0
            batch_rows = []
            for row in epoch_rows:
                batch_rows.append(row)
                if len(batch_rows) < args.micro_batch_size:
                    continue
                value = train_micro_batch(batch_rows, epoch, selected_handle, step + 1)
                epoch_count += len(batch_rows)
                batch_rows.clear()
                if len(first_losses) < 10:
                    first_losses.append(value)
                last_losses.append(value)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    print(f"step={step}/{total_steps} loss={value:.6f}", flush=True)
            if batch_rows:
                value = train_micro_batch(batch_rows, epoch, selected_handle, step + 1)
                epoch_count += len(batch_rows)
                if len(first_losses) < 10:
                    first_losses.append(value)
                last_losses.append(value)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    print(f"step={step}/{total_steps} loss={value:.6f}", flush=True)
            if epoch_count != selected_count:
                raise ValueError(f"epoch {epoch} read {epoch_count} rows, expected {selected_count}")

    adapter.eval()
    head.eval()
    adapter_path = output_dir / "adapter"
    if args.adapter_program == STANDARD_PROGRAM:
        adapter.save_pretrained(adapter_path, safe_serialization=True)
    else:
        save_program_adapter(model.model, adapter_path, args.adapter_program, args.rank, target_modules)
    adapted_head_path = output_dir / "choice_head.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "hidden_size": head.hidden_size,
        "temperature": checkpoint["temperature"],
        "model_config_sha256": checkpoint["model_config_sha256"],
        "backbone_dtype": "fp32", "padding_side": "right",
        "base_head_checkpoint_sha256": sha256(head_path),
        "base_run_report_sha256": sha256(base_report_path) if base_report_path else None,
        "base_adapter_model_sha256": base_adapter_hash,
    }, adapted_head_path)
    report = {
        "status": "full_train_complete" if full_episode and args.all_training_rows else "pilot_train_complete",
        "scope": "visible sampled-negative Choice-LoRA adaptation; no hidden evaluation",
        "model_dir": str(model_dir),
        "model_config_sha256": sha256(model_dir / "config.json"),
        "base_head_checkpoint": str(head_path),
        "base_head_checkpoint_sha256": sha256(head_path),
        "base_run_report_sha256": sha256(base_report_path) if base_report_path else None,
        "base_adapter_model_sha256": base_adapter_hash,
        "episode_manifest_sha256": sha256(manifest_path),
        "adapter_inheritance": inheritance,
        "episode": manifest["episode"],
        "class_count": class_count,
        "train_label_count": train_label_count,
        "D_train_sha256": sha256(train_path),
        "selected_examples_sha256": sha256(selected_path),
        "seed": args.seed, "rows_per_label": None if args.all_training_rows else args.rows_per_label,
        "all_training_rows": args.all_training_rows,
        "D_train_count": manifest["outputs"]["D_train"]["count"],
        "trained_examples": selected_count,
        "training_pool_coverage": selected_count / manifest["outputs"]["D_train"]["count"],
        "selection_policy": "all_rows_chunk_shuffle_4096" if args.all_training_rows else "per_label_lowest_sha256",
        "negative_count": args.negative_count,
        "candidate_forwards": selected_count * args.epochs * (args.negative_count + 1),
        "optimizer_steps": step,
        "rank": args.rank, "adapter_program": args.adapter_program,
        "adapter_program_metadata": program_metadata(args.adapter_program, args.rank, target_modules),
        "adapter_program_fingerprint": hashlib.sha256(json.dumps(
            program_metadata(args.adapter_program, args.rank, target_modules),
            sort_keys=True,
        ).encode()).hexdigest(),
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "physical_micro_batch_size": physical_micro_batch_size,
        "gradient_accumulation": args.micro_batch_size // physical_micro_batch_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "protocol_max_prompt_tokens": protocol_max_prompt_tokens,
        "truncated_prompt_count": truncated_prompt_count,
        "backbone_dtype": "fp32", "target_modules": target_modules,
        "adapter_trainable_parameter_count": sum(p.numel() for p in adapter_parameters),
        "head_trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "first_ten_loss_mean": sum(first_losses) / len(first_losses),
        "last_ten_loss_mean": sum(last_losses) / len(last_losses),
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": sha256(adapter_path / "adapter_model.safetensors"),
        "head_path": str(adapted_head_path),
        "head_sha256": sha256(adapted_head_path),
        "training_elapsed_seconds": time.time() - training_started,
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
