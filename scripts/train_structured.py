"""Resumable distributed joint training over every registered P_train row.

Run with torchrun. Static and structural candidates share this trainer,
permutation, loss, fixed rank, optimizer settings and checkpoint format.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from deciform.adapter_program import STANDARD_PROGRAM, program_metadata, save_program_adapter
from deciform.adapter_transfer import initialize_adapter
from deciform.latent_program import LATENT_PROGRAM, initialize_latent, latent_metadata, save_latent
from deciform.joint_data import JointRows, atomic_json, file_sha256, shard_batch
from deciform.runtime_settings import (configure_checkpoint_layers, validate_runtime_migration,
                                      worker_resize_state)
from deciform.choice import ChoiceHead, last_token_pool, render_choice_prompts
from deciform.sampling import candidate_subset, choose_candidates


class JointChoiceModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask, sizes, targets, candidate_stride=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        scores = self.head(last_token_pool(hidden, attention_mask))
        losses = []
        cursor = 0
        for size, target in zip(sizes, targets, strict=True):
            losses.append(F.cross_entropy(scores[cursor:cursor + size][None],
                                           scores.new_tensor([target], dtype=torch.long)))
            cursor += candidate_stride or size
        return torch.stack(losses)


def code_manifest():
    paths = [Path(__file__), *sorted((PROJECT / "src/deciform").glob("*.py"))]
    return {str(p.relative_to(WORKSPACE)): file_sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="checkpoint directory including optimizer/RNG/cursor")
    parser.add_argument("--migrate-runtime", action="store_true",
                        help="restore --resume into a new output with recorded execution-only changes")
    parser.add_argument("--resize-workers", action="store_true",
                        help="with --migrate-runtime, change DDP workers while preserving the global batch and optimizer")
    parser.add_argument("--parent-checkpoint", type=Path, help="branch from parent weights and data cursor; reset optimizer")
    parser.add_argument("--start-from-completed", type=Path,
                        help="start a new registered data stage from completed parent weights; reset optimizer/cursor")
    parser.add_argument("--stop-after-steps", type=int, help="explicit absolute step boundary; saves a paused checkpoint")
    args = parser.parse_args()
    if sum(value is not None for value in
           (args.resume, args.parent_checkpoint, args.start_from_completed)) > 1:
        raise ValueError("resume, structural branching and new data stages are separate operations")
    if args.migrate_runtime and not args.resume:
        raise ValueError("--migrate-runtime requires --resume")
    if args.resize_workers and not args.migrate_runtime:
        raise ValueError("--resize-workers requires --migrate-runtime and a new output")
    config = json.loads(args.config.read_text())
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(config.get("cpu_threads_per_worker", 4))
    torch.set_float32_matmul_precision(config.get("matmul_precision", "highest"))
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    output = args.output_dir.resolve()
    plan_path = (WORKSPACE / config["joint_plan"]).resolve()
    plan = json.loads(plan_path.read_text())
    if plan["scope"] not in ("all_P_train_joint", "registered_continuation") or plan["status"] != "complete":
        raise ValueError("expected a verified joint training plan")
    if plan["scope"] == "registered_continuation" and not (args.start_from_completed or args.resume):
        raise ValueError("registered continuation requires --start-from-completed or --resume")
    if plan["scope"] != "registered_continuation" and args.start_from_completed:
        raise ValueError("--start-from-completed requires a registered continuation plan")
    validation_path = Path(plan["validation_plan"])
    validation = json.loads(validation_path.read_text())
    if file_sha256(validation_path) != plan["validation_plan_sha256"]:
        raise ValueError("validation plan changed")
    if sorted(s["name"] for s in plan["sources"]) != sorted(validation["P_train"]):
        raise ValueError("joint run must include every P_train source exactly once")
    if plan["total_rows"] != config["expected_train_rows"]:
        raise ValueError("unexpected joint training population")
    verification = [None]
    if rank == 0:
        try:
            for source in plan["sources"]:
                if file_sha256(Path(source["path"])) != source["sha256"]:
                    raise ValueError(f"training source changed: {source['name']}")
            verification[0] = "ok"
        except Exception as exc:
            verification[0] = str(exc)
    if world > 1:
        dist.broadcast_object_list(verification, src=0)
    if verification[0] != "ok":
        raise ValueError(verification[0])
    batch = config["global_batch_size"]
    physical = config["physical_rows_per_gpu"]
    if physical < 1:
        raise ValueError("physical_rows_per_gpu must be positive")
    if batch % world or batch // world % physical:
        raise ValueError("batch must divide by workers and physical microbatch")
    metadata = {"config": config, "plan_sha256": file_sha256(plan_path),
                "code": code_manifest(), "world_size": world,
                "python": sys.executable, "torch": torch.__version__,
                "cuda": torch.version.cuda, "command": sys.argv,
                "fla_core": __import__("importlib.metadata", fromlist=["version"]).version("fla-core")
                    if config.get("kernel_path") else None,
                "scope": plan["scope"],
                "parent_checkpoint": str((args.parent_checkpoint or args.start_from_completed).resolve())
                    if (args.parent_checkpoint or args.start_from_completed) else None}
    if args.migrate_runtime:
        original = json.loads((args.resume.resolve().parent.parent / "run_manifest.json").read_text())
        metadata["runtime_migration"] = {"checkpoint": str(args.resume.resolve()),
            **validate_runtime_migration(original, metadata, allow_worker_resize=args.resize_workers)}
    elif args.resume:
        original = json.loads((output / "run_manifest.json").read_text())
        for key in ("config", "plan_sha256", "code", "world_size"):
            if original[key] != metadata[key]:
                raise ValueError(f"resume {key} changed; create an explicit new branch")
    if rank == 0 and (not args.resume or args.migrate_runtime):
        output.mkdir(parents=True, exist_ok=False)
        (output / "code").mkdir()
        for relative in metadata["code"]:
            destination = output / "code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((WORKSPACE / relative).read_bytes())
        atomic_json(output / "run_manifest.json", metadata)
        atomic_json(output / "input_plan.json", plan)
    if world > 1:
        dist.barrier()
    theta = json.loads((WORKSPACE / config["theta0_config"]).read_text())
    checkpoint_dir = (args.resume or args.parent_checkpoint or args.start_from_completed)
    if checkpoint_dir:
        checkpoint_dir = checkpoint_dir.resolve()
        for relative, expected in json.loads((checkpoint_dir / "hashes.json").read_text()).items():
            if file_sha256(checkpoint_dir / relative) != expected:
                raise ValueError(f"checkpoint hash mismatch: {relative}")
        parent_progress = json.loads((checkpoint_dir / "progress.json").read_text())
        if args.start_from_completed:
            if parent_progress["status"] != "complete" or parent_progress["sample_visits"] != parent_progress["expected_sample_visits"]:
                raise ValueError("data-stage parent must be a complete checkpoint")
            if parent_progress["plan_sha256"] != plan["parent_plan_sha256"]:
                raise ValueError("data-stage parent plan differs")
            if str(checkpoint_dir) != plan["parent_checkpoint"]:
                raise ValueError("data-stage parent checkpoint differs")
            saved = None
        else:
            saved = parent_progress
            if saved["plan_sha256"] != metadata["plan_sha256"]:
                raise ValueError("parent/resume pool differs")
            if saved["world_size"] != world and not args.resize_workers:
                raise ValueError("parent/resume pool or worker count differs")
        if args.parent_checkpoint or args.start_from_completed:
            parent_manifest = json.loads((checkpoint_dir.parent.parent / "run_manifest.json").read_text())
            if args.start_from_completed and parent_manifest["plan_sha256"] != plan["parent_plan_sha256"]:
                raise ValueError("data-stage parent run manifest differs")
            for key in ("seed", "global_batch_size", "physical_rows_per_gpu", "learning_rate", "rank",
                        "negative_count", "epochs", "matmul_precision", "gradient_checkpointing",
                        "backbone_dtype", "optimizer_reset_every_steps", "padding_length",
                        "padding_multiple", "pad_candidate_slots", "sort_microbatches_by_length", "kernel_path"):
                if parent_manifest["config"].get(key) != config.get(key):
                    raise ValueError(f"structural comparison must hold {key} fixed")
        head_path = checkpoint_dir / "choice_head.pt"
        adapter_dir = checkpoint_dir / "adapter"
    else:
        head_path = WORKSPACE / theta["head_checkpoint"]
        adapter_dir = WORKSPACE / theta["adapter_dir"]
        for path, expected in [(head_path, theta["head_checkpoint_sha256"]),
                               (adapter_dir / "adapter_model.safetensors", theta["adapter_model_sha256"])]:
            if file_sha256(path) != expected:
                raise ValueError("initial checkpoint hash mismatch")
        saved = None
    head_state = torch.load(head_path, map_location="cpu", weights_only=True)
    model_dir = WORKSPACE / config["model_dir"]
    if head_state["model_config_sha256"] != file_sha256(model_dir / "config.json"):
        raise ValueError("model and Choice head mismatch")
    if head_state["backbone_dtype"] != "fp32":
        raise ValueError("joint FP32 trainer requires an FP32 head checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[config.get("backbone_dtype", "fp32")]
    outer = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, dtype=dtype,
        device_map={"": local_rank}, low_cpu_mem_usage=True,
    )
    outer.requires_grad_(False)
    if config["adapter_program"] == LATENT_PROGRAM:
        backbone, inheritance = initialize_latent(outer.model, config["adapter_genome"], config["rank"],
                                                  config["target_modules"], adapter_dir)
    else:
        backbone, inheritance = initialize_adapter(outer.model, config["adapter_program"], config["rank"],
                                                   config["target_modules"], adapter_dir)
    del outer
    if config["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        checkpoint_policy = configure_checkpoint_layers(backbone, config.get("checkpoint_layer_stride", 1))
        if rank == 0:
            atomic_json(output / "checkpoint_policy.json", checkpoint_policy)
    head = ChoiceHead(head_state["hidden_size"]).to(device)
    head.load_state_dict(head_state["state_dict"])
    model = JointChoiceModel(backbone, head)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
    wrapped = (DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
               if world > 1 else model)
    data = JointRows(plan)
    sources = [s["name"] for s in plan["sources"]]
    source_caps = {s["name"]: s["max_prompt_tokens"] for s in plan["sources"]}
    local_counts = Counter()
    coverage_prefix = Counter()
    trace = "0" * 64
    step = 0
    epoch = 0
    cursor = 0
    if saved:
        step, epoch, cursor = saved["optimizer_steps"], saved["epoch"], saved["cursor"]
        # Structural branches inherit data coverage, but use a fresh optimizer.
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        if saved["world_size"] != world:
            resized = worker_resize_state(saved, state["workers"], state.get("coverage_prefix", {}),
                                          world, seed)
            coverage_prefix.update(resized["coverage_prefix"])
            trace = resized["worker_traces"][rank]
            torch.manual_seed(resized["worker_seeds"][rank])
            torch.cuda.manual_seed(resized["worker_seeds"][rank])
            random.seed(resized["worker_seeds"][rank])
            if rank == 0:
                atomic_json(output / "worker_resize.json", {
                    "source_checkpoint": str(checkpoint_dir), "from_world_size": saved["world_size"],
                    "to_world_size": world, "optimizer_steps": step, "epoch": epoch, "cursor": cursor,
                    "global_batch_size": batch, "optimizer_restored": True,
                    "rng_policy": "deterministic_reseed_all_workers", "bitwise_replay": False, **resized})
        else:
            coverage_prefix.update(state.get("coverage_prefix", {}))
            local_counts.update(state["workers"][rank]["counts"])
            trace = state["workers"][rank]["trace"]
            torch.set_rng_state(state["workers"][rank]["torch_rng"])
            torch.cuda.set_rng_state(state["workers"][rank]["cuda_rng"], device)
            random.setstate(state["workers"][rank]["python_rng"])
        if args.resume:
            optimizer.load_state_dict(state["optimizer"])
        del state
    started = time.monotonic()
    initial_seen = epoch * data.total + cursor
    initial_step = step
    log_path = output / "metrics.jsonl"
    previous_checkpoint = str(checkpoint_dir) if checkpoint_dir else str(adapter_dir)
    last_saved_step = step if args.resume else -1
    last_loss = None
    total_steps = math.ceil(data.total / batch) * config["epochs"]

    def progress(status):
        counts = torch.tensor([local_counts[s] for s in sources], dtype=torch.long, device=device)
        if world > 1:
            dist.all_reduce(counts)
        counts = {source: value + coverage_prefix[source]
                  for source, value in zip(sources, counts.cpu().tolist(), strict=True)}
        seen = sum(counts.values())
        elapsed = time.monotonic() - started
        speed = (seen - initial_seen) / elapsed if elapsed else 0.0
        return {"status": status, "scope": metadata["scope"], "optimizer_steps": step,
                "expected_optimizer_steps": total_steps, "epoch": epoch, "cursor": cursor,
                "source_rows_seen": counts, "sample_visits": seen,
                "expected_sample_visits": data.total * config["epochs"],
                "training_coverage": seen / (data.total * config["epochs"]),
                "source_coverage": {s["name"]: counts[s["name"]] / (s["count"] * config["epochs"])
                                    for s in plan["sources"]},
                "global_rows_per_second": speed,
                "estimated_remaining_hours": (data.total * config["epochs"] - seen) / speed / 3600 if speed else None,
                "session_elapsed_seconds": elapsed, "last_loss": last_loss,
                "world_size": world, "plan_sha256": metadata["plan_sha256"],
                "adapter_program": config["adapter_program"], "adapter_inheritance": inheritance,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "peak_memory_gib_local": torch.cuda.max_memory_allocated() / 2**30}

    def checkpoint(status):
        nonlocal previous_checkpoint, last_saved_step
        report = progress(status)
        if step == last_saved_step:
            if rank == 0:
                report["checkpoint"] = previous_checkpoint
                atomic_json(output / "progress.json", report)
            return
        worker_state = {"counts": dict(local_counts), "trace": trace,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                        "python_rng": random.getstate()}
        workers = [None] * world
        if world > 1:
            dist.all_gather_object(workers, worker_state)
        else:
            workers[0] = worker_state
        if rank == 0:
            target = output / "checkpoints" / f"step_{step:08d}"
            temporary = target.with_name(target.name + ".partial")
            temporary.mkdir(parents=True, exist_ok=False)
            if config["adapter_program"] == STANDARD_PROGRAM:
                backbone.save_pretrained(temporary / "adapter", safe_serialization=True)
            elif config["adapter_program"] == LATENT_PROGRAM:
                save_latent(backbone, temporary / "adapter", config["adapter_genome"],
                            config["rank"], config["target_modules"])
            else:
                save_program_adapter(backbone, temporary / "adapter", config["adapter_program"],
                                     config["rank"], config["target_modules"])
            torch.save({**head_state, "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "padding_side": "right", "backbone_dtype": "fp32",
                        "training_backbone_dtype": config.get("backbone_dtype", "fp32")}, temporary / "choice_head.pt")
            torch.save({"optimizer": optimizer.state_dict(), "workers": workers,
                        "coverage_prefix": dict(coverage_prefix)}, temporary / "training_state.pt")
            atomic_json(temporary / "progress.json", report)
            atomic_json(temporary / "hashes.json", {str(p.relative_to(temporary)): file_sha256(p)
                        for p in temporary.rglob("*") if p.is_file()})
            temporary.rename(target)
            report["checkpoint"] = str(target)
            atomic_json(output / "latest.json", report)
            atomic_json(output / "progress.json", report)
            if status == "complete":
                atomic_json(output / "report.json", report)
            with (output / "lineage.jsonl").open("a") as handle:
                handle.write(json.dumps({"event": "checkpoint", "step": step,
                    "parent_checkpoint": previous_checkpoint,
                    "checkpoint": str(target), "adapter_program": config["adapter_program"],
                    "data_cursor": {"epoch": epoch, "cursor": cursor}, "sample_visits": report["sample_visits"],
                    "inheritance": inheritance, "status": status}) + "\n")
            print(json.dumps(report), flush=True)
        previous_checkpoint = str(output / "checkpoints" / f"step_{step:08d}")
        last_saved_step = step
        if world > 1:
            dist.barrier()

    if rank == 0:
        atomic_json(output / "initialization.json", {"inheritance": inheritance,
            "adapter_metadata": (latent_metadata(config["adapter_genome"], config["rank"], config["target_modules"])
                                 if config["adapter_program"] == LATENT_PROGRAM else
                                 program_metadata(config["adapter_program"], config["rank"], config["target_modules"])),
            "head_sha256": file_sha256(head_path),
            "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors")})
    try:
        while epoch < config["epochs"]:
            order = data.permutation(seed, epoch)
            while cursor < data.total:
                if args.stop_after_steps is not None and step >= args.stop_after_steps:
                    if step > initial_step:
                        checkpoint("paused_at_explicit_step_boundary")
                    return
                indices = shard_batch(order, cursor, batch, rank, world)
                reset_interval = config.get("optimizer_reset_every_steps", 0)
                if reset_interval and step > 0 and step % reset_interval == 0:
                    optimizer.state.clear()
                real_global_rows = min(batch, data.total - cursor)
                optimizer.zero_grad(set_to_none=True)
                local_loss = 0.0
                prepared = []
                for index in indices:
                    source, row = data.get(index if index is not None else int(order[0]))
                    neg = min(config["negative_count"], len(row["question"]["options"]) - 1)
                    subset = candidate_subset(row, choose_candidates(row, seed, neg))
                    ids, rendered = render_choice_prompts(subset)
                    lengths = [len(x) for x in tokenizer(rendered, add_special_tokens=True)["input_ids"]]
                    if source_caps[source] is not None and max(lengths) > source_caps[source]:
                        raise ValueError(f"{source}/{row['id']}: prompt exceeds the pinned protocol cap")
                    prepared.append((max(lengths), index, source, ids, rendered, ids.index(row["label"])))
                if config.get("sort_microbatches_by_length", False):
                    prepared.sort(key=lambda item: item[0])
                token_budget = config.get("microbatch_token_budget")
                start = 0
                while start < len(indices):
                    if token_budget is None:
                        micro_physical = physical
                    else:
                        # Rows are sorted by length. Pack each microbatch
                        # against its own longest row so one unusually long
                        # example does not force all shorter rows in this
                        # global batch down to microbatch size one.
                        micro_physical = 1
                        for candidate_size in range(min(physical, len(indices) - start), 0, -1):
                            candidate = prepared[start:start + candidate_size]
                            candidate_length = max(item[0] for item in candidate)
                            candidate_count = max(len(item[3]) for item in candidate)
                            if candidate_size * candidate_length * candidate_count <= token_budget:
                                micro_physical = candidate_size
                                break
                    prompts, sizes, targets, weights = [], [], [], []
                    # Respect the token-budget-derived microbatch size. Using
                    # ``physical`` here defeats the long-row memory guard and
                    # can turn a batch-1 decision back into a full physical
                    # batch when a single prompt is very long.
                    chunk = prepared[start:start + micro_physical]
                    for _, index, source, ids, rendered, target in chunk:
                        prompts.extend(rendered)
                        if config.get("pad_candidate_slots", False):
                            # Extra rows are batch padding only. Their logits
                            # are excluded from CE and from data coverage.
                            prompts.extend([rendered[0]] * (config["negative_count"] + 1 - len(ids)))
                        sizes.append(len(ids))
                        targets.append(target)
                        weights.append(float(index is not None))
                        if index is not None:
                            local_counts[source] += 1
                            trace = hashlib.sha256(f"{trace}:{epoch}:{index}".encode()).hexdigest()
                    fixed_padding = config.get("padding_length")
                    tokens = tokenizer(prompts, padding="max_length" if fixed_padding else True,
                                       max_length=fixed_padding, pad_to_multiple_of=config.get("padding_multiple"),
                                       truncation=False, return_tensors="pt")
                    tokens = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
                    sync = start + len(chunk) == len(indices)
                    context = nullcontext() if sync or world == 1 else wrapped.no_sync()
                    with context:
                        losses = wrapped(**tokens, sizes=sizes, targets=targets,
                            candidate_stride=config["negative_count"] + 1 if config.get("pad_candidate_slots") else None)
                        loss = (losses * losses.new_tensor(weights)).sum()
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f"nonfinite loss at step {step}, rank {rank}")
                        (loss * world / real_global_rows).backward()
                    local_loss += float(loss.detach())
                    start += len(chunk)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_tensor = torch.tensor(local_loss, device=device)
                if world > 1:
                    dist.all_reduce(loss_tensor)
                last_loss = float(loss_tensor) / real_global_rows
                step += 1
                cursor += real_global_rows
                if step % config["log_every_steps"] == 0 or step == initial_step + 1:
                    report = progress("running")
                    if rank == 0:
                        atomic_json(output / "progress.json", report)
                        with log_path.open("a") as handle:
                            handle.write(json.dumps(report) + "\n")
                        print(json.dumps(report), flush=True)
                if cursor < data.total and step % config["checkpoint_every_steps"] == 0:
                    checkpoint("running")
            epoch += 1
            cursor = 0
        final = progress("complete")
        if any(final["source_rows_seen"][s["name"]] != s["count"] * config["epochs"] for s in plan["sources"]):
            raise RuntimeError("full pass ended without exact per-source coverage")
        checkpoint("complete")
    except Exception as exc:
        # Do not enter a collective destroy while peers may be waiting inside
        # backward/all_reduce. Let torchrun terminate the remaining workers.
        traceback.print_exc()
        atomic_json(output / f"failure_rank{rank}.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "optimizer_steps_completed": step, "epoch": epoch, "cursor": cursor,
            "last_checkpoint": previous_checkpoint,
        })
        sys.stderr.flush()
        if world > 1:
            os._exit(1)
        raise
    finally:
        data.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
