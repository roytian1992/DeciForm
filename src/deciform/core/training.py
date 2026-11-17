"""Sample-weighted synchronous training; no hidden parent checkpoint or sampling."""

import copy
import hashlib
import json
import math
import os
import random
import time
import sys
import platform
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import read_jsonl, sha256, atomic_json
from .model import DecisionModel
from .objectives import request_loss
from .checkpoint import (
    trainable_state,
    restore,
    save_export,
    load_export,
    rng_state,
    set_rng,
    atomic_torch_save,
)


def synchronize(parameters, world):
    if world == 1:
        return
    flags = torch.tensor(
        [p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    active = []
    for p, flag in zip(parameters, flags.tolist()):
        if flag:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            active.append(p)
    # Bounded flat buckets avoid one collective per parameter and excess memory.
    bucket = []
    size = 0

    def reduce(ps):
        flat = torch.cat([p.grad.reshape(-1) for p in ps])
        dist.all_reduce(flat)
        offset = 0
        for p in ps:
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p))
            offset += n

    for p in active:
        if bucket and (size + p.numel() > 8_000_000 or bucket[0].dtype != p.dtype):
            reduce(bucket)
            bucket = []
            size = 0
        bucket.append(p)
        size += p.numel()
    if bucket:
        reduce(bucket)


def train(model_config, train_config, data_path, output, resume=False, initialize_from=None):
    if resume and initialize_from:
        raise ValueError("resume and initialize_from are mutually exclusive")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available() and train_config.get("device", "cuda") != "cpu"
    device = torch.device("cuda", local) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    output = Path(output)
    seed = int(train_config.get("seed", model_config.get("seed", 20261101)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = list(read_jsonl(data_path, True))
    epochs = int(train_config.get("epochs", 1))
    batch = int(train_config.get("global_batch", 64))
    micro = int(train_config.get("micro_batch", 1))
    if int(train_config.get("save_every", 100)) < 1:
        raise ValueError("save_every must be positive")
    if not rows or epochs < 1 or batch < 1 or micro < 1:
        raise ValueError("Nonempty data and positive epoch/batch values required")
    steps_per_epoch = math.ceil(len(rows) / batch)
    total_steps = epochs * steps_per_epoch
    code_hashes = {
        str(p.relative_to(Path(__file__).parent)): sha256(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))
    }
    identity = {
        "code_sha256": code_hashes,
        "model": model_config,
        "training": train_config,
        "data_sha256": sha256(data_path),
        "rows": len(rows),
        "world_size": world,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (output / "STOP").exists():
        raise ValueError("Remove STOP explicitly before starting or resuming")
    if resume:
        previous_run = json.loads((output / "run.json").read_text())
        if previous_run["fingerprint"] != fingerprint:
            raise ValueError("Exact resume requires unchanged code/data/config/world size")
        if not (output / "resume.pt").is_file():
            raise FileNotFoundError("Missing resume.pt")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh training requires an empty output directory")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        import transformers

        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        }
        if resume:
            atomic_json(
                output / "resume_event.json",
                {
                    "command": sys.argv,
                    "environment": environment,
                    "fingerprint": fingerprint,
                    "updated_unix": time.time(),
                },
            )
        else:
            atomic_json(
                output / "run.json",
                {
                    **identity,
                    "environment": environment,
                    "command": sys.argv,
                    "fingerprint": fingerprint,
                    "total_steps": total_steps,
                    "initialized_from": str(initialize_from) if initialize_from else None,
                    "parent_weights_sha256": (
                        sha256(Path(initialize_from) / "trainables.safetensors")
                        if initialize_from
                        else None
                    ),
                },
            )
    if world > 1:
        dist.barrier()
    model = DecisionModel(model_config, device=device)
    if initialize_from:
        from safetensors.torch import load_file

        previous = json.loads((Path(initialize_from) / "config.json").read_text())
        if previous != model_config:
            raise ValueError("Continuation requires matching model configuration")
        restore(model, load_file(str(Path(initialize_from) / "trainables.safetensors")))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in parameters:
            dist.broadcast(p.data, src=0)
    groups = model.optimizer_groups()
    optimizer = torch.optim.AdamW(groups, weight_decay=train_config.get("weight_decay", 0.01))
    warmup = float(train_config.get("warmup_fraction", 0.3))
    if not 0 <= warmup < 1:
        raise ValueError("warmup_fraction must be in [0,1)")
    # OneCycle mirrors research defaults; very short smoke jobs use explicit constant LR.
    schedule = train_config.get("schedule", "onecycle")
    if schedule == "onecycle":
        if total_steps < 4:
            raise ValueError("Use schedule=constant for smoke runs with fewer than4 steps")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in groups], total_steps=total_steps, pct_start=warmup
        )
    elif schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError("Unknown schedule")
    start_step = 0
    elapsed = 0.0
    sample_visits = 0
    if resume:
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Exact resume requires unchanged code/data/config/world size")
        restore(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        sample_visits = state["sample_visits"]
        elapsed = state["train_seconds"]
        set_rng(state["rng"][rank])
    else:
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
    model.train()
    started = time.monotonic()

    def save(step):
        local_rng = rng_state()
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, local_rng)
        else:
            states = [local_rng]
        if rank == 0:
            atomic_torch_save(
                {
                    "fingerprint": fingerprint,
                    "step": step,
                    "sample_visits": sample_visits,
                    "model": trainable_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": states,
                    "train_seconds": elapsed + time.monotonic() - started,
                },
                output / "resume.pt",
            )
        if world > 1:
            dist.barrier()

    try:
        order = None
        last_epoch = None
        for step in range(start_step, total_steps):
            epoch, within = divmod(step, steps_per_epoch)
            if epoch != last_epoch:
                order = np.random.default_rng(seed + epoch).permutation(len(rows)).tolist()
                last_epoch = epoch
            selected = order[within * batch : (within + 1) * batch]
            local_indices = selected[rank::world]
            weights = [float(rows[i].get("weight", 1.0)) for i in selected]
            if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError("Invalid global sample weights")
            denominator = math.fsum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for begin in range(0, len(local_indices), micro):
                records = [rows[i] for i in local_indices[begin : begin + micro]]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=cuda and model_config.get("bf16", True),
                ):
                    losses = model.losses(records)
                    w = losses.new_tensor([float(r.get("weight", 1.0)) for r in records])
                    loss = (losses * w).sum() / denominator
                loss.backward()
                loss_sum += float(loss.detach())
            synchronize(parameters, world)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, train_config.get("max_grad_norm", 1.0), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            sample_visits += len(selected)
            values = torch.tensor([loss_sum], device=device)
            if world > 1:
                dist.all_reduce(values)
            if rank == 0:
                atomic_json(
                    output / "progress.json",
                    {
                        "phase": "training",
                        "step": step + 1,
                        "total_steps": total_steps,
                        "epoch": epoch + 1,
                        "sample_visits": sample_visits,
                        "loss": float(values[0]),
                        "gradient_norm": float(norm),
                        "train_seconds": elapsed + time.monotonic() - started,
                        "updated_unix": time.time(),
                    },
                )
                print(
                    json.dumps(
                        {"step": step + 1, "total_steps": total_steps, "loss": float(values[0])}
                    ),
                    flush=True,
                )
            stop = torch.tensor([int((output / "STOP").exists())], device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if (
                (step + 1) % train_config.get("save_every", 100) == 0
                or bool(stop.item())
                or step + 1 == total_steps
            ):
                save(step + 1)
            if stop.item():
                if rank == 0:
                    atomic_json(output / "status.json", {"phase": "paused", "step": step + 1})
                return
        if rank == 0:
            save_export(model, output / "checkpoint")
            atomic_json(
                output / "status.json",
                {
                    "phase": "complete",
                    "step": total_steps,
                    "sample_visits": sample_visits,
                    "train_seconds": elapsed + time.monotonic() - started,
                },
            )
    except BaseException as e:
        if rank == 0:
            atomic_json(output / "failure.json", {"error": str(e), "updated_unix": time.time()})
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()
