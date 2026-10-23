"""Explicit execution-only changes when continuing an existing training state."""
from __future__ import annotations

from collections import Counter
import hashlib
import json


RUNTIME_FIELDS = {
    "name", "physical_rows_per_gpu", "gradient_checkpointing", "checkpoint_layer_stride",
    "cpu_threads_per_worker", "log_every_steps", "checkpoint_every_steps",
    "microbatch_token_budget",
}


def validate_runtime_migration(original, current, *, allow_worker_resize=False):
    if original["plan_sha256"] != current["plan_sha256"]:
        raise ValueError("runtime migration cannot change plan_sha256")
    resized = original["world_size"] != current["world_size"]
    if resized and not allow_worker_resize:
        raise ValueError("runtime migration cannot change world_size without explicit worker resize")
    world = current["world_size"]
    batch = current["config"]["global_batch_size"]
    physical = current["config"]["physical_rows_per_gpu"]
    if not isinstance(world, int) or world < 1 or batch % world or batch // world % physical:
        raise ValueError("worker resize must preserve complete global and physical batches")
    before, after = original["config"], current["config"]
    changes = {k: {"before": before.get(k), "after": after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    forbidden = changes.keys() - RUNTIME_FIELDS
    if forbidden:
        raise ValueError(f"runtime migration cannot change training settings: {sorted(forbidden)}")
    code_changes = {k: {"before": original["code"].get(k), "after": current["code"].get(k)}
                    for k in original["code"].keys() | current["code"].keys()
                    if original["code"].get(k) != current["code"].get(k)}
    allowed_code = {"scripts/train.py", "src/deciform/runtime_settings.py"}
    if code_changes.keys() - allowed_code:
        raise ValueError("runtime migration changed model, adapter, data or loss source code")
    return {"config_changes": changes, "code_changes": code_changes,
            "optimizer_restored": True, "data_cursor_restored": True,
            "worker_resize": ({"before": original["world_size"], "after": world,
                               "rng_policy": "deterministic_reseed_all_workers",
                               "coverage_policy": "global_prefix_plus_new_worker_counts"}
                              if resized else None),
            "bitwise_replay": False}


def worker_resize_state(saved, workers, coverage_prefix, world, seed):
    """Carry historical coverage once, without inventing new per-worker histories.

    The source checkpoint retains old worker RNG/trace states. New workers start
    new traces and deterministic RNG streams; their counters cover only work
    after the resize. Global progress adds the immutable prefix exactly once.
    """
    if not isinstance(world, int) or world < 1:
        raise ValueError("new world_size must be a positive integer")
    if len(workers) != saved["world_size"]:
        raise ValueError("checkpoint worker population differs from recorded world_size")
    counts = Counter(coverage_prefix)
    for worker in workers:
        counts.update(worker["counts"])
    expected = saved["source_rows_seen"]
    if (any(v < 0 for v in counts.values()) or counts.keys() - expected.keys()
            or {source: counts[source] for source in expected} != expected):
        raise ValueError("checkpoint worker coverage differs from global source counts")
    if sum(counts.values()) != saved["sample_visits"]:
        raise ValueError("checkpoint sample visits differ from coverage")
    identity = json.dumps({"seed": seed, "step": saved["optimizer_steps"],
                           "epoch": saved["epoch"], "cursor": saved["cursor"],
                           "plan_sha256": saved["plan_sha256"], "world_size": world,
                           "worker_traces": [worker["trace"] for worker in workers]}, sort_keys=True)
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    seeds = [int(hashlib.sha256(f"{fingerprint}:{rank}".encode()).hexdigest()[:15], 16)
             for rank in range(world)]
    traces = [hashlib.sha256(f"resize:{fingerprint}:{rank}".encode()).hexdigest()
              for rank in range(world)]
    return {"coverage_prefix": {source: counts[source] for source in expected}, "worker_seeds": seeds,
            "worker_traces": traces, "migration_fingerprint": fingerprint}


def configure_checkpoint_layers(model, stride: int):
    from transformers.modeling_layers import GradientCheckpointingLayer
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("checkpoint_layer_stride must be a positive integer")
    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    if not layers:
        raise ValueError("no supported checkpointing layers found")
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index % stride == 0
    return {"checkpoint_layer_stride": stride, "total_layers": len(layers),
            "checkpointed_layers": sum(m.gradient_checkpointing for m in layers)}
