"""Evaluate a joint checkpoint on the fixed per-source D_mv feedback panels."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoTokenizer
from deciform.adapter_transfer import initialize_adapter
from deciform.adapter_program import STANDARD_PROGRAM
from deciform.latent_program import LATENT_PROGRAM, initialize_latent
from deciform.joint_data import atomic_json, file_sha256
from deciform.evaluation_batching import assigned_prompt_indices, restore_prompt_order
from deciform.development_validation import summarize
from deciform.reporting import population_counts
from deciform.choice import ChoiceHead, choice_response, encode_prompts, render_choice_prompts

PERMUTATION_WARNING_TOLERANCE = 1e-3
PERMUTATION_FAILURE_TOLERANCE = 5e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--joint-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta0", action="store_true", help="checkpoint is a theta0 lock JSON")
    parser.add_argument("--role", choices=["D_mv", "D_assess_visible"], default="D_mv")
    parser.add_argument("--development-plan", type=Path,
                        help="add separately reported natural and balanced development panels")
    parser.add_argument("--skip-joint-assess", action="store_true",
                        help="evaluate only supplied development panels, without loading episode assessment roles")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() != world:
        raise ValueError("evaluation requires one process per visible GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    plan = json.loads(args.joint_plan.read_text())
    evaluation_sources = []
    for source in ([] if args.skip_joint_assess else plan["sources"]):
        episode = Path(source["episode_manifest"])
        manifest = json.loads(episode.read_text())
        if file_sha256(episode) != source["episode_manifest_sha256"]:
            raise ValueError("episode manifest changed")
        evaluation_sources.append({**source, "source_family": source["name"],
            "path": str(episode.parent / f"{args.role}.jsonl"),
            "sha256": manifest["outputs"][args.role]["sha256"],
            "count": manifest["outputs"][args.role]["count"], "panel": args.role})
    if args.development_plan:
        development = json.loads(args.development_plan.read_text())
        if (development["status"] != "complete" or development["purpose"] != "development_monitoring_only"
                or development["joint_plan_sha256"] != file_sha256(args.joint_plan)
                or development["cross_panel_text_overlap"] != 0):
            raise ValueError("development plan is not audited for this training pool")
        evaluation_sources.extend(development["sources"])
    if len({s['name'] for s in evaluation_sources}) != len(evaluation_sources):
        raise ValueError("duplicate evaluation source names")
    for source in evaluation_sources:
        if file_sha256(Path(source['path'])) != source['sha256']:
            raise ValueError(f"evaluation input hash mismatch: {source['name']}")
    if args.theta0:
        theta = json.loads(args.checkpoint.read_text())
        adapter_dir = WORKSPACE / theta["adapter_dir"]
        head_path = WORKSPACE / theta["head_checkpoint"]
        if (file_sha256(head_path) != theta["head_checkpoint_sha256"] or
            file_sha256(adapter_dir / "adapter_model.safetensors") != theta["adapter_model_sha256"]):
            raise ValueError("initialization checkpoint hashes changed")
    else:
        adapter_dir, head_path = args.checkpoint / "adapter", args.checkpoint / "choice_head.pt"
        hashes = json.loads((args.checkpoint / "hashes.json").read_text())
        for key in ("choice_head.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            if file_sha256(args.checkpoint / key) != hashes[key]:
                raise ValueError(f"checkpoint hash mismatch: {key}")
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if state["model_config_sha256"] != file_sha256(args.model_dir / "config.json"):
        raise ValueError("backbone and head mismatch")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    program = config.get("adapter_program", STANDARD_PROGRAM)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.float32, device_map={"": local_rank}, low_cpu_mem_usage=True)
    if program == LATENT_PROGRAM:
        model.model, _ = initialize_latent(model.model, config["genome"], config["rank"],
                                           config["target_modules"], adapter_dir)
    else:
        model.model, _ = initialize_adapter(model.model, program, config.get("r", config.get("rank")),
                                            config["target_modules"], adapter_dir)
    model.eval().requires_grad_(False)
    head = ChoiceHead(state["hidden_size"]).to(device).eval()
    head.load_state_dict(state["state_dict"])
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for source in evaluation_sources:
        path = Path(source["path"])
        # JSON strings may contain Unicode line/paragraph separators. Only
        # physical JSONL record boundaries are delimiters, not str.splitlines().
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != source['count'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError("evaluation count mismatch or duplicate IDs")
        prompts, sizes = [], []
        for row in rows:
            ids, rendered = render_choice_prompts(row)
            prompts.extend(rendered)
            sizes.append(len(ids))
        # Assign complete original batches, preserving their padding and FP32
        # arithmetic. Only inference placement changes between one and N GPUs.
        indices = assigned_prompt_indices(len(prompts), 16, rank, world)
        source_cap = source.get("max_prompt_tokens")
        features = encode_prompts(
            model, tokenizer, [prompts[i] for i in indices], device=device,
            batch_size=16, max_prompt_tokens=source_cap,
            bucket_by_length=source_cap is None,
        ) if indices else None
        if world > 1:
            parts = [None] * world if rank == 0 else None
            dist.gather_object((indices, features), parts, dst=0)
            if rank != 0:
                continue
            features = restore_prompt_order(parts, len(prompts))
        cursor = 0
        predictions = []
        with torch.inference_mode():
            for row, size in zip(rows, sizes, strict=True):
                ids, _ = render_choice_prompts(row)
                response = choice_response(ids, head(features[cursor:cursor + size].cuda()),
                                             temperature=state["temperature"])
                predictions.append({"id": row["id"], "label": row["label"], **response})
                cursor += size
            # Preserve the established end-to-end permutation check per source.
            ids, rendered = render_choice_prompts(rows[0], list(reversed(range(sizes[0]))))
            reverse = encode_prompts(
                model, tokenizer, rendered, device=device, batch_size=16,
                max_prompt_tokens=source_cap, bucket_by_length=source_cap is None,
            )
            response = choice_response(ids, head(reverse.cuda()), temperature=state["temperature"])
            error = max(abs(response["probabilities"][k] - predictions[0]["probabilities"][k]) for k in ids)
            # The check is a numerical diagnostic, not an additional metric.
            # FLA/attention kernels can produce small batch-order-dependent
            # differences even in FP32. Keep the established 1e-3 warning
            # threshold, but fail only for a materially larger discrepancy.
            if error > PERMUTATION_FAILURE_TOLERANCE:
                raise RuntimeError(f"{source['name']}: option permutation error {error} > "
                                   f"{PERMUTATION_FAILURE_TOLERANCE}")
            permutation_warning = error > PERMUTATION_WARNING_TOLERANCE
        results[source["name"]] = {**summarize(predictions),
                                    "source_family": source.get("source_family", source["name"]),
                                    "input_sha256": file_sha256(path),
                                    "panel": source['panel'],
                                    "permutation_max_probability_error": error,
                                    "permutation_warning": permutation_warning,
                                    "permutation_warning_tolerance": PERMUTATION_WARNING_TOLERANCE,
                                    "permutation_failure_tolerance": PERMUTATION_FAILURE_TOLERANCE}
        (args.output_dir / f"{source['name']}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
        print(json.dumps({"source": source["name"], **{k:results[source['name']][k]
            for k in ('record_count','accuracy','macro_f1','brier','ece','nll')}}), flush=True)
        atomic_json(args.output_dir / 'progress.json', {'status':'running','completed_sources':len(results),
            'expected_sources':len(evaluation_sources), 'latest_source':source['name'], 'source_metrics':results,
            'elapsed_seconds':time.monotonic()-started})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return
    panels = {}
    for panel in sorted({v['panel'] for v in results.values()}):
        values = [v for v in results.values() if v['panel']==panel]
        panels[panel] = {'source_count':len(values),'unique_source_count':len(values),
            'source_panel_entry_count':len(values),'record_count':sum(v['record_count'] for v in values),
            'macro': {k: sum(v[k] for v in values)/len(values) for k in ('accuracy','macro_f1','brier','ece','nll')}}
    report_role = args.role if args.role in panels else 'development_monitoring'
    default_panel = 'natural' if 'natural' in panels else 'balanced'
    macro = panels[args.role]['macro'] if args.role in panels else panels[default_panel]['macro']
    counts = population_counts(results)
    for panel, values in panels.items():
        values.update(counts['panel_counts'][panel])
    atomic_json(args.output_dir / "report.json", {"status": "complete", "role": report_role,
        "source_metrics": results, "macro": macro, **counts,
        "record_count": sum(v["record_count"] for v in results.values()),
        "checkpoint": str(args.checkpoint.resolve()), "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
        "head_sha256": file_sha256(head_path), "joint_plan_sha256": file_sha256(args.joint_plan),
        "backbone_dtype": "fp32", "temperature": state["temperature"],
        "world_size": world, "physical_prompt_batch_size": 16,
        "panel_metrics": panels,
        "development_plan_sha256": file_sha256(args.development_plan) if args.development_plan else None,
        "checkpoint_progress": json.loads((args.checkpoint/'progress.json').read_text()) if not args.theta0 else None,
        "code": {str(p):file_sha256(p) for p in (Path(__file__),PROJECT/'src/deciform/development_validation.py',
            PROJECT/'src/deciform/evaluation_batching.py',PROJECT/'src/deciform/choice.py')},
        "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
