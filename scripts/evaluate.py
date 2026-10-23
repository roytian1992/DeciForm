"""Evaluate a frozen checkpoint on one locked test inventory, without truncation."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT
sys.path.insert(0, str(PROJECT / "src"))
from deciform.joint_data import atomic_json, file_sha256
from deciform.evaluation import load_inventory, summarize_unified


def aggregate(args, manifest, rows, source_counts):
    expected = {r["id"]: r for r in rows}
    predictions, seen = [], set()
    workers = []
    for rank in range(args.workers):
        status = json.loads((args.output_dir / f"rank_{rank}_progress.json").read_text())
        provenance = json.loads((args.output_dir / f"rank_{rank}_manifest.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"worker {rank} has not completed")
        if provenance["test_manifest_sha256"] != file_sha256(args.manifest):
            raise ValueError("worker used a different test inventory")
        if provenance["checkpoint"] != str(args.checkpoint.resolve()):
            raise ValueError("worker used a different checkpoint")
        workers.append(provenance)
        with (args.output_dir / f"rank_{rank}_predictions.jsonl").open() as handle:
            for line in handle:
                p = json.loads(line)
                row = expected.get(p["id"])
                if row is None or p["id"] in seen:
                    raise ValueError("unknown or duplicate test prediction")
                if row["label"] != p["label"] or row["source_family"] != p["source_family"]:
                    raise ValueError("prediction label/source mismatch")
                ids = row["question"].get("option_ids", row["question"]["options"])
                if set(p["probabilities"]) != set(ids):
                    raise ValueError("prediction option set mismatch")
                seen.add(p["id"])
                predictions.append(p)
    if set(expected) != seen:
        raise ValueError("test prediction coverage is incomplete")
    metrics = summarize_unified(predictions)
    report = {"status": "complete", "role": "unified_test", **metrics,
              "expected_record_count": len(rows), "failed_records": 0, "skipped_records": 0,
              "checkpoint": str(args.checkpoint.resolve()),
              "test_manifest": str(args.manifest.resolve()),
              "test_manifest_sha256": file_sha256(args.manifest),
              "overlap_audit": str(args.audit.resolve()),
              "overlap_audit_sha256": file_sha256(args.audit),
              "source_counts": source_counts, "workers": workers,
              "completed_at": datetime.datetime.now().astimezone().isoformat()}
    atomic_json(args.output_dir / "report.json", report)
    atomic_json(args.output_dir / "progress.json", {"status": "complete",
        "evaluated_records": len(predictions), "expected_records": len(rows),
        "report": str((args.output_dir / "report.json").resolve())})
    print(json.dumps({"status": "complete", **metrics}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    manifest, rows, counts = load_inventory(args.manifest)
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "pass" or audit["test_manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("a passing train/test isolation audit is required")
    if args.aggregate_only:
        aggregate(args, manifest, rows, counts)
        return
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from deciform.lora import initialize_adapter
    from deciform.lora import STANDARD_PROGRAM
    from deciform.choice import ChoiceHead, choice_response, encode_prompts, render_choice_prompts
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.workers or torch.cuda.device_count() != world:
        raise ValueError("one evaluation worker per visible GPU is required")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", local_rank)
    assigned = rows[rank::world]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / f"rank_{rank}_progress.json"
    prediction_path = args.output_dir / f"rank_{rank}_predictions.jsonl"
    manifest_path = args.output_dir / f"rank_{rank}_manifest.json"
    if prediction_path.exists() or manifest_path.exists():
        raise FileExistsError("evaluation outputs already exist; use a fresh output directory")
    started = time.monotonic()
    completed = 0
    def progress(status, **extra):
        elapsed = time.monotonic()-started
        atomic_json(progress_path, {"status": status, "rank": rank, "world_size": world,
            "evaluated_records": completed, "expected_records": len(assigned),
            "elapsed_seconds": elapsed, "rows_per_second": completed/elapsed if elapsed else 0,
            "updated_at": datetime.datetime.now().astimezone().isoformat(), **extra})
    progress("loading_model")
    try:
        hashes = json.loads((args.checkpoint / "hashes.json").read_text())
        for key in ("choice_head.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            if file_sha256(args.checkpoint / key) != hashes[key]:
                raise ValueError(f"checkpoint hash mismatch: {key}")
        state = torch.load(args.checkpoint / "choice_head.pt", map_location="cpu", weights_only=True)
        if state["model_config_sha256"] != file_sha256(args.model_dir / "config.json"):
            raise ValueError("backbone and head mismatch")
        config = json.loads((args.checkpoint / "adapter/adapter_config.json").read_text())
        program = config.get("adapter_program", STANDARD_PROGRAM)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForImageTextToText.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.float32, device_map={"": local_rank}, low_cpu_mem_usage=True)
        model.model, _ = initialize_adapter(model.model, program, config.get("r", config.get("rank")),
                                             config["target_modules"], args.checkpoint / "adapter")
        model.eval().requires_grad_(False)
        head = ChoiceHead(state["hidden_size"]).to(device).eval()
        head.load_state_dict(state["state_dict"])
        code_paths = [Path(__file__), *sorted((PROJECT/"src/deciform").glob("*.py"))]
        atomic_json(manifest_path, {"rank": rank, "world_size": world,
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_hashes": hashes,
            "test_manifest": str(args.manifest.resolve()), "test_manifest_sha256": file_sha256(args.manifest),
            "model_dir": str(args.model_dir.resolve()), "model_config_sha256": state["model_config_sha256"],
            "temperature": state["temperature"], "temperature_fitted_on_test": False,
            "backbone_dtype": "fp32", "physical_prompt_batch_size": 16, "row_chunk_size": 64,
            "max_prompt_tokens": None, "truncation": False, "bucket_by_length": True,
            "legacy_stream_token_caps": "ignored: full text is evaluated for every locked record",
            "python": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "command": sys.argv, "code": {str(p):file_sha256(p) for p in code_paths}})
        progress("running")
        permutation_checks = {}
        with prediction_path.open("x", encoding="utf-8") as output:
            for start in range(0, len(assigned), 64):
                chunk = assigned[start:start+64]
                prompts, sizes, ids_per_row = [], [], []
                for row in chunk:
                    ids, rendered = render_choice_prompts(row)
                    ids_per_row.append(ids)
                    sizes.append(len(ids))
                    prompts.extend(rendered)
                features = encode_prompts(model, tokenizer, prompts, device=device, batch_size=16,
                    max_prompt_tokens=None, bucket_by_length=True)
                cursor = 0
                with torch.inference_mode():
                    for row, ids, size in zip(chunk, ids_per_row, sizes, strict=True):
                        response = choice_response(ids, head(features[cursor:cursor+size].to(device)),
                                                   temperature=state["temperature"])
                        cursor += size
                        family = row["source_family"]
                        if family not in permutation_checks:
                            reverse_ids, rendered = render_choice_prompts(row, list(reversed(range(size))))
                            reverse = encode_prompts(model, tokenizer, rendered, device=device, batch_size=16,
                                max_prompt_tokens=None, bucket_by_length=True)
                            reverse_response = choice_response(reverse_ids, head(reverse.to(device)),
                                                               temperature=state["temperature"])
                            error = max(abs(response["probabilities"][k]-reverse_response["probabilities"][k]) for k in ids)
                            permutation_checks[family] = {"max_probability_error": error, "warning": error>1e-3}
                            if error > 5e-3:
                                raise RuntimeError(f"option permutation diagnostic failed: {family}, {error}")
                        p = {"id": row["id"], "source_family": family,
                             "test_stream": row["test_stream"], "label": row["label"], **response}
                        output.write(json.dumps(p, ensure_ascii=False)+"\n")
                output.flush()
                completed += len(chunk)
                progress("running", latest_source=chunk[-1]["source_family"], permutation_checks=permutation_checks)
                print(json.dumps({"rank": rank, "evaluated": completed, "expected": len(assigned)}), flush=True)
        progress("complete", permutation_checks=permutation_checks, prediction_sha256=file_sha256(prediction_path))
    except Exception as exc:
        progress("failed", error=repr(exc), traceback=traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
