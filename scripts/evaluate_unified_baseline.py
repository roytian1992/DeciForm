"""Evaluate released native baselines on every row of the common test set."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parent
EXTERNAL=WORKSPACE/"RSI_Research/external"
sys.path[:0]=[str(PROJECT/"src"),str(EXTERNAL)]
from deciform.joint_data import file_sha256,atomic_json
from deciform.evaluation import load_inventory,summarize_unified
from deciform.unified_native import native_probabilities,option_fields
from deciform.unified_comparison import read_run_predictions,validate_prediction


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--model",choices=["kev","open_jev"],required=True)
    parser.add_argument("--aggregate-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    entry=config["models"][args.model]
    checkpoint=WORKSPACE/entry["checkpoint"]
    output=WORKSPACE/entry["output_dir"]
    base=WORKSPACE/entry["backbone"]
    manifest_path=WORKSPACE/config["test_manifest"]
    audit_path=WORKSPACE/config["overlap_audit"]
    manifest,rows,counts=load_inventory(manifest_path)
    manifest_hash=file_sha256(manifest_path)
    audit=json.loads(audit_path.read_text())
    if audit["status"]!="pass" or audit["test_manifest_sha256"]!=manifest_hash:
        raise ValueError("matching passing train/test audit required")
    if args.aggregate_only:
        predictions,status=read_run_predictions(output,rows,manifest_hash,require_complete=True,checkpoint=checkpoint)
        metrics=summarize_unified(predictions)
        atomic_json(output/"report.json",{"status":"complete","model":args.model,**metrics,
            "test_manifest_sha256":manifest_hash,"test_manifest":str(manifest_path),
            "checkpoint":str(checkpoint),"source_counts":counts,"workers":status["workers"],
            "failed_records":0,"skipped_records":0,"completed_at":now()})
        atomic_json(output/"progress.json",{"status":"complete","evaluated_records":len(predictions),
                                              "expected_records":len(rows),"updated_at":now()})
        print(json.dumps({"model":args.model,"status":"complete","records":len(predictions),
                          "aggregate":metrics["aggregate"]}),flush=True)
        return
    import torch
    import transformers
    rank=int(os.environ.get("RANK","0"))
    local_rank=int(os.environ.get("LOCAL_RANK","0"))
    world=int(os.environ.get("WORLD_SIZE","1"))
    if world!=config["execution"]["workers"] or torch.cuda.device_count()!=world:
        raise ValueError("worker/GPU count mismatch")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device=f"cuda:{local_rank}"
    assigned=rows[rank::world]
    batch_size=config["execution"]["native_record_batch_size"]
    output.mkdir(parents=True,exist_ok=True)
    prediction_path=output/f"rank_{rank}_predictions.jsonl"
    progress_path=output/f"rank_{rank}_progress.json"
    provenance_path=output/f"rank_{rank}_manifest.json"
    checkpoint_hashes={str(p.relative_to(checkpoint)):file_sha256(p) for p in sorted(checkpoint.rglob("*"))
                       if p.is_file() and p.suffix in {".pt",".json",".safetensors"} and ".cache" not in p.parts}
    native_files=([EXTERNAL/"open_jev_loader.py",EXTERNAL/"open-jev-source/jev/model.py",EXTERNAL/"open-jev-source/jev/api.py"]
                  if args.model=="open_jev" else [EXTERNAL/"kev_checkpoint_loader.py",EXTERNAL/"kev-source/kev/model.py",EXTERNAL/"kev-source/kev/checkpoint.py"])
    code_files=[Path(__file__),PROJECT/"src/deciform/unified_native.py",PROJECT/"src/deciform/evaluation.py",
                PROJECT/"src/deciform/unified_comparison.py",*native_files]
    code_hashes={str(p):file_sha256(p) for p in code_files}
    completed=0
    previous_elapsed=0
    if prediction_path.exists() or provenance_path.exists():
        if not args.resume:
            raise FileExistsError("outputs already exist; use --resume after inspection")
        previous=json.loads(provenance_path.read_text())
        for key,value in {"test_manifest_sha256":manifest_hash,"checkpoint_hashes":checkpoint_hashes,
                          "code":code_hashes,"comparison_config_sha256":file_sha256(args.config)}.items():
            if previous[key]!=value:
                raise ValueError(f"resume provenance changed: {key}")
        if prediction_path.exists():
            with prediction_path.open() as handle:
                for line in handle:
                    row=json.loads(line)
                    validate_prediction(row,assigned[completed])
                    completed+=1
        if progress_path.exists():previous_elapsed=json.loads(progress_path.read_text()).get("elapsed_seconds",0)
        if completed==len(assigned):
            atomic_json(progress_path,{"status":"complete","rank":rank,"world_size":world,
                "evaluated_records":completed,"expected_records":len(assigned),
                "elapsed_seconds":previous_elapsed,"prediction_sha256":file_sha256(prediction_path),
                "updated_at":now()})
            return
    started=time.monotonic()
    def progress(status,**extra):
        elapsed=previous_elapsed+time.monotonic()-started
        atomic_json(progress_path,{"status":status,"model":args.model,"rank":rank,"world_size":world,
            "evaluated_records":completed,"expected_records":len(assigned),
            "elapsed_seconds":elapsed,"rows_per_second":completed/elapsed if elapsed else 0,
            "updated_at":now(),**extra})
    progress("loading_model")
    try:
        if args.model=="open_jev":
            for key in ["JEV_LOAD_8BIT","JEV_LOAD_4BIT","JEV_DEVICE_MAP","JEV_MAX_MEMORY","JEV_ALLOW_OFFLOAD"]:
                os.environ.pop(key,None)
            os.environ["JEV_TORCH_DTYPE"]="bfloat16"
            from open_jev_loader import load_open_jev
            tokenizer,model=load_open_jev(checkpoint,device,base)
            release_limit=model.max_length
            base_config=json.loads((base/"config.json").read_text())
            model.max_length=base_config.get("text_config",base_config)["max_position_embeddings"]
            context={"release_max_length":release_limit,"runtime_max_length":model.max_length,
                     "reason":"evaluate full text up to architecture limit; no truncation"}
            temperature=float(model.temperature)
            dtype="bfloat16"
        else:
            from kev_checkpoint_loader import load_kev
            tokenizer,model=load_kev(checkpoint,device,base)
            from kev.model import SERVE_MAX_STATE,SERVE_MAX_BRANCH
            context={"max_state_tokens":SERVE_MAX_STATE,"max_branch_tokens":SERVE_MAX_BRANCH,
                     "admission":"strict published serving context; no truncation"}
            temperature=float(model.head.temperature)
            dtype="float32"
        model.eval().requires_grad_(False)
        provenance={"model":args.model,"rank":rank,"world_size":world,
            "checkpoint":str(checkpoint.resolve()),"checkpoint_hashes":checkpoint_hashes,
            "test_manifest":str(manifest_path),"test_manifest_sha256":manifest_hash,
            "comparison_config_sha256":file_sha256(args.config),"backbone":str(base),
            "base_config_sha256":file_sha256(base/"config.json"),"temperature":temperature,
            "temperature_fitted_on_test":False,"dtype":dtype,"context":context,
            "native_record_batch_size":batch_size,"option_text":"option_definitions with option_ids restored",
            "truncation":False,"skip_records":False,"command":sys.argv,"python":sys.executable,
            "torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,
            "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"code":code_hashes}
        if not provenance_path.exists():atomic_json(provenance_path,provenance)
        progress("running")
        with prediction_path.open("a" if args.resume else "x",encoding="utf-8") as handle:
            for start in range(completed,len(assigned),batch_size):
                batch=assigned[start:start+batch_size]
                probabilities=native_probabilities(args.model,model,tokenizer,batch)
                for row,values in zip(batch,probabilities,strict=True):
                    ids,_=option_fields(row)
                    if len(values)!=len(ids):raise ValueError("baseline option count mismatch")
                    selected=max(range(len(values)),key=values.__getitem__)
                    prediction={"id":row["id"],"label":row["label"],"source_family":row["source_family"],
                        "test_stream":row["test_stream"],"prediction":ids[selected],
                        "probabilities":dict(zip(ids,values,strict=True)),"confidence":values[selected]}
                    validate_prediction(prediction,row)
                    handle.write(json.dumps(prediction,ensure_ascii=False)+"\n")
                handle.flush()
                completed+=len(batch)
                progress("running",latest_source=batch[-1]["source_family"])
                if completed%32==0:print(json.dumps({"model":args.model,"rank":rank,"completed":completed,"expected":len(assigned)}),flush=True)
        progress("complete",prediction_sha256=file_sha256(prediction_path))
    except Exception as exc:
        progress("failed",error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=="__main__":
    main()
