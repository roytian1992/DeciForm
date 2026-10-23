"""Report one test inventory by our training relationship and compare models."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parent
sys.path.insert(0,str(PROJECT/"src"))
from deciform.joint_data import atomic_json,file_sha256
from deciform.evaluation import load_inventory
from deciform.unified_comparison import relationship_inventory,read_run_predictions,grouped_results,GROUP_LABELS


def paired_accuracy(left,right,rows,inventory,seed=20261023,repeats=10000):
    import numpy as np
    left={r["id"]:r for r in left}
    right={r["id"]:r for r in right}
    groups={"all":rows}
    groups.update({key:[r for r in rows if inventory["source_relationships"][r["source_family"]]["group"]==key]
                   for key in GROUP_LABELS})
    groups.update({source:[r for r in rows if r["source_family"]==source]
                   for source in inventory["source_relationships"]})
    rng=np.random.default_rng(seed)
    result={}
    for name,selected in groups.items():
        if not all(r["id"] in left and r["id"] in right for r in selected):continue
        differences=[int(left[r["id"]]["prediction"]==r["label"])-int(right[r["id"]]["prediction"]==r["label"]) for r in selected]
        counts=np.bincount(np.asarray(differences)+1,minlength=3)
        draws=rng.multinomial(len(selected),counts/len(selected),size=repeats)
        estimates=(draws[:,2]-draws[:,0])/len(selected)
        result[name]={"record_count":len(selected),"accuracy_delta_ours_minus_baseline":sum(differences)/len(selected),
            "paired_bootstrap_95":np.quantile(estimates,[.025,.975]).tolist(),
            "ours_only_correct":int(counts[2]),"baseline_only_correct":int(counts[0]),
            "seed":seed,"bootstrap_repeats":repeats,
            "resampling_unit":"paired record; fixed empirical source mixture, no training-seed uncertainty"}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    manifest_path=WORKSPACE/config["test_manifest"]
    manifest,rows,counts=load_inventory(manifest_path)
    manifest_hash=file_sha256(manifest_path)
    inventory=relationship_inventory(manifest,rows)
    models={}
    predictions_by_model={}
    for key,entry in config["models"].items():
        predictions,status=read_run_predictions(WORKSPACE/entry["output_dir"],rows,manifest_hash,
                                                checkpoint=WORKSPACE/entry["checkpoint"])
        predictions_by_model[key]=predictions
        models[key]={"display_name":entry["display_name"],"status":status["status"],
            "output_dir":str(WORKSPACE/entry["output_dir"]),
            "evaluated_records":status["evaluated_records"],"expected_records":len(rows),
            "worker_status":[w["progress"] for w in status["workers"]],
            **grouped_results(predictions,inventory)}
    paired={}
    for baseline in ("kev","open_jev"):
        paired[baseline]=paired_accuracy(predictions_by_model["ours_static_full"],predictions_by_model[baseline],rows,inventory)
    report={"status":"complete" if all(v["status"]=="complete" for v in models.values()) else "partial",
        "updated_at":datetime.datetime.now().astimezone().isoformat(),
        "test_manifest":str(manifest_path),"test_manifest_sha256":manifest_hash,
        "config_sha256":file_sha256(args.config),"inventory":inventory,"models":models,
        "paired_accuracy":paired,
        "report_code":{str(p):file_sha256(p) for p in [Path(__file__),PROJECT/"src/deciform/unified_comparison.py"]}}
    out=WORKSPACE/config["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"comparison.json",report)
    atomic_json(out/"source_relationships.json",{"test_manifest_sha256":manifest_hash,**inventory})
    labels=[v["display_name"] for v in models.values()]
    lines=["# 统一测试集：来源关系与模型比较","",f"更新：{report['updated_at']}","",
        "分类参照我们完成的微调；不自动代表 Open-Jev/Kev 或基础模型预训练的来源暴露。",
        "当前 Ours 为 Static-Full + delta checkpoint。两个分组互斥，合计 68,156 条。","",
        "| 类别 / 数据集 | 测试条数 | "+" | ".join(labels)+" |",
        "|---|---:|"+"---:|"*len(labels)]
    def cell(result):
        if result["complete"]:return f"{100*result['metrics']['accuracy']:.2f}%"
        return f"待完成（{result['evaluated_records']:,}/{result['expected_records']:,}）"
    for group in GROUP_LABELS:
        meta=inventory["groups"][group]
        lines.append("| **"+GROUP_LABELS[group]+"** | "+f"{meta['record_count']:,}"+" | "+" | ".join(cell(m["groups"][group]) for m in models.values())+" |")
        for source,count in meta["source_counts"].items():
            lines.append(f"| {source} | {count:,} | "+" | ".join(cell(m["sources"][source]) for m in models.values())+" |")
    lines.append("| **全集** | "+f"{len(rows):,}"+" | "+" | ".join(cell(m["overall"]) for m in models.values())+" |")
    lines.extend(["","表内为 accuracy；仅在对应来源/分组覆盖完整后显示分数。DBpedia14 占全集多数，需同时查看分组与逐来源结果。","",
                  "| 模型 | 范围 | N | Accuracy | Macro-F1 | Brier | ECE | NLL |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for model in models.values():
        for key,item in [("全集",model["overall"]),*[(GROUP_LABELS[k],v) for k,v in model["groups"].items()]]:
            if item["complete"]:
                m=item["metrics"]
                lines.append(f"| {model['display_name']} | {key} | {item['expected_records']:,} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} | {m['nll']:.4f} |")
    lines.extend(["","Macro-F1 为各来源 macro-F1 的平均；其余指标在对应记录上计算，ECE 重新分箱。","",
                  "| 配对比较 | 范围 | N | Δ accuracy（百分点） | 95% paired bootstrap CI |",
                  "|---|---|---:|---:|---|"])
    for baseline,scopes in paired.items():
        for scope,item in scopes.items():
            low,high=item["paired_bootstrap_95"]
            scope_label=GROUP_LABELS.get(scope,"全集" if scope=="all" else scope)
            lines.append(f"| Ours − {config['models'][baseline]['display_name']} | {scope_label} | {item['record_count']:,} | {100*item['accuracy_delta_ours_minus_baseline']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines.append("")
    temporary=out/"comparison.md.tmp"
    temporary.write_text("\n".join(lines),encoding="utf-8")
    temporary.replace(out/"comparison.md")
    print(json.dumps({"status":report["status"],"output":str(out),
        "groups":inventory["groups"],"models":{k:{f:v[f] for f in ['status','evaluated_records','expected_records']} for k,v in models.items()}},ensure_ascii=False))


if __name__=="__main__":
    main()
