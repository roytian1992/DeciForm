"""One fixed test, per-source rows, before/after Static and released baselines."""
import argparse
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parent
sys.path.insert(0,str(PROJECT/'src'))
from deciform.joint_data import atomic_json,file_sha256
from deciform.evaluation import load_inventory,summarize_unified
from deciform.unified_comparison import read_run_predictions

def accuracy_summary(predictions,counts):
    grouped=defaultdict(list)
    for row in predictions:grouped[row['source_family']].append(row)
    sources={}
    for source,expected in sorted(counts.items()):
        rows=grouped[source]
        correct=sum(r['prediction']==r['label'] for r in rows)
        complete=len(rows)==expected
        sources[source]={'expected':expected,'evaluated':len(rows),'complete':complete,
                         'correct':correct if complete else None,
                         'accuracy':correct/expected if complete else None}
    complete=all(s['complete'] for s in sources.values())
    return {'sources':sources,'complete':complete,'evaluated':len(predictions),'expected':sum(counts.values()),
            'macro_source_accuracy':sum(s['accuracy'] for s in sources.values())/len(sources) if complete else None,
            'pooled_accuracy':sum(s['correct'] for s in sources.values())/sum(counts.values()) if complete else None}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    primary_models=config.get('primary_models',list(config['models']))
    if not primary_models or any(k not in config['models'] for k in primary_models):
        raise ValueError('primary_models must name configured models')
    path=WORKSPACE/config['test_manifest']
    _,rows,counts=load_inventory(path)
    digest=file_sha256(path)
    models={}
    predictions_by_model={}
    for key,entry in config['models'].items():
        predictions,status=read_run_predictions(WORKSPACE/entry['output_dir'],rows,digest,
                                                checkpoint=WORKSPACE/entry['checkpoint'])
        summary=accuracy_summary(predictions,counts)
        summary.update(display_name=entry['display_name'],status=status['status'],
                       checkpoint=str(WORKSPACE/entry['checkpoint']))
        if not summary['complete'] and (WORKSPACE/entry['output_dir']/'deferred.json').exists():
            summary['status']='deferred'
        summary['metrics']=summarize_unified(predictions) if summary['complete'] else None
        summary['training_exposure']=entry.get('training_exposure',{})
        models[key]=summary;predictions_by_model[key]=predictions
    paired={}
    after={r['id']:r for r in predictions_by_model['ours_after']}
    for key in ('ours_before','kev','open_jev'):
        other={r['id']:r for r in predictions_by_model[key]}
        comparison={}
        for source in counts:
            selected=[r for r in rows if r['source_family']==source]
            if not all(r['id'] in after and r['id'] in other for r in selected):continue
            differences=[int(after[r['id']]['prediction']==r['label'])-int(other[r['id']]['prediction']==r['label']) for r in selected]
            comparison[source]={'records':len(selected),'accuracy_delta':sum(differences)/len(selected),
                                'after_only_correct':differences.count(1),'other_only_correct':differences.count(-1)}
        paired[key]=comparison
    report={'status':'complete' if all(models[k]['status']=='complete' for k in primary_models) else 'partial',
            'updated_at':datetime.datetime.now().astimezone().isoformat(),'test_manifest':str(path),
            'test_manifest_sha256':digest,'config_sha256':file_sha256(args.config),
            'report_code_sha256':file_sha256(Path(__file__)),'source_counts':counts,'models':models,
            'primary_models':primary_models,
            'paired_deltas':paired,'scope':'fixed-test engineering comparison; released baselines have different training histories'}
    out=WORKSPACE/config['output_dir'];out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'comparison.json',report)
    primary={k:models[k] for k in primary_models}
    names=[m['display_name'] for m in primary.values()]
    lines=['# 固定测试集：Static 与开源模型对比', '',f"更新时间：{report['updated_at']}",'',
           '测试 ID、标签、选项固定；每个数据集只计一次。分数在对应来源测完后显示。',
           '补训前后均为 Static 模型。Kev/Open-Jev 是发布版，不代表同训练预算的方法对照。','',
           '| 数据集 | N | '+' | '.join(names)+' |','|---|---:|'+'---:|'*len(names)]
    for source,n in sorted(counts.items()):
        cells=[]
        for model in primary.values():
            s=model['sources'][source]
            cells.append(f"{100*s['accuracy']:.2f}% ({s['correct']:,}/{n:,})" if s['complete'] else f"待完成 {s['evaluated']:,}/{n:,}")
        lines.append(f'| {source} | {n:,} | '+' | '.join(cells)+' |')
    for key,label in [('macro_source_accuracy','各来源准确率平均（等权）'),('pooled_accuracy','所有样本准确率（样本加权）')]:
        cells=[f"{100*m[key]:.2f}%" if m[key] is not None else '待全部完成' for m in primary.values()]
        lines.append(f'| **{label}** | {len(rows):,} | '+' | '.join(cells)+' |')
    if 'ours_before' not in primary_models:
        previous=models['ours_before']
        lines.extend(['',f"补训前评测：{previous['status']}，保留 {previous['evaluated']:,}/{previous['expected']:,} 条预测；作为可选的前后对照，不影响主对比完成。"])
    lines.extend(['','来源暴露说明：补训前的 F2 priority 已在初始化阶段接触过同类规则；旧表的“未见来源”标注只检查了后续两阶段，已纠正。',
        'Kev 使用过 Banking77、SST-5、TREC 等来源；Open-Jev 发布的 release-v2 未列这些来源。来源暴露和具体测试样本重合分别记录。',
        '当前集合用于工程比较；最终论文确认集须在方案冻结后评测，并审计外部模型的样本重合。',''])
    tmp=out/'comparison.md.tmp';tmp.write_text(chr(10).join(lines));tmp.replace(out/'comparison.md')
    summary={k:{n:v[n] for n in ('status','evaluated','expected')} for k,v in models.items()}
    print(json.dumps({'status':report['status'],'models':summary}),flush=True)

if __name__=='__main__':
    main()
