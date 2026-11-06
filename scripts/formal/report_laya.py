"""Collect completed Laya results and replace one current-results document block."""
import argparse
import json
import time
from pathlib import Path

from common import PROJECT, atomic, sha

BEGIN='<!-- LAYA_CURRENT_START -->'
END='<!-- LAYA_CURRENT_END -->'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def update(root):
    results={}
    for arm in ('sequential','parallel'):
        out=root/arm
        status=read(out/'status.json') or {'phase':'pending'}
        quality=read(out/'evaluations/test_in_domain.original.json')
        position=read(out/'position_sensitivity.json')
        done=read(out/'complete.json')
        record={'status':status['phase'],'quality':quality,'position':position,'complete':done,
                'sources':{'quality':str(out/'evaluations/test_in_domain.original.json'),
                           'position':str(out/'position_sensitivity.json')}}
        if quality:
            assert quality['rows']==15000 and quality['native_full_candidates']
            record['quality_sha256']=sha(out/'evaluations/test_in_domain.original.json')
        if position:
            assert position['tasks']==['choice','noul'] and position['native_full_candidates']
        results[arm]=record
    d2path=PROJECT/'runs/formal_independent_20261105_v1/evaluations/epoch_1.test_in_domain.json'
    d2=read(d2path)
    summary={'updated_unix':time.time(),'arms':results,'d2_reference':d2,
             'd2_reference_path':str(d2path),'d2_reference_sha256':sha(d2path)}
    atomic(root/'results_summary.json',summary)
    lines=[BEGIN,'','## 12. Laya 跨骨干对照（同 20 万条）','',
        '运行目录：`runs/formal_laya_20261106_v1/`；执行与迁移研究说明：',
        '[Laya 实验记录](experiment_log_20261106_laya.md)。','',
        '两组都从原始 ModernBERT-large 预训练编码器和随机 Laya head 开始，',
        '复用原 200k 固定训练视图、1 epoch、global batch 64 和原生完整 dev/test。',
        'Laya 使用上游 RLCD、全参数编码器微调及原生余弦学习率；与 D2 是同监督量的跨骨干比较。','',
        '| 模型 | Test Choice Acc ↑ | Test Noul Acc ↑ | Test Score median MAE ↓ | 状态 |',
        '|---|---:|---:|---:|---|']
    lines.append(f"| 新版 D2 | {d2['task/choice']['accuracy']:.4f} | {d2['task/noul']['accuracy']:.4f} | {d2['task/score']['mae']:.4f} | 已完成 |")
    for arm,r in results.items():
        q=r['quality']
        vals=[f"{q['task/choice']['accuracy']:.4f}",f"{q['task/noul']['accuracy']:.4f}",f"{q['task/score']['mae']:.4f}"] if q else ['Placeholder']*3
        lines.append('| Laya '+arm+' | '+' | '.join(vals)+' | '+('已完成' if r['complete'] else r['status'])+' |')
    lines+=['','位置审计严格复用新版 D2 的 Choice/Noul 10,000 条、七种顺序和语义 ID 对齐规则。',
            'Score 不加入分类改判率。','',
            '| 模型 | Choice reverse flip | Choice any-order flip | Noul reverse flip | Noul any-order flip |',
            '|---|---:|---:|---:|---:|']
    for arm,r in results.items():
        p=r['position']
        vals=[f"{p['reports']['reverse']['tasks']['choice']['flip_rate']:.4%}",
              f"{p['any_order_flip_rate']['choice']:.4%}",
              f"{p['reports']['reverse']['tasks']['noul']['flip_rate']:.4%}",
              f"{p['any_order_flip_rate']['noul']:.4%}"] if p else ['Placeholder']*4
        lines.append('| Laya '+arm+' | '+' | '.join(vals)+' |')
    lines+=['','质量来源：各 arm 的 `evaluations/test_in_domain.original.json`；',
        'Score 的 median、mean、argmax 和 mean-nearest 同时保存在 `evaluations/test_in_domain.score_rules.json`。',
        '位置来源：各 arm 的 `position_sensitivity.json`。待完成单元格只在完整结果产生后覆盖。',
        '文本与候选均未硬截断；原始 8,192-token 范围外有 train/dev/test 各 4/1/2 条，',
        '最长 9,662 token，使用原 RoPE 频率完整计算，并已通过实际 GPU 前向/反向预检。','',END]
    doc=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    content=doc.read_text()
    block='\n'.join(lines)+'\n'
    if BEGIN in content:
        start=content.index(BEGIN);end=content.index(END,start)+len(END)
        content=content[:start]+block.rstrip()+content[end:]
    else:
        content=content.rstrip()+'\n\n'+block
    temp=doc.with_suffix('.md.laya.tmp');temp.write_text(content);temp.replace(doc)
    return all(r['complete'] or r['status']=='failed' for r in results.values())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--watch',action='store_true')
    args=ap.parse_args()
    while True:
        finished=update(args.root.resolve())
        if finished or not args.watch:
            return
        time.sleep(60)


if __name__=='__main__':
    main()
