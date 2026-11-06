"""Overwrite the transfer study block using original, complete result artifacts."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from common import PROJECT, atomic, sha

START='<!-- LAYA_TRANSFER_START -->'
END='<!-- LAYA_TRANSFER_END -->'


def read(p):
    return json.loads(p.read_text()) if p.exists() else None


def update(root):
    manifest=read(root/'experiment_manifest.json')
    records={}
    lines=[START,'','## 13. Laya 上的机制迁移（同骨干、同数据、同损失）','',
        '保留 ModernBERT 全参数微调、原 Laya scorer 和 RLCD；增加任务路由低秩残差、Choice QF、任务几何读出。',
        '两布局分别从相同原始编码器和随机 head 开始；不继承已训练 Laya。新增分支初始输出为零。',
        '同 200k 固定训练视图、1 epoch、batch64、原生完整 Dev/Test。单 seed；不能称为跨 seed 稳定性。','',
        '| 布局 | 模型 | Choice Acc ↑ | Noul Acc ↑ | Score median MAE ↓ | 状态 |',
        '|---|---|---:|---:|---:|---|']
    for arm in ('sequential','parallel'):
        out=root/arm;cfg=read(out/'config.json');base=Path(cfg['baseline_reference'])
        original=read(base/'evaluations/test_in_domain.original.json')
        assert original['rows']==15000
        status=read(out/'status.json') or {'phase':'pending'}
        done=read(out/'complete.json')
        quality=read(out/'evaluations/test_in_domain.original.json')
        position=read(out/'position_sensitivity.json')
        record={'baseline':str(base),'status':status['phase'],'complete':done,'quality':quality,'position':position}
        for label,d in [('原 Laya',original),('Laya＋迁移机制',quality)]:
            state='已完成' if label=='原 Laya' or done else {'train':'训练/评测中','preflight':'预检中','failed':'失败需检查'}.get(status['phase'],status['phase'])
            values=[f"{d['task/choice']['accuracy']:.4f}",f"{d['task/noul']['accuracy']:.4f}",f"{d['task/score']['mae']:.4f}"] if d else ['Placeholder']*3
            lines.append(f'| {arm} | {label} | '+' | '.join(values)+f' | {state} |')
        if quality:
            assert quality['rows']==15000 and quality['native_full_candidates']
            delta={'choice_accuracy':quality['task/choice']['accuracy']-original['task/choice']['accuracy'],
                   'noul_accuracy':quality['task/noul']['accuracy']-original['task/noul']['accuracy'],
                   'score_mae_reduction':original['task/score']['mae']-quality['task/score']['mae']}
            record.update(delta_improvement=delta,all_three_improve=all(x>0 for x in delta.values()),
                          quality_path=str(out/'evaluations/test_in_domain.original.json'),
                          quality_sha256=sha(out/'evaluations/test_in_domain.original.json'))
        records[arm]=record
    lines+=['','| 布局 | Choice 任一顺序改判率 | Noul 任一顺序改判率 | 训练秒数 | 新增参数 |',
            '|---|---:|---:|---:|---:|']
    for arm,record in records.items():
        p=record['position'];d=record['complete'];initial=read(root/arm/'initialization.json')
        values=[f"{p['any_order_flip_rate'][t]:.4%}" if p else 'Placeholder' for t in ('choice','noul')]
        values += [f"{d['train_seconds']:.1f}" if d else 'Placeholder',
                   f"{initial['mechanisms']['added_parameters']:,}" if initial else 'Placeholder']
        lines.append(f'| {arm} | '+' | '.join(values)+' |')
    lines+=['','所有迁移运行均报告完整三任务结果，不按测试集选择有利子集或覆盖原基线。',
        '收益同时包含新增共享低秩/双线性容量和任务机制，当前两组不隔离各组件贡献。',
        '保留原布局；不将候选局部 head 等同于候选独立编码。',
        f'运行根：{chr(96)}{root.relative_to(PROJECT)}{chr(96)}；',
        '方法与命令：[迁移实验记录](experiment_log_20261106_laya_mechanisms.md)。','',END]
    atomic(root/'results_summary.json',{'updated_unix':time.time(),'arms':records,'all_layouts_three_task_win':
        all(r.get('all_three_improve',False) and r['complete'] for r in records.values())})
    doc=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=doc.read_text();block='\n'.join(lines)
        if START in s:
            left=s.index(START);right=s.index(END,left)+len(END);s=s[:left]+block+s[right:]
        else:s=s.rstrip()+'\n\n'+block+'\n'
        temp=doc.with_suffix('.md.transfer.tmp');temp.write_text(s);temp.replace(doc)
    return all(r['complete'] or r['status']=='failed' for r in records.values())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    while True:
        done=update(a.root.resolve())
        if not a.watch or done:break
        time.sleep(30)
