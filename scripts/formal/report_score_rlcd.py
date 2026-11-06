"""Update only the RLCD Score study block using complete endpoint artifacts."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from common import PROJECT, atomic, sha

START='<!-- D2_RLCD_SCORE_START -->'
END='<!-- D2_RLCD_SCORE_END -->'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def update(root):
    manifest=read(root/'experiment_manifest.json')
    lines=[START,'','## D2＋Laya RLCD：固定架构的 Score 损失对照','',
      '固定专项 D2 的 Qwen backbone、模块、初始化、数据、分批、优化器、结构正则与中位数解码；仅将数据目标改为 Laya0.4.26 原生 RLCD。',
      '每个面板独立从原 Base 训练1轮，batch64；不继承20万条模型。ASAP 7,784条；连续面板8,129条。评测完整支持，不截断、不进行Score顺序实验。','',
      '| 面板 | 模型 | Test MAE ↓ | 等题目/来源 NMAE ↓ | 状态 |',
      '|---|---|---:|---:|---|']
    records={}
    for panel,entry in manifest['arms'].items():
        out=root/panel;ref=Path(entry['reference'])
        cfg=read(out/'config.json');status=read(out/'status.json') or {'phase':'pending'}
        done=read(out/'complete.json')
        result=read(out/'evaluations/epoch_1.test_in_domain.json') if done else None
        reference=read(ref/'evaluations/epoch_1.test_in_domain.json')
        assert reference['rows']==(2598 if panel=='asap' else 2379)
        if result:
            assert result['rows']==reference['rows'] and result['complete_output_grid']
            assert done['sample_visits']==cfg['train_rows'] and done['steps']==cfg['total_steps']
        names=[('ASAP','task/score')] if panel=='asap' else [('STS-B','source/stsb'),('WMT英中','source/wmt_en_zh')]
        for label,data in [('D2 CDF-log',reference),('D2＋RLCD',result)]:
            state='已完成' if label=='D2 CDF-log' or result else {'preflight':'预检中','train':'训练/评测中','failed':'失败需检查'}.get(status['phase'],status['phase'])
            for name,key in names:
                mae=f"{data[key]['mae']:.6f}" if data else 'Placeholder'
                nmae=f"{data['macro_source_nmae']:.6f}" if data else 'Placeholder'
                lines.append(f'| {name} | {label} | {mae} | {nmae} | {state} |')
        records[panel]={'status':status,'complete':done,'baseline':reference,'result':result,
            'result_path':str(out/'evaluations/epoch_1.test_in_domain.json') if result else None,
            'result_sha256':sha(out/'evaluations/epoch_1.test_in_domain.json') if result else None}
    lines+=['','连续面板的 NMAE 是 STS-B/WMT 两源等权汇总，不是各单源指标；原生 MAE 分别报告。',
        '单seed探索；不按测试结果选模型或更换解码。训练成本、完整原始结果和逐样本预测保存在运行目录。',
        '运行根：runs/formal_d2_rlcd_score_20261106_v1。方法与协议见 [RLCD迁移记录](experiment_log_20261106_d2_rlcd_score.md)。','',END]
    atomic(root/'results_summary.json',{'updated_unix':time.time(),'arms':records})
    doc=PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with (PROJECT/'runs/final_record_update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=doc.read_text();block='\n'.join(lines)
        if START in s:
            a=s.index(START);b=s.index(END,a)+len(END);s=s[:a]+block+s[b:]
        else:s=s.rstrip()+'\n\n'+block+'\n'
        temp=doc.with_suffix('.md.rlcd.tmp');temp.write_text(s);temp.replace(doc)
    return all(v['complete'] or v['status']['phase']=='failed' for v in records.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        done=update(args.root.resolve())
        if done or not args.watch:break
        time.sleep(30)
