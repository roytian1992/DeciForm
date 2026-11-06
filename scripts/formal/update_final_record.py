"""Overwrite current result tables from completed artifacts; never append snapshots."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
DOC = PROJECT / 'docs/experiment_record_final_d2_20261105.md'
RECHECK = 'runs/formal_score_recheck_20261106_v1'
EXPLORATION = 'runs/formal_asap_exploration_20261106_v1'
CURVES = 'runs/formal_independent_curves_20261105_v1'
TRIE = 'runs/formal_numeric_trie_20261106_v1'
TOKEN = 'runs/formal_numeric_trie_consistency_20261106_v4'
WATCH = [f'{RECHECK}/asap_json_replay', f'{CURVES}/d2_100k',
         f'{EXPLORATION}/e1_evidence_cdf', f'{EXPLORATION}/e2_median_laplace',
         f'{EXPLORATION}/e3_evidence_median_laplace',
         *[f'{TOKEN}/{arm}' for arm in ('asap_token_ce','asap_token_ce_cdf','continuous_token_ce','continuous_token_ce_cdf')]]


def read(path):
    return json.loads((PROJECT / path).read_text())


def report(root, recovery=False):
    suffix = 'evaluation_recovery_20261105_v2' if recovery else 'evaluations'
    path = f'{root}/{suffix}/epoch_1.test_in_domain.json'
    if not (PROJECT / root / 'complete.json').exists():
        return None
    if not (PROJECT / path).exists():
        return None
    return read(path)


def status(root):
    p = PROJECT / root / 'status.json'
    if (PROJECT / root / 'complete.json').exists():
        return '已完成'
    if not p.exists():
        manifest = PROJECT / root / 'experiment_manifest.json'
        if manifest.exists():
            m = read(manifest)
            queue = manifest.parent.parent / ('queue_'+m['gpus'].replace(',','')+'.json')
            if queue.exists():
                phase = read(queue).get('phase')
                if phase == 'waiting_for_dependency': return '排队：等待前序实验完成'
                if phase == 'waiting_for_memory_release': return '排队：等待显存释放'
                if phase == 'dependency_failed': return '前序失败，需检查'
        return '待启动'
    d = read(p)
    if d.get('phase') == 'failed': return '失败，待检查'
    if d.get('phase') == 'preflight': return '预检中'
    phase = d.get('progress', {}).get('phase', d.get('phase'))
    return {'training': '训练中', 'train': '训练中', 'evaluating': '评测中',
            'preflight': '预检中', 'failed': '失败，待检查'}.get(phase, phase)


def score_tables():
    asap_gen = (f'{RECHECK}/asap_json_replay' if report(f'{RECHECK}/asap_json_replay')
                else 'runs/formal_score_baselines_20261105_v1/asap_json_sft')
    rows = [
        ('JSON 生成式 LoRA', asap_gen),
        ('Kev', 'runs/formal_score_baselines_20261105_v1/asap_kev'),
        ('独立候选 D2', 'runs/formal_independent_asap_score_20261105_v5/independent_d2_asap'),
        ('独立候选 CE 对照', 'runs/formal_independent_score_completion_20261105_v1/asap_ce'),
        ('数值 CDF', 'runs/formal_score_numeric_asap_20261105_v1'),
        ('R1 CDF＋均值回归', 'runs/formal_score_objectives_20261106_v1/r1_cdf_mean'),
        ('R2 纯 CRPS', 'runs/formal_score_objectives_20261106_v1/r2_crps'),
        ('E1 查询条件证据汇聚＋CDF', f'{EXPLORATION}/e1_evidence_cdf'),
        ('E2 中位数回归＋Laplace 有序分布', f'{EXPLORATION}/e2_median_laplace'),
        ('E3 证据汇聚＋中位数分布', f'{EXPLORATION}/e3_evidence_median_laplace'),
        ('N1 数字前缀树＋CE', f'{TRIE}/asap_n1_trie_ce'),
        ('N2 数字前缀树＋CE/CDF', f'{TRIE}/asap_n2_trie_ce_cdf'),
        ('T1 数字前缀树＋CE＋数字序列监督', f'{TOKEN}/asap_token_ce'),
        ('T2 数字前缀树＋CE/CDF＋数字序列监督', f'{TOKEN}/asap_token_ce_cdf')]
    lines = ['### 9.1 ASAP 专项', '', '| 模型 | Test MAE ↓ | 八题等权 NMAE ↓ | 状态 |',
             '|---|---:|---:|---|']
    sources = []
    for label, root in rows:
        d = report(root)
        state = status(root)
        if label == 'JSON 生成式 LoRA' and root == asap_gen and 'replay' not in root:
            state = '当前有效结果；独立重跑中，完成后覆盖'
        if d:
            assert d['rows'] == 2598
            lines.append(f"| {label} | {d['all']['mae']:.6f} | {d['macro_source_nmae']:.6f} | {state} |")
        else:
            lines.append(f'| {label} | Placeholder | Placeholder | {state} |')
        sources.append(f'- {label}（ASAP）：`{root}`')
    d = read('runs/formal_zero_shot_instruct_20261105_v3/instruct/asap.json')['report']
    lines.append(f"| 官方指令 LLM，零任务微调 | {d['all']['mae']:.6f} | {d['macro_source_nmae']:.6f} | 已完成 |")
    lines += ['', 'MAE 使用各题原生分数；NMAE 先按各题合法分数跨度归一化，再对八题等权平均。两者均保留，不能只挑有利的一项。', '',
              '### 9.2 STS-B＋WMT 英中连续评分专项', '',
              '| 模型 | STS-B MAE ↓ | WMT 英中 MAE ↓ | 两源等权 NMAE ↓ | 状态 |', '|---|---:|---:|---:|---|']
    entries = [
        ('JSON 生成式 LoRA', f'{RECHECK}/continuous_json_replay'),
        ('Kev', 'runs/formal_score_baselines_20261105_v1/continuous_kev'),
        ('独立候选 D2', 'runs/formal_score_numeric_20261105_v1/d2_batched'),
        ('数值 CDF', 'runs/formal_score_numeric_20261105_v1/numeric_cdf'),
        ('R1 CDF＋均值回归', f'{RECHECK}/continuous_r1_cdf_mean'),
        ('R2 纯 CRPS', f'{RECHECK}/continuous_r2_crps'),
        ('N1 数字前缀树＋CE', f'{TRIE}/continuous_n1_trie_ce'),
        ('N2 数字前缀树＋CE/CDF', f'{TRIE}/continuous_n2_trie_ce_cdf'),
        ('T1 数字前缀树＋CE＋数字序列监督', f'{TOKEN}/continuous_token_ce'),
        ('T2 数字前缀树＋CE/CDF＋数字序列监督', f'{TOKEN}/continuous_token_ce_cdf')]
    for label, root in entries:
        d = report(root)
        if d is None:
            lines.append(f'| {label} | Placeholder | Placeholder | Placeholder | {status(root)} |')
            sources.append(f'- {label}（连续评分）：{chr(96)}{root}{chr(96)}')
            continue
        a, b = d['source/stsb'], d['source/wmt_en_zh']
        assert a['rows'] == 1379 and b['rows'] == 1000
        lines.append(f"| {label} | {a['mae']:.6f} | {b['mae']:.6f} | {(a['nmae']+b['nmae'])/2:.6f} | {status(root)} |")
        sources.append(f'- {label}（连续评分）：`{root}`')
    d = read('runs/formal_zero_shot_instruct_20261105_v3/instruct/continuous.json')['report']
    a, b = d['source/stsb'], d['source/wmt_en_zh']
    lines.append(f"| 官方指令 LLM，零任务微调 | {a['mae']:.6f} | {b['mae']:.6f} | {(a['nmae']+b['nmae'])/2:.6f} | 已完成 |")
    lines += ['', '本表只汇总 STS-B/英中，不含德语测试。Kev 复用历史混合训练模型，训练包含英德；其余当前训练模型为 8,129 条。用户接受此历史差异，保留真实训练人口，不声称完全相同数据的因果对照。', '',
              '概率模型统一使用预先固定的完整网格中位数；生成模型使用实际 JSON 数值。期望解码仅存于原始诊断，不按测试结果替换主输出。', '',
              '### 9.3 当前结果的精确来源', '',
              '以下运行的有效指标均取 `evaluations/epoch_1.test_in_domain.json`，并检查完成标记；被重跑替代的成绩不再列入表。', '', *sources, '',
              '- 官方指令 LLM：`runs/formal_zero_shot_instruct_20261105_v3/instruct/{asap,continuous}.json` 的 `report`。']
    return '\n'.join(lines)


def build():
    lines = ['## 8. 新版 D2 与 Kev 数据量敏感实验', '',
             '固定嵌套训练索引、1 epoch、相同候选采样视图；测试始终原生完整候选。200k 复用正式主实验。', '',
             '| 训练量 | 模型 | Choice Acc ↑ | Noul Acc ↑ | Score median MAE ↓ | 状态 |',
             '|---:|---|---:|---:|---:|---|']
    kev = read(f'{CURVES}/kev_curve_current.json')['rows']
    for n in [25, 50, 100, 200]:
        k = kev[f'Kev_{n}k']
        lines.append(f"| {n*1000:,} | Kev | {k['choice_accuracy']:.4f} | {k['noul_accuracy']:.4f} | {k['score_mae']:.4f} | 已完成 |")
        root = f'{CURVES}/d2_{n}k' if n < 200 else 'runs/formal_independent_20261105_v1'
        d = report(root)
        cells = (f"{d['task/choice']['accuracy']:.4f} | {d['task/noul']['accuracy']:.4f} | {d['task/score']['mae']:.4f}" if d else 'Placeholder | Placeholder | Placeholder')
        lines.append(f'| {n*1000:,} | 新版 D2 | {cells} | {status(root)} |')
    lines += ['', f'D2 来源：`{CURVES}` 和正式 200k 主运行；Kev 来源：`runs/formal_independent_curves_20261105_v1/kev_curve_current.json` 的 Kev 条目，已经统一 median。旧 D2 曲线不再展示。', '',
              '## 9. 独立 RQ-Score：训练范围与当前结果', '',
              '**每个方法在两个面板分别从原始 Base 训练一个模型；没有把 ASAP 与 STS-B/WMT 混合成一个统一模型，也没有继承主实验 20 万条 checkpoint。**', '',
              '| 面板 | 训练数据 | Dev | Test |', '|---|---|---:|---:|',
              '| ASAP | 八题合计 7,784 条 | 2,593 | 2,598 |',
              '| 连续评分 | STS-B 4,882＋WMT 英中 3,247＝8,129 条 | 2,500 | STS-B 1,379＋英中 1,000＝2,379 |', '',
              '均为1 epoch、batch64、同源独立划分测试。ASAP合法档位数为4/5/6/11/31/61；STS-B为0:0.1:5全51档，英中为0:1:100全101档，原始小数标签保留。输入不硬截断，评测不采样。', '',
              score_tables(), '', '## 10. 待完成与研究边界', '',
              '- N1/N2 数字前缀树已完成三数据集评测；两个方法均在 STS-B/WMT 优于当前生成式重跑，但 ASAP 原生 MAE 仍较高，尚未达成三数据集同时胜出。N1/N2 的 ASAP 等题 NMAE 优于生成式，不能与原生 MAE 混淆。',
              '- T1/T2 在同一前向中增加精确数字序列监督，ASAP 两组立即运行，连续面板等待 Laya 完整训练及评测释放显存后接续；状态见第9节。',
              '- 本文保留主模型 D2 与 A1–A3 的最终结果；Score 专项不触发三任务重训或新 A4。',
              '- 跨源三任务面板暂缓，待其他实验完成后另行设计；不汇报缺少 Choice 的旧外部源面板。',
              '- 当前单 seed；不声称跨 seed 稳定性或所有消融组件均有收益。', '',
              '## 11. 维护与溯源', '',
              '本文件采用覆盖式更新，不追加带日期的历史结果快照。原始报告、权重、失败与重跑记录保留在各 runs 和执行日志。',
              '维护脚本：`scripts/formal/update_final_record.py`。主表 median 口径、当前 Score 重跑和新版学习曲线统一从原始报告读取。',
              '数字前缀树及辅助监督说明：`docs/experiment_log_20261106_numeric_trie.md`；ASAP新实验说明：`docs/experiment_log_20261106_asap_exploration.md`；生成式复核说明：`docs/experiment_log_20261106_score_recheck.md`。']
    return '\n'.join(lines)+'\n'


def update():
    lock = (PROJECT / 'runs/final_record_update.lock').open('a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        s = DOC.read_text()
        now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
        s = re.sub(r'更新时间：[^\n]+', f'更新时间：{now}（Asia/Shanghai）', s, count=1)
        start, end = '<!-- CURRENT_RESULTS_START -->', '<!-- CURRENT_RESULTS_END -->'
        left = s.index(start)+len(start)
        right = s.index(end)
        s = s[:left]+'\n\n'+build()+'\n'+s[right:]
        tmp = DOC.with_suffix('.md.tmp')
        tmp.write_text(s)
        tmp.replace(DOC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        signatures = tuple((r, (PROJECT/r/'complete.json').exists(), status(r)) for r in WATCH)
        if signatures != previous:
            update()
            previous = signatures
        if not args.watch or all(flag for _, flag, _ in signatures):
            break
        time.sleep(30)
