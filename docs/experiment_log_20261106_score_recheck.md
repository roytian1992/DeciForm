# Score 生成式 LoRA 复核与三面板 loss 补全

## 目的与边界

2026-11-06，用户要求重新验证生成式 LoRA 的结果，并询问 CDF＋回归辅助
与纯 CRPS 在 ASAP、STS-B、WMT 英中三个测试集上的成绩。本轮从原始
Qwen3.5-9B-Base 重新训练生成式 LoRA，同时补齐此前仅做过 ASAP 的两种
数值分布 loss 的连续评分实验。保留旧结果，重跑完成前不把它们判为错误。
主模型独立候选 D2 和 A1–A3 不变；2–3 卡的数据量曲线继续运行。

## 旧结果审计

证据：`runs/score_generative_integrity_20261106_v1/audit.json`，状态 passed。
脚本：`scripts/formal/audit_score_generative.py`。

- 原 ASAP 与连续面板全部 31,311 条 train/dev/test 行重新编码后，与冻结缓存
  的 prompt 和 answer token 完全一致。修改或删除 label 不改变 prompt。
- 两个面板的 train 对 dev/test 的 ID、group ID、规范化正文交集均为零。
- 从原始输出严格解析 JSON、按原始 ID 匹配 gold，独立重算 dev/test 的
  MAE 和等来源 NMAE，均与旧报告一致。检查了完整数量、重复 ID、数值范围；
  不从无效输出中抽取数字，不丢弃失败样本。原测试输出全部有效。
- ASAP 原结果 MAE 0.8006158583525789，八题等权 NMAE 0.08734053000552051。

这支持本地数据、缓存、预测和计分流程的一致性；不是预训练污染审计，
也不能替代独立训练重跑。新生成 worker 在调用 predict 前显式删除 label
及 answer token，只交付 prompt token；旧生成实现本身也只使用 prompt。

## 冻结设计与调度

运行根：`runs/formal_score_recheck_20261106_v1`。
共同设置：fresh base、无父 checkpoint、seed 20261101、1 epoch、global batch 64。
完整文本，无硬截断；数值模型评测完整合法网格，主解码固定为中位数。
生成模型沿用 JSON 提示、greedy、最多 64 个新 token、原严格 parser。

| 运行 | GPU | train/dev/test | updates | 说明 |
|---|---|---|---|---|
| asap_json_replay | 0–1 | 7,784 / 2,593 / 2,598 | 122 | 原 ASAP 数据、种子、优化配置、初始化完全复跑 |
| continuous_r1_cdf_mean | 4–5 | 8,129 / 2,500 / 2,379 | 128 | 数值 CDF＋归一化预测均值 MSE，权重 1 |
| continuous_r2_crps | 6–7 | 同上 | 128 | 纯归一化 CRPS，固定尺度 6.076770158115425 |
| continuous_json_replay | 4–5，R1 后自动接续 | 同上 | 128 | 在当前 STS-B/英中人口上重新训练生成式 LoRA |

连续面板保持当前过滤后的 STS-B/英中数据。历史生成基线训练为 11,457 条，
本次为 8,129 条，因此此分支是当前面板的重新对照，不称完全相同人口的复现。
用户已接受历史英德混合差异，日常对比不因此阻塞。没有新增德语作业。
缓存按 ID 从旧缓存复用，要求当前原始行完全相等；审计为
`data/score_baseline_encodings_stsb_enzh_20261106_v1/audit.json`。

R1/R2 采用与已完成 continuous numeric CDF 相同的架构、完整数值 head
初始化、batch packing、优化器、数据和解码，只替换预先登记的目标。
连续标签原始小数保留；51/101 档位用于完整分布支持，不先把标签四舍五入。
对于插值目标 CDF Q 与预测 CDF F，所用平方项与真实标签 CRPS 相差
一个只由标签决定的常数，梯度完全相同：

$$
\operatorname{CRPS}(p,y)
=\sum_j\Delta t_j(F_j-Q_j)^2
+\sum_j\Delta t_jQ_j(1-Q_j).
$$

32 项测试通过：`tests/test_continuous_score_objectives.py` 和
`tests/test_score_baselines.py`，覆盖分数解析、连续标签及上述等价性/梯度。
原数值架构及 loss 的 25 项既有测试已在前轮通过。新代码单独保存，不修改
正在运行的原 worker、模型或 loss。

## 可复现材料与运行监控

根及各 arm 的 `experiment_manifest.json` 冻结 config、数据、编码、源码
SHA-256、原参考、启动命令；根 `code/` 保存完整依赖源码快照。环境与原运行
相同：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。

2026-11-06 02:41 Asia/Shanghai 启动三个 tmux 队列：
`deciform_score_recheck_g01_20261106`、`deciform_score_recheck_g45_20261106`、
`deciform_score_recheck_g67_20261106`。精确命令记录于根 `launch.json`：
上述 Python 执行 `scripts/formal/schedule_research_queue.py --root <root> --gpus <pair>`。
每项独立执行分布式 preflight，成功后自动训练和完整 dev/test；初始化比对
失败、loss 非有限、worker 异常或超时都会停止并记录 status/alerts。
队列失败不跳过；只允许从本运行自己的 resume.pt 恢复。保活监管继续工作。

## 当前结果

| 指标（越低越好） | CDF＋回归 R1 | 纯 CRPS R2 | 状态 |
|---|---:|---:|---|
| ASAP MAE，2,598 条 | 0.911855273287144 | 0.9314857582755967 | 已完成，上一运行根 |
| ASAP 八题等权 NMAE | 0.10067745514854054 | 0.10257481081069555 | 已完成 |
| STS-B MAE，1,379 条 | 0.5243705583756354 | 0.5245025380710663 | 已完成 |
| WMT 英中 MAE，1,000 条 | 10.324999967575073 | 11.230666624069213 | 已完成 |
| STS-B/英中等来源 NMAE | 0.1040620556754389 | 0.10860358692745278 | 已完成 |

ASAP 证据：`runs/formal_score_objectives_20261106_v1/results_summary.json`。
ASAP 两种新目标改善 pooled MAE，但未改善原数值 CDF 的等权 NMAE；
生成式 LoRA 旧结果仍领先，等待独立重跑确认。

生成式 ASAP/连续面板重跑结果均为 Placeholder。不得把队列启动当作完成，
不得把 ASAP 的 loss 成绩填入另外两个数据集。

### 02:46 — 已核验实际训练

根 `launch_verified.json` 已核对全部冻结代码/config hash、独立运行的初始化
和参考完全一致、无继承权重、preflight passed 及真实 optimizer update。
ASAP 生成式首步已完成；两种连续目标已到 110/128，约 1.65 秒/步。
首步 ASAP 约28.4秒，按旧运行计完整训练＋评测约75–90分钟。
连续目标预计数分钟内完成并自动交接生成式重跑，后者约10–15分钟。

### 02:48 — R1/R2 三面板结果齐备

两种连续目标均完成128步、完整2,500 dev与2,379 test，exit code 0。
训练分别211.27/210.58秒，完整评测38.93/39.54秒；4–5卡队列已自动接上
生成式连续面板预检。精确来源、计数、原始报告hash与各解码诊断见根
`objective_results_summary.json`；上表更新为全部实际结果。

同数值架构的原CDF在STS-B/WMT英中为0.533332/10.050000：R1/R2的
STS-B均小幅改善，WMT英中均变差，等源NMAE也变差。ASAP情况见上。
因此不声称新目标全面优于原CDF。WMT的期望解码诊断R1/R2为
9.084329/9.377413，优于各自median，但没有事后切换主解码规则；
若研究decoder，应在dev上预先选择并统一报告所有参考。
生成式重跑尚未完成，旧生成成绩仍保留为待独立重复确认的参照。
