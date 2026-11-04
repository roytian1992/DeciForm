# 2026-11-04 独立连续评分 CE/CDF 对照

## 范围与当前状态

用户授权先用 4–5 卡执行 Score 专项计划，其他任务释放后接续。
S3 soft CE 在 2026-11-04 19:20 Asia/Shanghai 启动两卡 GPU 预检，
已通过并完成训练与全量评测；S4 soft CDF 在 S3 释放同一组 4–5 卡后
自动接续，也已完成训练与全量评测。
ASAP S1/S2 因正文/rubric 未准入暂不启动。当前 D2 full 不变。
Choice/Noul 换序实验在原队列继续，没有添加 Score 换序实验。

## 输入与冻结配置

- 项目：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
- 代码提交：`3b26707`；新增专项代码不修改当前 A3/换序/D2 的运行源码。
- 数据：`data/score_specialist_continuous_20261104_v1/manifest.json`。
- 缓存与长度：同目录 `cache/audit.json`，状态 `passed`。
- 原始 STS-B：`data/score_scales_stsb_20261101_v1/raw/stsbenchmark.tsv.gz`。
- 原始 WMT：`data/typed_source_research_20261101_v1/raw/wmt--wmt20_mlqe_task1/`。
- 冻结架构参考：`runs/formal_score_distribution_20261104_v1/d2_ordinal_main/`。
- 原始 backbone：`../RSI_Research/external/baseline_models/qwen35_9b_base`。
- 新运行：`runs/formal_score_specialist_20261104_v1/`，每组保存 config、
  experiment_manifest、分配卡后的 runtime_manifest，根目录保存代码快照。

| 来源 | 原始 train | 准入 train | dev（全保留） | test（全保留） | 完整输出网格 |
|---|---:|---:|---:|---:|---|
| STS-B | 5,749 | 4,882 | 1,500 | 1,379 | 0–5，步长 0.1，51 点 |
| WMT en-de | 7,000 | 3,328 | 1,000 | 1,000 | 0–100，步长 1，101 点 |
| WMT en-zh | 7,000 | 3,247 | 1,000 | 1,000 | 0–100，步长 1，101 点 |
| 总计 | 19,749 | 11,457 | 3,500 | 3,379 | 均不采样 |

排除训练 8,292 条：8,259 条与 dev/test 共享相连句子/文档组，
33 条重复输入与标签。训练—heldout 组重叠为 0；dev/test 仍共享 534 组，
保留原测试人口且不用 dev 选择 checkpoint。上述计数来源为数据 manifest，
每个排除记录保存于 `exclusions.jsonl`。旧正式面板只审计规范化完整输入的
精确相同项（0），不宣称进行过语义级去重或排除了 backbone 预训练污染。

train/dev/test 原始小数标签数分别 7,476 / 2,728 / 2,493；不取整。
最长编码 train 838、dev 830、test 830 tokens；无截断、无删候选。
新网格是连续标签的数值表示，不是来源天然具有 51/101 个类别。

## 对照与指标

两组使用完全相同的 D2 架构、全新 base 初始化、数据、编码、seed 20261101、
batch 64、两卡、半层 activation checkpointing、1 epoch/180 updates。
完整 head/LoRA/pointer 初值须与原 D2 的 fresh initialization 校验相同，
这只是初始化证明，不加载原 D2 的训练权重。

相邻网格插值目标 q 保证期望等于原始小数 y。S3 使用 `-sum q log p`；
S4 使用 `c * sum w_k BCE(F_k,Q_k)`，c 固定 3.0383850790577127，
网格点标签时严格等于原 D2 loss。两组保留相同结构正则化。
只有 Score 接受数据监督，不能用此实验证明 Choice QF 或多任务路由收益。

每来源以固定网格中位数对原始 y 计算 native MAE；跨来源主汇总为
等来源权重 NMAE。报告均值、argmax、最近网格均值诊断与量化误差下限，
不依据测试结果更换主解码。不同尺度的 pooled raw MAE 只作保存字段，
不能用它比较总体质量。仅有一次 epoch 终点，不进行 dev/test 最佳模型筛选。

## 环境、命令与监控

Conda：`qwen35-train`；Python：
`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
包版本记录在各组 manifest 中。工作目录为项目根。

```bash
python scripts/formal/build_score_specialist_data.py
python scripts/formal/cache_score_specialist.py --data data/score_specialist_continuous_20261104_v1 --workers 8
python -m pytest -q tests/test_score_specialist.py tests/test_score_distribution.py
python scripts/formal/prepare_score_specialist.py
python -u scripts/formal/schedule_score_specialist.py --root runs/formal_score_specialist_20261104_v1
```

实际使用上述绝对 Python 路径。最后一条在 tmux
`deciform_score_specialist_20261104` 中运行；输出 `scheduler.log`。
S3 用 4–5；S4 优先在 A3 成功释放 0–3 后用 0–1，或 Choice 队列成功
释放 6–7 后使用 6–7；S3 更早结束则接用 4–5。每次启动还要求每张卡显存
低于 1,000 MiB，保留现有任务。两组都自动完整网格 dev/test 评测。

根目录 `status.json` 记录排队，组内 `status.json`、`progress.json`、
`preflight*.json`、`train.log`、`loss_diagnostics.jsonl` 记录状态。
异常写 `alerts.json`/`failure.rank*.json`，15 分钟无进度也记警报。
每 25 步保存 `resume.pt`（包含优化器、调度器和每 rank RNG）。
若异常且存在 resume.pt，先核对/停止该组遗留进程，再用相同 Python 执行
`scripts/formal/supervise_score_specialist.py --manifest <该组/runtime_manifest.json> --resume`。
不直接重复启动根调度器；已有 runtime manifest 的组禁止静默重新训练。

评测输出：每组 `evaluations/epoch_1.{dev,test_in_domain}.json`、
对应 `.score_rules.json` 与 rank JSONL；两组完成后生成根 `results_summary.json`。
根结果已生成，状态为 `complete`。S3/S4 test 的等来源 NMAE 为
`0.108133/0.101199`；S4 的等来源 NMAE
相对下降约 6.4%。S4 在 STS-B 和 WMT en-zh 更好，在 WMT en-de 的中位数
MAE 略高，因此不能概括成所有来源均提升。均值解码是诊断项，未替换预注册的
中位数 headline。

## 轨迹与限制

### 2026-11-04 19:20 Asia/Shanghai

已构建、组隔离并冻结连续面板；验证缓存和标签不经取整；新增隔离训练接口、
soft CE/CDF 和调度器，55 项数学/编码/指标测试通过，8 个源码编译检查通过。
环境没有 SciPy，Spearman 使用 NumPy 平均并列秩的相关系数实现并测试，
没有改动正在使用的 Conda 环境。代码已提交并快照，GPU 零更新预检已启动。
未请求远程备份。

当时的下一步为确认 S3 预检和训练，再由队列接续 S4；ASAP 下载和 rubric 准备另行推进。
CDF 的系数来自五档 D2，本次不按测试调系数，预检/训练记录实际梯度尺度。
小数据只跑一轮可能未收敛；不能只给某一组追加 epoch。

### 2026-11-04 19:24 Asia/Shanghai：训练启动核验

S3 两 rank 的 fresh base/LoRA/head/pointer 初值一致且匹配冻结 D2 的初始值。
零更新预检通过，最长完整输入反向传播和完整网格推理正常，峰值 54.7459 GiB。
已观测训练更新完成（详见 `launch_verified.json` 的实时步数），所有冻结源码
校验通过，无 failure/alert；当时 S4 仍按成功释放条件排队，尚无最终测评结果。

### 2026-11-04 20:58 Asia/Shanghai：S3/S4 完成状态核验

S3/S4 实际完成时间分别为 19:47:50 / 20:15:16（各组 status.json）。
两组均完成 180/180 步、11,457 条样本的一轮训练和 dev/test_in_domain
完整评分网格评测，退出码均为 0。S3 训练 1,269.4 秒、评测 228.2 秒；
S4 训练 1,274.3 秒、评测 229.2 秒。测试结果读取自根
`results_summary.json` 及两组 `.score_rules.json`，没有测试集选 checkpoint。
当前 4–5 已释放；ASAP S1/S2 仍等待完整原始输入，D2 full 保持不变。

| 测试指标（中位数解码，越低越好） | S3 soft CE | S4 soft CDF |
|---|---:|---:|
| STS-B MAE，0–5，1,379 条 | 0.526014 | 0.453189 |
| WMT en-de MAE，0–100，1,000 条 | 9.906500 | 10.115167 |
| WMT en-zh MAE，0–100，1,000 条 | 12.013000 | 11.180667 |
| 等来源 NMAE，3,379 条 | 0.108133 | 0.101199 |

总体相对误差下降 6.4124%，尚未完成按组配对 bootstrap，不能据此声称
显著性或跨 seed 稳定性。主结论仍按中位数解码报告。
