# 新版 D2 全部相关实验记录

更新时间：2026-11-05（Asia/Shanghai）

本文档是新版 D2 的统一实验记录。新版 D2 是当前冻结的最终主模型；旧版 D2、普通 LoRA、Kev、生成式 LoRA 和 Score 专项实验均保留在这里作为历史对照或外部补充。没有结果的项目明确写为 `Placeholder`，不能把旧版结果填入新版结果栏。

## 1. 实验对象和版本边界

### 新版 D2（当前最终版本）

- 运行目录：`runs/formal_independent_20261105_v1/`
- 代码说明：`docs/experiment_log_20261105_independent_candidates.md`
- 训练完成文件：`complete.json`、`epoch_1.pt`
- 训练初始化：本地 Qwen3.5-9B base，fresh base only；不继承 static-full、旧 D2、Kev 或其他已训练 checkpoint
- 可训练参数：46,304,262
- LoRA：rank 16、alpha 32、dropout 0.05，覆盖 Q/K/V/O、上下投影以及 DeltaNet 投影
- task routing、task-conditioned head geometry、Choice quadratic readout、ordinal Score readout 全部启用
- 训练：20 万条、1 epoch、3,125 个 optimizer updates、global batch 64
- 新版的核心改动：候选项和 query 使用共享前缀后分支的独立编码；候选分支不读取其它候选的 token，输出按语义候选 ID 对齐，因此候选显示顺序不改变对应候选的分数

### 旧版 D2

- 运行目录：`runs/formal_score_distribution_20261104_v1/d2_ordinal_main/`
- 旧版仍保留 task routing、QF、head geometry 和 ordinal Score loss，但使用旧的完整因果候选编码
- 旧版结果只作为历史对照；不能与新版结果混写为同一个模型

## 2. 固定数据和训练/评测协议

数据目录：`data/formal_200k_20261101_v1/`

| Split | 总量 | Choice | Noul | Score | 候选处理 |
|---|---:|---:|---:|---:|---|
| Train | 200,000 | 50,000 | 75,000 | 75,000 | 固定一视图；Choice 正例加至多 3 个固定负例 |
| Dev | 6,000 | 2,000 | 2,000 | 2,000 | 完整原生候选集 |
| Test in-domain | 15,000 | 5,000 | 5,000 | 5,000 | 完整原生候选集 |
| Test source holdout | 5,000 | — | 2,000 | 3,000 | 已准备，主线正式结果暂未纳入 |

数据完整审计：`data/formal_200k_20261101_v1/cache/audit.json`。训练候选视图、记录顺序和标签均固定；评测阶段绝不做负采样、候选截断或候选数量重采样。没有文本硬截断，超长请求按完整输入处理。

统一设置：seed `20261101`、BF16 backbone、non-reentrant gradient checkpointing、OneCycleLR、30% warmup、半层 checkpoint policy。主指标为 Choice accuracy、Noul accuracy、Score native median MAE，Score MAE 越低越好。

## 3. 新版 D2 的结构和目标函数

令状态和问题前缀为 (P)，候选为 (c_i)。新版编码为

\[
q=F_\theta(P,\texttt{decide}),\qquad
h_i=F_\theta(P,\texttt{opt}\ c_i\ \texttt{/opt}).
\]

每个候选分支从相同的前缀 attention KV、DeltaNet 状态和实际前缀位置开始；前缀只计算一次，分支梯度仍回传到前缀。候选隐藏状态按语义 ID 排序后交给 D2 head。评测时使用整组原生候选，softmax 或 ordinal readout 的归一化支持集就是该题目的完整候选集。

损失保持新版 D2 的正式定义：

- Choice：categorical cross-entropy；
- Noul：二元 Bernoulli NLL/BCE；
- Score：native ordered-CDF ordinal log loss，固定全局 scale `3.0383850790577127`；
- 另有 D2 的结构正则项，task routing 和 head geometry 使用锚定、零均值、有界参数化。

新版没有把 Score 标签改成连续回归标签，也没有给 JSON 生成模型人为构造概率分布。

## 4. 新版 D2 主结果（已完成）

来源：`runs/formal_independent_20261105_v1/evaluations/epoch_1.{dev,test_in_domain}.json`

| Model / split | Choice Acc | Noul Acc | Score MAE | Overall exact/primary summary |
|---|---:|---:|---:|---:|
| 新版 D2 / Dev | 0.7650 | 0.9020 | 0.4150 | macro task accuracy 0.76617 |
| 新版 D2 / Test in-domain | **0.7302** | **0.8880** | **0.4168** | all-row accuracy 0.75107 |

新版 D2 的训练耗时约 20,073 秒，自动 dev/test 评测耗时约 677 秒；这些是本次 4 卡运行记录，不作为模型质量指标。

## 5. 新版 D2 位置敏感性实验（已完成）

运行目录：`runs/formal_independent_position_20261105_v1/`

使用新版 D2 的 `epoch_1.pt`，对 Test in-domain 的完整原生候选集进行以下 7 种显示顺序：原序、倒序、左旋一位、右旋一位、seed 11/29/47 的确定性洗牌。按照现行协议只审计 Choice/Noul；Score 的数值顺序实验不与该分类位置审计混合。

结果：Choice 5,000 条、Noul 5,000 条，共 10,000 条配对样本，在 7 种顺序下均为 **0 次决策翻转**、**最大概率变化 0**；Choice accuracy 始终为 0.7312，Noul accuracy 始终为 0.8878。

完整汇总：`runs/formal_independent_position_20261105_v1/position_sensitivity.json`。

旧版和其它基线的历史位置敏感性只作参照：

| 模型 | Native Choice reverse flip | Native Choice any-order flip | Native Noul reverse flip | 说明 |
|---|---:|---:|---:|---|
| 新版 D2 | 0 | 0 | 0 | 7 种固定顺序，完整原生候选 |
| 旧版 D2 | 7.02% | 14.02% | 6.20% | 历史 `classification_report.json` |
| Kev | 7.82% | 16.28% | 4.94% | 同一历史审计 |
| 生成式 LoRA | 11.98% | 18.96% | 1.00%（instruction-list） | native Noul 没有候选列表接口 |
| 普通 LoRA | 0.42% | 0.44% | 2.30%（instruction-list） | native Noul 没有候选列表接口 |

来源：`runs/formal_answer_order_20261104_v1/classification_report.json`。这些历史模型与新版位置审计的顺序集合不同，因此该表用于机制诊断，不替代同协议的最终质量比较。

## 6. 新版 D2 A1–A3 架构消融（训练中，结果 Placeholder）

运行根目录：`runs/formal_independent_ablations_20261105_v1/`。三组均 fresh base、同 20 万训练记录、同固定候选视图、同 1 epoch、同完整原生 dev/test 评测；只关闭一个结构组件。

| Arm | 关闭组件 | GPUs | 可训练参数 | 当前状态 | Dev/Test 结果 |
|---|---|---|---:|---|---|
| A1 | task routing | 2–3 | 46,113,798 | 训练中，约 860/3125 步 | **Placeholder：待自动评测** |
| A2 | quadratic QF | 4–5 | 46,025,670 | 训练中，约 830/3125 步 | **Placeholder：待自动评测** |
| A3 | head geometry | 6–7 | 46,107,654 | 训练中，约 840/3125 步 | **Placeholder：待自动评测** |

每组的 `preflight_passed.json` 已通过，初始化 hash 与新版 D2 fresh initialization 一致；没有继承新版 D2 的训练权重。完成后只填入各自 `evaluations/epoch_1.dev.json` 和 `evaluations/epoch_1.test_in_domain.json` 的结果。

## 7. 同一正式数据上的主要 baselines

来源：`runs/formal_main_20261101_v2_one_epoch/`。四个模型均使用相同 20 万数据、1 epoch 和完整原生候选评测，且均从 fresh base 开始。

| Model | Trainable / interface | Test Choice Acc | Test Noul Acc | Test Score MAE |
|---|---|---:|---:|---:|
| 普通 LoRA | Q/V LoRA + ordinary typed head | 0.7086 | 0.8622 | 0.7836 |
| Kev | Kev pointer/candidate encoding + all-target LoRA | 0.7248 | 0.8874 | 0.4396 |
| 生成式 LoRA | JSON teacher-forced generation | 0.7252 | 0.8804 | 0.4354 |
| 旧版 TCDL/D2 前身 | 旧候选编码 + task-specific head/loss | 0.7194 | 0.8874 | 0.4376 |
| **新版 D2** | 独立候选分支 + task routing/QF/geometry/ordinal | **0.7302** | **0.8880** | **0.4168** |

该表的结论只能表述为：在当前固定的一轮、20 万条、同一测试协议下，新版 D2 高于这组正式 baselines。它不是跨 seed 显著性结论；目前没有多 seed 结果。

## 8. 旧版 D2 的完整历史结果

### 8.1 旧版 D2 主结果

来源：`runs/formal_score_distribution_20261104_v1/d2_ordinal_main/`。

| Split | Choice Acc | Noul Acc | Score MAE |
|---|---:|---:|---:|
| Dev | 0.7535 | 0.9005 | 0.4130 |
| Test in-domain | 0.7252 | 0.8884 | 0.4180 |

旧版与新版的 Test 差异：Choice `+0.0050`，Noul `-0.0004`，Score MAE `-0.0012`（新版更低）。新版的主要额外收益是候选显示顺序的严格稳定性。

### 8.2 旧版 D2 的 Score 目标对照

来源：`runs/formal_score_distribution_20261104_v1/`。

| 版本 | Test Choice | Test Noul | Test Score MAE | 目标 |
|---|---:|---:|---:|---|
| D1 mean-moment | 0.7218 | 0.8898 | 0.4182 | CE + distribution mean moment |
| D2 ordinal-main | 0.7252 | 0.8884 | **0.4180** | ordered-CDF ordinal log loss |

### 8.3 旧版 D2 的 A1–A3 消融

来源：`runs/formal_d2_ablations_20261104_v1/`。

| Arm | 关闭组件 | Test Choice | Test Noul | Test Score MAE |
|---|---|---:|---:|---:|
| A1 | task routing | 0.7202 | 0.8888 | 0.4218 |
| A2 | quadratic QF | 0.7226 | 0.8886 | 0.4206 |
| A3 | head geometry | 0.7192 | 0.8898 | 0.4192 |
| Full old D2 | 全部启用 | 0.7252 | 0.8884 | 0.4180 |

这些旧版消融不能替代正在运行的新版 A1–A3，因为候选编码拓扑已经改变；这里只保留历史机制证据。

### 8.4 旧版 D2 数据量敏感性

来源：`runs/formal_d2_curves_20261104_v1/`。所有点均为 1 epoch、完整原生测试候选；25k/50k/100k 是旧版 D2 的训练子集，200k 复用旧版 D2 主结果。

| Train rows | Test Choice | Test Noul | Test Score MAE |
|---:|---:|---:|---:|
| 25,000 | 0.6882 | 0.8596 | 0.5080 |
| 50,000 | 0.7034 | 0.8734 | 0.4656 |
| 100,000 | 0.7126 | 0.8802 | 0.4450 |
| 200,000 | 0.7252 | 0.8884 | 0.4180 |

## 9. 独立 Score 专项 baselines（不同数据，不能与主表横比）

这些实验用于检查不同分数范围，不是新版 D2 主数据上的直接 baseline。

来源：`runs/formal_score_baselines_20261105_v1/`。

| Panel | Model | Test MAE | Test NMAE | 说明 |
|---|---|---:|---:|---|
| ASAP author fold 0 | Kev | 1.05427 | 0.110759 | 原生整数等级 |
| ASAP author fold 0 | JSON-generative LoRA | 0.80062 | 0.087341 | 生成数值，允许原生范围内整数 |
| STS-B + WMT continuous | Kev | 7.20263 | 0.124375 | pointer grid |
| STS-B + WMT continuous | JSON-generative LoRA | 7.64234 | 0.113773 | 生成连续数值 |

已有 Score specialist S1–S4 结果也保留在 `runs/formal_score_asap_20261104_v1/` 和 `runs/formal_score_specialist_20261104_v1/`；它们数据源、训练行数和接口不同，暂不放入主 D2 质量排序。

## 10. 待完成实验 Placeholder

以下项目尚未有新版 D2 的正式结果：

1. **A1–A3 新版消融最终结果**：训练完成后自动评测并填表。
2. **新版 D2 数据量曲线**：25k、50k、100k；旧版曲线已有，但不能冒充新版曲线。
3. **新版 D2 source-holdout test**：5,000 条外部来源面板已准备，主线暂未汇报。
4. **新版 D2 与 OpenJev 的同数据直接对照**：当前正式主表包含普通 LoRA、Kev、生成式 LoRA；OpenJev 同协议结果为 Placeholder。
5. **多 seed / 置信区间**：当前所有正式主结果为单 seed `20261101`，暂不做显著性声称。
6. **跨分数档位的新版 D2 专项实验**：已有 Kev/生成式 LoRA Score 专项结果，新版 D2 对应实验为 Placeholder。
7. **校准、概率质量和 score rule 的后续分析**：当前主报告固定使用 native median MAE；NLL、Brier、RPS 和 expected MAE 已保存，但不是主选择指标。

## 11. 复现入口和证据文件

- 新版 D2 配置：`runs/formal_independent_20261105_v1/config.json`
- 新版 D2 manifest：`runs/formal_independent_20261105_v1/experiment_manifest.json`
- 新版 D2 测试报告：`runs/formal_independent_20261105_v1/evaluations/epoch_1.test_in_domain.json`
- 新版位置审计：`runs/formal_independent_position_20261105_v1/position_sensitivity.json`
- 新版消融根 manifest：`runs/formal_independent_ablations_20261105_v1/experiment_manifest.json`
- 旧版 D2 配置和报告：`runs/formal_score_distribution_20261104_v1/d2_ordinal_main/`
- 正式 baselines：`runs/formal_main_20261101_v2_one_epoch/`
- Score baselines：`runs/formal_score_baselines_20261105_v1/`
- 统一实验日志：`docs/experiment_log_20261105_independent_candidates.md`、`docs/experiment_log_20261105_final_d2_followup.md`、`docs/experiment_log_20261105_score_baselines.md`

所有运行目录中的 `config.json`、`experiment_manifest.json`、`status.json`、`complete.json`、初始化 hash、代码 SHA256 和输入 cache SHA256 是实验审计的权威记录；本文档只做集中索引和结果汇总。
