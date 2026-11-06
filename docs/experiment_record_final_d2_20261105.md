# 新版 D2 全部相关实验记录

更新时间：2026-11-06 10:05（Asia/Shanghai）

本文档只展示当前有效结果：同一模型、同一实验口径仅保留一条结果，重跑完成后覆盖对应单元格；未完成明确写 Placeholder。历史快照、被替代结果和过程轨迹留在原始 runs 与执行日志，不在本文重复展示。主实验与 Score 专项使用不同模型和训练数据，分别列出。

**零样本 LLM 结果已补齐：** 官方 Qwen3.5-9B 指令模型于 2026-11-05
20:28:55 +08:00 完成全部 80,977 次预测。主测试结果见第 7 节，
Score 专项见第 9 节，Choice/Noul 位置变换见第 5 节；三部分均已完成。
“零样本”指未经本项目数据微调，模型本身已有官方指令后训练。

**2026-11-05 研究范围确认：** 多档位 Score 优化作为独立 **RQ-Score**，
不触发三任务 D2 重训、不重做 A1–A3，也不新增 A4。第 9 节保留专项
对照，与主结果分别解释；具体问题和边界见
[Score 专项规划](score_specialist_plan_20261104.md)。

## 1. 实验对象和版本边界

### 新版 D2（当前最终版本）

- 运行目录：`runs/formal_independent_20261105_v1/`
- 代码说明：`docs/experiment_log_20261105_independent_candidates.md`
- 训练完成文件：`complete.json`、`epoch_1.pt`
- 训练初始化：本地 Qwen3.5-9B base，fresh base only；不继承 static-full、Kev 或其他已训练 checkpoint
- 可训练参数：46,304,262
- LoRA：rank 16、alpha 32、dropout 0.05，覆盖 Q/K/V/O、上下投影以及 DeltaNet 投影
- task routing、task-conditioned head geometry、Choice quadratic readout、ordinal Score readout 全部启用
- 训练：20 万条、1 epoch、3,125 个 optimizer updates、global batch 64
- 新版的核心改动：候选项和 query 使用共享前缀后分支的独立编码；候选分支不读取其它候选的 token，输出按语义候选 ID 对齐，因此候选显示顺序不改变对应候选的分数

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

$$
q=F_\theta(P,\texttt{decide}),\qquad
h_i=F_\theta(P,\texttt{opt}\ c_i\ \texttt{/opt}).
$$

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

其它基线的位置敏感性只作参照：

| 模型 | Native Choice reverse flip | Native Choice any-order flip | Native Noul reverse flip | 说明 |
|---|---:|---:|---:|---|
| 新版 D2 | 0 | 0 | 0 | 7 种固定顺序，完整原生候选 |
| Kev | 7.82% | 16.28% | 4.94% | 同一历史审计 |
| 生成式 LoRA | 11.98% | 18.96% | 1.00%（instruction-list） | native Noul 没有候选列表接口 |
| 普通 LoRA | 0.42% | 0.44% | 2.30%（instruction-list） | native Noul 没有候选列表接口 |

来源：`runs/formal_answer_order_20261104_v1/classification_report.json`。这些历史模型与新版位置审计的顺序集合不同，因此该表用于机制诊断，不替代同协议的最终质量比较。

官方 Qwen3.5-9B 指令模型零样本位置实验已于 2026-11-05 20:28 完成，
来源为 `runs/formal_zero_shot_instruct_20261105_v3/instruct/choice_order.json`
和 `noul_order.json`。

| 零样本位置面板 | 题目数 | 顺序数 | 倒序改判率 | 至少一次改判率 | 双方有效倒序配对数 | 有效配对倒序改判率 |
|---|---:|---:|---:|---:|---:|---:|
| Choice，与新版 D2 相同七顺序 | 5,000 | 7 | 21.98% | 42.66% | 4,995 | 21.9019% |
| Choice，与历史基线相同五顺序 | 5,000 | 5 | 21.98% | 39.18% | 4,995 | 21.9019% |
| Noul，提示词中的显式布尔列表 | 5,000 | 2 | 2.54% | 2.54% | 4,988 | 2.4459% |

Choice 两套顺序共享原序和倒序，实际共 10 个命名条件、50,000 次预测，
不能把两行相加成 60,000 次独立预测。Noul 共 10,000 次预测。
新版 D2 的相应原生候选审计均为 0；Noul 两者干预接口不同，不能按同输入
消融解释。七顺序和五顺序 any-order 指标分开比较。双方均无效的输出不
代表正确决策稳定，因此同时保留有效配对口径。

## 6. 新版 D2 A1–A3 架构消融（已全部完成）

运行根目录：`runs/formal_independent_ablations_20261105_v1/`。三组均 fresh base、同 20 万训练记录、同固定候选视图、同 1 epoch、同完整原生 dev/test 评测；只关闭一个结构组件。

| Arm | 关闭组件 | GPUs | 可训练参数 | 当前状态 | Dev/Test 结果 |
|---|---|---|---:|---|---|
| A1 | task routing | 2–3 | 46,113,798 | 训练、完整评测、倒序检查完成 | 见下表 |
| A2 | quadratic QF | 4–5 | 46,025,670 | 训练、完整评测、倒序检查完成 | 见下表 |
| A3 | head geometry | 6–7 | 46,107,654 | 训练、完整评测、倒序检查完成 | 见下表 |

每组的 `preflight_passed.json` 已通过，初始化 hash 与新版 D2 fresh initialization 一致；没有继承新版 D2 的训练权重。原评测因遗漏结果函数导入而退出，训练权重已完整保存。2026-11-05 16:51 从各自最终权重恢复评测，17:10 全部完成，额外训练更新为零；原始失败记录保留。以下读取各组 `evaluation_recovery_20261105_v2/epoch_1.dev.json`、`epoch_1.test_in_domain.json`；恢复详情见 `docs/experiment_log_20261105_final_d2_followup.md`。

| Arm | Dev Choice | Dev Noul | Dev Score MAE | Test Choice | Test Noul | Test Score MAE |
|---|---:|---:|---:|---:|---:|---:|
| Full 新版 D2 | 0.7650 | 0.9020 | 0.4150 | 0.7302 | 0.8880 | 0.4168 |
| A1 无 task routing | 0.7680 | 0.8985 | 0.4135 | 0.7322 | 0.8878 | 0.4170 |
| A2 无 QF | 0.7640 | 0.8990 | 0.4135 | 0.7322 | 0.8892 | 0.4142 |
| A3 无 head geometry | 0.7660 | 0.9010 | 0.4160 | 0.7332 | 0.8866 | 0.4172 |

三组各评完 Dev 6,000、Test 15,000 条。另各完成 Choice 5,000 + Noul 5,000 条原序/倒序配对检查：均 0 翻转，最大概率变化 0。顺序检查与主评测采用不同 packing，主质量指标只使用上表的主评测文件。当前单 seed 结果不支持三个组件均提升新版主指标；A2 的 Test 三项点估计均优于 Full，不能把微小差异直接认定为模块的稳定因果收益。Full 为四卡训练、消融为两卡，存在数值和 RNG 分区差异。

## 7. 同一正式数据上的主要 baselines

训练基线来源：`runs/formal_main_20261101_v2_one_epoch/`。该目录的四个模型
均使用相同 20 万数据、1 epoch 和完整原生候选评测，且均从 fresh base
开始；新版 D2 来源见第 4 节。另列的官方指令模型只做零样本推理，
不属于同训练量对照，来源为 `runs/formal_zero_shot_instruct_20261105_v3/instruct/main.json`。

| Model | Trainable / interface | Test Choice Acc | Test Noul Acc | Test Score MAE |
|---|---|---:|---:|---:|
| 普通 LoRA | Q/V LoRA + ordinary typed head | 0.7086 | 0.8622 | 0.5196 |
| Kev | Kev pointer/candidate encoding + all-target LoRA | 0.7248 | 0.8874 | 0.4256 |
| 生成式 LoRA | JSON teacher-forced generation | 0.7252 | 0.8804 | 0.4354 |
| **新版 D2** | 独立候选分支 + task routing/QF/geometry/ordinal | **0.7302** | **0.8880** | **0.4168** |
| 官方 Qwen3.5-9B zero-shot | 官方指令权重直接生成 JSON；零任务微调 | 0.6612 | 0.7858 | 0.8964 |

概率型基线的 Score 同样使用已冻结的 native median：来源为 `runs/formal_score_distribution_20261104_v1/reference_score_rules/test_in_domain.json`。生成式基线使用实际 JSON 数值。旧 argmax 数值已被覆盖，不与 median 混排。

在当前固定的一轮、20 万条训练条件下，新版 D2 的三项测试点估计均优于
表内普通 LoRA、Kev 和 JSON 生成式 LoRA。零样本官方指令模型是单列的
实用参照，不能把它描述成也用 20 万条训练过的基线。这些结果不是跨 seed
显著性结论；目前没有多 seed 结果。

2026-11-05 用户明确选择**只补官方 Qwen3.5-9B 指令模型**，覆盖主测试集、两套 Score 专项和 Choice/Noul 位置实验。运行目录 `runs/formal_zero_shot_instruct_20261105_v3`；20:28:55 +08:00 全部完成，80,977 次预测，零 optimizer 更新。主表来源为 `instruct/main.json`，15,000 题全部计入；64 个无效输出中 Choice/Noul/Score 分别为 5/58/1，分类按错计，Score 按全范围误差计入 MAE。原 Base 队列已取消且没有产生预测。普通训练基线均用过 20 万条，本项用过本项目训练数据为 **0 条**，但有官方后训练，因此是现成 LLM 的零样本参照，不能混写成同监督量或严格同 Base 权重实验。任务正文和 JSON 解析规则相同；指令版采用官方 chat template、关闭 thinking、输出预算 64 tokens。详见 `docs/experiment_log_20261105_zero_shot_json.md`。

<!-- CURRENT_RESULTS_START -->

## 8. 新版 D2 与 Kev 数据量敏感实验

固定嵌套训练索引、1 epoch、相同候选采样视图；测试始终原生完整候选。200k 复用正式主实验。

| 训练量 | 模型 | Choice Acc ↑ | Noul Acc ↑ | Score median MAE ↓ | 状态 |
|---:|---|---:|---:|---:|---|
| 25,000 | Kev | 0.6832 | 0.8608 | 0.5158 | 已完成 |
| 25,000 | 新版 D2 | 0.6996 | 0.8570 | 0.5080 | 已完成 |
| 50,000 | Kev | 0.7014 | 0.8720 | 0.4676 | 已完成 |
| 50,000 | 新版 D2 | 0.7156 | 0.8636 | 0.4646 | 已完成 |
| 100,000 | Kev | 0.7090 | 0.8792 | 0.4386 | 已完成 |
| 100,000 | 新版 D2 | 0.7240 | 0.8770 | 0.4400 | 已完成 |
| 200,000 | Kev | 0.7248 | 0.8874 | 0.4256 | 已完成 |
| 200,000 | 新版 D2 | 0.7302 | 0.8880 | 0.4168 | 已完成 |

D2 来源：`runs/formal_independent_curves_20261105_v1` 和正式 200k 主运行；Kev 当前数据量结果来源：`runs/formal_independent_curves_20261105_v1/kev_curve_current.json`，已经统一 median。该表只保留新版 D2 与 Kev 的当前数据量曲线。

## 9. 独立 RQ-Score：训练范围与当前结果

**每个方法在两个面板分别从原始 Base 训练一个模型；没有把 ASAP 与 STS-B/WMT 混合成一个统一模型，也没有继承主实验 20 万条 checkpoint。**

| 面板 | 训练数据 | Dev | Test |
|---|---|---:|---:|
| ASAP | 八题合计 7,784 条 | 2,593 | 2,598 |
| 连续评分 | STS-B 4,882＋WMT 英中 3,247＝8,129 条 | 2,500 | STS-B 1,379＋英中 1,000＝2,379 |

均为1 epoch、batch64、同源独立划分测试。ASAP合法档位数为4/5/6/11/31/61；STS-B为0:0.1:5全51档，英中为0:1:100全101档，原始小数标签保留。输入不硬截断，评测不采样。

### 9.1 ASAP 专项

| 模型 | Test MAE ↓ | 八题等权 NMAE ↓ | 状态 |
|---|---:|---:|---|
| JSON 生成式 LoRA | 0.803695 | 0.086374 | 已完成 |
| Kev | 1.054273 | 0.110759 | 已完成 |
| 独立候选 D2 | 0.981524 | 0.103547 | 已完成 |
| 独立候选 CE 对照 | 1.029253 | 0.114263 | 已完成 |
| 数值 CDF | 0.954196 | 0.100579 | 已完成 |
| R1 CDF＋均值回归 | 0.911855 | 0.100677 | 已完成 |
| R2 纯 CRPS | 0.931486 | 0.102575 | 已完成 |
| E1 查询条件证据汇聚＋CDF | 0.980370 | 0.102086 | 已完成 |
| E2 中位数回归＋Laplace 有序分布 | 0.910316 | 0.099332 | 已完成 |
| E3 证据汇聚＋中位数分布 | 0.890685 | 0.097932 | 已完成 |
| 官方指令 LLM，零任务微调 | 2.101232 | 0.241090 | 已完成 |

MAE 使用各题原生分数；NMAE 先按各题合法分数跨度归一化，再对八题等权平均。两者均保留，不能只挑有利的一项。

### 9.2 STS-B＋WMT 英中连续评分专项

| 模型 | STS-B MAE ↓ | WMT 英中 MAE ↓ | 两源等权 NMAE ↓ |
|---|---:|---:|---:|
| JSON 生成式 LoRA | 0.450487 | 15.555333 | 0.122825 |
| Kev | 0.696784 | 12.906000 | 0.134208 |
| 独立候选 D2 | 0.472252 | 10.605333 | 0.100252 |
| 数值 CDF | 0.533332 | 10.050000 | 0.103583 |
| R1 CDF＋均值回归 | 0.524371 | 10.325000 | 0.104062 |
| R2 纯 CRPS | 0.524503 | 11.230667 | 0.108604 |
| 官方指令 LLM，零任务微调 | 0.787721 | 23.735667 | 0.197450 |

本表只汇总 STS-B/英中，不含德语测试。Kev 复用历史混合训练模型，训练包含英德；其余当前训练模型为 8,129 条。用户接受此历史差异，保留真实训练人口，不声称完全相同数据的因果对照。

概率模型统一使用预先固定的完整网格中位数；生成模型使用实际 JSON 数值。期望解码仅存于原始诊断，不按测试结果替换主输出。

### 9.3 当前结果的精确来源

以下运行的有效指标均取 `evaluations/epoch_1.test_in_domain.json`，并检查完成标记；被重跑替代的成绩不再列入表。

- JSON 生成式 LoRA（ASAP）：`runs/formal_score_recheck_20261106_v1/asap_json_replay`
- Kev（ASAP）：`runs/formal_score_baselines_20261105_v1/asap_kev`
- 独立候选 D2（ASAP）：`runs/formal_independent_asap_score_20261105_v5/independent_d2_asap`
- 独立候选 CE 对照（ASAP）：`runs/formal_independent_score_completion_20261105_v1/asap_ce`
- 数值 CDF（ASAP）：`runs/formal_score_numeric_asap_20261105_v1`
- R1 CDF＋均值回归（ASAP）：`runs/formal_score_objectives_20261106_v1/r1_cdf_mean`
- R2 纯 CRPS（ASAP）：`runs/formal_score_objectives_20261106_v1/r2_crps`
- E1 查询条件证据汇聚＋CDF（ASAP）：`runs/formal_asap_exploration_20261106_v1/e1_evidence_cdf`
- E2 中位数回归＋Laplace 有序分布（ASAP）：`runs/formal_asap_exploration_20261106_v1/e2_median_laplace`
- E3 证据汇聚＋中位数分布（ASAP）：`runs/formal_asap_exploration_20261106_v1/e3_evidence_median_laplace`
- JSON 生成式 LoRA（连续评分）：`runs/formal_score_recheck_20261106_v1/continuous_json_replay`
- Kev（连续评分）：`runs/formal_score_baselines_20261105_v1/continuous_kev`
- 独立候选 D2（连续评分）：`runs/formal_score_numeric_20261105_v1/d2_batched`
- 数值 CDF（连续评分）：`runs/formal_score_numeric_20261105_v1/numeric_cdf`
- R1 CDF＋均值回归（连续评分）：`runs/formal_score_recheck_20261106_v1/continuous_r1_cdf_mean`
- R2 纯 CRPS（连续评分）：`runs/formal_score_recheck_20261106_v1/continuous_r2_crps`

- 官方指令 LLM：`runs/formal_zero_shot_instruct_20261105_v3/instruct/{asap,continuous}.json` 的 `report`。

## 10. 待完成与研究边界

- 三组新的 ASAP 结构/目标探索见第9节，运行根与状态随实际完成覆盖更新。
- 本文保留主模型 D2 与 A1–A3 的最终结果；Score 专项不触发三任务重训或新 A4。
- 跨源三任务面板暂缓，待其他实验完成后另行设计；不汇报缺少 Choice 的旧外部源面板。
- 当前单 seed；不声称跨 seed 稳定性或所有消融组件均有收益。

## 11. 维护与溯源

本文件采用覆盖式更新，不追加带日期的历史结果快照。原始报告、权重、失败与重跑记录保留在各 runs 和执行日志。
维护脚本：`scripts/formal/update_final_record.py`。主表 median 口径、当前 Score 重跑和新版学习曲线统一从原始报告读取。
ASAP新实验说明：`docs/experiment_log_20261106_asap_exploration.md`；生成式复核说明：`docs/experiment_log_20261106_score_recheck.md`。

<!-- CURRENT_RESULTS_END -->
