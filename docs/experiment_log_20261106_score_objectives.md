# ASAP 数值模块：回归辅助与 CRPS 目标

## 用户授权与研究边界

2026-11-06 用户要求并行尝试两种 Score 优化：一组在 loss 中加入回归机制，
另一组采用有数学依据的替代方案。当前 ASAP 最强已测基线是 JSON 生成式
LoRA，不能只用 D2 自身作为质量门槛。这次固定已完成的单次编码数值模块，
仅改变数据损失，使用空闲的 4–5、6–7 卡。0–3 卡数据量曲线继续原队列。

这两组属于独立 RQ-Score；不替换三任务主 D2、不重训 20 万主模型或 A1–A3，
不新增 A4，不添加德语数据。此次只训练 ASAP 两个新配置，没有新增 seed、
轮数或数据视图。新损失是已知统计目标的实验应用，不声称发明了 MSE 或 CRPS。

## 运行、数据与匹配条件

项目：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
运行根目录：`runs/formal_score_objectives_20261106_v1`。
配置生成器：`scripts/formal/prepare_numeric_objectives.py`。

| 运行 ID | GPU | 数据目标 |
|---|---|---|
| r1_cdf_mean | 4–5 | 原 CDF 对数损失 + 归一化预测均值平方误差，权重 1 |
| r2_crps | 6–7 | 纯归一化 CRPS，固定尺度 2 × 原 D2 alpha |

输入：`data/score_specialist_asap_fold0_20261104_v1`，7,784 train、2,593 dev、
2,598 test。8 个题目的原生完整评分支持集含 4/5/6/11/31/61 档，标签、题目、
作文正文、rubric 和已有划分完全复用；没有硬截断或候选采样。
cache audit SHA-256：
`75ce13db5b8d91848a3802ad8eed32d3c6a5649a9f91163c8d57803c9e65dc6f`。

共同条件：原始 Qwen3.5-9B-Base fresh initialization，`parent_checkpoint=null`；
seed 20261101、1 epoch、batch 64、122 updates、30% OneCycle warmup。
半层 activation checkpointing、padded-token budget 8,192、最多四行微批，
与已有数值模块 ASAP 运行完全一致。所有学习率、参数组、结构正则和可训练
参数量 44,734,978 保持原值。损失按全局样本数聚合，不按微批等权平均。

不从数值模块、D2、Kev、生成式 LoRA 或 static-full 学过的权重继续训练。
监督程序核对基础 LoRA、pointer、原 D2 head 初始化，以及**整个数值 head**
的 fresh hash，避免只有 backbone 相同、head 起点却不同。

结构对照：`runs/formal_score_numeric_asap_20261105_v1`，同一次文本/rubric/query
编码、任务 rank routing、query projection/geometry、32 个自适应宽度概率箱
和边界原子，线性投影到完整合法分数。原实现文件保持不变：
`models_score_research.py`、`numeric_score_distribution.py`。

## 数学定义与可检验假设

令完整分数集合为 $s_0<\dots<s_{K-1}$，跨度 $R=s_{K-1}-s_0$，
$t_i=(s_i-s_0)/R$，真实分数归一化为 $z=(y-s_0)/R$。
模型预测 $p_i$，累积概率为 $F_j=\sum_{i\le j}p_i$；原生整数目标
$Q_j=\mathbf1[y\le s_j]$。固定 $\alpha=3.0383850790577127$。

原始目标是

$$
L_{\mathrm{CDF}}=\alpha\sum_{j=0}^{K-2}(t_{j+1}-t_j)
\operatorname{BCE}(F_j,Q_j).
$$

R1 使用同一分布的期望分数，不增加独立回归 head：

$$
\mu=\sum_{i=0}^{K-1}p_i t_i,\qquad
L_{\mathrm{R1}}=L_{\mathrm{CDF}}+(\mu-z)^2.
$$

平方误差监督条件均值。对无限表达能力的条件分布而言，CDF proper loss 和
均值平方误差在真实条件分布处共同最优，因此两者的总体目标没有人为矛盾。
回归项是**期望的误差平方**，不是 $\sum_i p_i(t_i-z)^2$；后者包含预测
方差惩罚，可能诱导分布过度集中。归一化使不同题目范围可比。权重固定为 1，
不根据本轮开发或测试结果搜索。

这一辅助项估计条件均值；它不等同于直接最小化中位数 MAE。在有限数据和
受限模型下可能无益甚至有害，必须保留原中位数主指标检验。

R2 用原生离散分布的 CRPS 替换 CDF 对数损失：

$$
L_{\mathrm{R2}}=2\alpha\sum_{j=0}^{K-2}(t_{j+1}-t_j)(F_j-Q_j)^2.
$$

尺度 $2\alpha$ 的依据：在单个阈值 $F_j=0.5$ 处，两种损失关于累积概率
的导数一致。这只是预先固定的尺度约定，不保证所有样本、所有训练阶段的
梯度范数一致。R2 不同时加入 CE、CDF-log 或回归项。

对原生支持上的 $Y,Y'\overset{\mathrm{iid}}\sim p$，归一化 CRPS 等价于

$$
\operatorname{CRPS}(p,y)/R
=\mathbb E|Y-y|/R-\tfrac12\mathbb E|Y-Y'|/R.
$$

它把数值误差与分布宽度结合；第二项使目标不同于仅惩罚期望绝对误差。
它是 proper distribution score；点质量预测时退化为归一化绝对误差。
假设是平方累积分布误差在本任务的小数据训练中可能优于对数阈值误差，
这不构成它必定获得更低 MAE 的保证。

理论来源：[Gneiting & Raftery, 2007, Strictly Proper Scoring Rules, Prediction,
and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。
以上两组是本项目基于该统计框架作出的具体设计；来源没有证明它们对 ASAP
或本数值模型更优。两组都保留原结构正则，与数据项分别记录。

## 评测与完整比较表

固定最终第 1 轮 endpoint，自动完整开发/测试；每条 Score 的主输出仍是
原生分数支持集上的预测中位数，主指标为逐题 MAE 和八题等权 NMAE。
期望、argmax、nearest-mean 都沿用既有辅助报告；不能看结果后挑选主 decoder。
ASAP 两组没有额外英文/德文翻译数据，也不更改训练分布。

| ASAP Test，2,598 条 | 原生 MAE ↓ | 八题等权 NMAE ↓ | 状态 |
|---|---:|---:|---|
| JSON 生成式 LoRA | 0.800616 | 0.087341 | 已完成；当前最强参考 |
| 原数值 CDF 模块 | 0.954196 | 0.100579 | 已完成；相同架构直接对照 |
| 独立候选 D2 | 0.981524 | 0.103547 | 已完成 |
| Kev | 1.054273 | 0.110759 | 已完成 |
| R1 CDF + mean regression | Placeholder | Placeholder | 新训练 |
| R2 pure CRPS | Placeholder | Placeholder | 新训练 |

基线的精确来源依次为以下各根的 `evaluations/epoch_1.test_in_domain.json`：

- `runs/formal_score_baselines_20261105_v1/asap_json_sft`
- `runs/formal_score_numeric_asap_20261105_v1`
- `runs/formal_independent_asap_score_20261105_v5/independent_d2_asap`
- `runs/formal_score_baselines_20261105_v1/asap_kev`

这些参考和两组新训练均使用同一 ASAP 原始 backbone/训练人口/一轮预算。
架构、输出接口、batch packing 并非所有基线都相同，效率需报告实测环境。

## 验证、启动与恢复

25 项 CPU 测试通过：`tests/test_numeric_score_objectives.py` 与
`tests/test_numeric_score_distribution.py`。包括 CRPS 的 energy 等价式、点质量
到 MAE 的退化、真实分布处总体梯度为零、仿射分数范围不变性、归一化均值
回归不额外惩罚方差、导数尺度与多个 K 下的有限梯度，以及原数值分布投影。

实际 9B 数值架构的既有完整长样本显存证据位于
`runs/score_execution_preflight_20261105_v1/numeric_asap_budget8192_half`；
两组还分别执行**当前新 loss**的两卡反向传播预检，包括长请求、密集批次、
第一完整全局批次、梯度聚合、诊断及真实预测/指标流程，成功后才训练。
此前四条最长作文同批的 OOM 探测不重试；仍使用已通过验证的 token budget，
完整处理单条最长请求。

新增隔离代码：`numeric_score_objectives.py`、`models_numeric_objectives.py`、
`worker_numeric_objectives.py`、`supervise_numeric_objectives.py`、
`prepare_numeric_objectives.py`，均在 `scripts/formal/`。现有训练/模型源文件
不修改。新 worker 沿用原循环并显式导入 ASAP 评测，正确记录原生整数网格；
不再遗留 `[51,101]` 或 continuous 面板 metadata。

Python：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
各 manifest 冻结环境、输入/代码/config SHA-256，根 `code/` 保存源码快照。
启动命令：该 Python 执行 `scripts/formal/supervise_numeric_objectives.py
--manifest <arm>/experiment_manifest.json`。运行在独立 tmux 会话
`deciform_score_objectives_{r1_cdf_mean,r2_crps}_20261106`。

监督程序持续记录 `status.json`、`progress.json` 和异常/停滞 alerts，训练后
自动评测。断点恢复仅用该运行自己的 `resume.pt`，命令追加 `--resume`；
必须先检查失败原因，不自动混入历史权重。保活监管仍在。
新增 `objective_components.jsonl` 按固定诊断步记录 CDF、均值误差、CRPS
分量，logit-gradient 诊断仅是输出层代理，不能当作 backbone 梯度。

不推送远端，不删除或覆盖任何既有 checkpoint/结果。

## 轨迹

### 2026-11-06 — 用户授权后实现并并行启动

核对了空闲 GPU、原数值模型、已完成的生成式 LoRA 基线和精确数据划分。
选择同架构两种目标，固定权重与 decoder，完成 25 项数学/数值测试。
生成独立配置及代码快照，已在 4–5 和 6–7 卡启动分布式预检，成功后自动训练。
最终启动核验以根目录 `launch_verified.json` 为准；尚无新质量结论。

### 2026-11-06 01:06 Asia/Shanghai — 两组实际训练核验

两组均通过实际两卡 preflight，fresh base、LoRA、pointer、完整数值 head
初始化 hash 完全相同，均完成首个训练更新。首批64条的原始分量逐数相同：
CDF 1.3306317431852221，normalized mean MSE 0.05999904337941042，
normalized CRPS 0.13850183540489525。对应 R1/R2 总损失分别约
1.390631/0.841644，差异来自预先登记的目标组合。首步训练峰值均约
53.72 GiB，约28.7秒/步；无失败文件。证明文件为根 `launch_verified.json`。
预计同架构完整训练及自动 dev/test 共约60–75分钟，最终以实际进度为准。
