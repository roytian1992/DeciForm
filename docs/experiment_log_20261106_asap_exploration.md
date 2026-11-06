# ASAP 证据汇聚与条件中位数研究

## 授权与执行范围

2026-11-06，用户要求覆盖式整理统一实验记录，并大胆探索 ASAP 任务的优化，
用空闲卡并行比较。随后确认2–3卡数据量实验已完成，授权继续推进。
本轮是 **ASAP 专项**：每组在同一原始 Base 上独立训练，只有 ASAP 八题的
7,784 条训练记录；不混入 STS-B/WMT，也不继承主实验或其他已训练模型。
开发2,593、测试2,598条，固定1 epoch/122更新/global batch64/seed20261101。
完整作文、rubric和问题，不硬截断；完整原生合法评分档位与中位数主输出。
三任务主 D2 与 A1–A3 保持冻结。

根目录：`runs/formal_asap_exploration_20261106_v1`。

| 运行 | GPU | 变化 | 可训练参数 |
|---|---|---|---:|
| E1 `e1_evidence_cdf` | 4–5 | 全前缀查询条件证据汇聚；原数值CDF与原CDF-log目标 | 45,128,451 |
| E2 `e2_median_laplace` | 6–7 | 位置/尺度Laplace评分分布＋显式中位数MAE监督 | 44,730,818 |
| E3 `e3_evidence_median_laplace` | 2–3 | 组合E1与E2 | 45,124,291 |

参考数值CDF为44,734,978可训练参数。所有组保持同样宽覆盖LoRA、router、
query投影、head geometry、训练顺序、微批策略和学习率。E1改变表示，
E2同时改变输出分布族和目标，E3检查组合；这是探索矩阵，不能把E2的变化
单独归因于loss。模型和超参数在看到新结果前冻结。

## 问题与数学动机

### E1：单个末尾query状态是否丢失长作文中的评分证据？

当前数值CDF只读末尾query隐藏状态。E1复用已有一次完整backbone前向的
所有有效prefix隐藏状态，不额外执行第二遍LLM。query依赖全文、问题与rubric，
四个可学习查询从完整prefix中抽取不同摘要：

$$
a_{jt}=\operatorname{softmax}_t\left(
\frac{(W_j\operatorname{LN}(q)+s_j)^\top V\operatorname{LN}(h_t)}{\sqrt{64}}
\right),\qquad
e_j=\sum_t a_{jt}V\operatorname{LN}(h_t).
$$

$$
q'=q+\sigma(g)W_o[e_1;e_2;e_3;e_4],\qquad \sigma(g_0)=0.1.
$$

池化覆盖全部有效prefix token，包含作文和rubric，排除padding及末尾query。
没有取top-k文本、额外候选、滑窗截断或新增标签。四个摘要不宣称对应四个
人工评分维度；attention权重也不自动等于可靠解释。目标保持原CDF-log，
便于检查表示改动的效果。额外计算随文本长度线性增长。

依据：已有AES研究使用词/句attention聚合长文评分证据，如
[Dong et al., 2017](https://aclanthology.org/K17-1017/)；
[Zhang and Litman, 2018](https://aclanthology.org/W18-0549/)研究source-dependent
essay scoring中的co-attention。这里是将相关归纳偏置用于当前冻结骨干和
轻量评分接口的假设，不把attention pooling本身宣称为新发明。

### E2：直接学习MAE对应的条件中位数，约束自由分布的估计方差

原数值模块每条输出32个bin宽与34个mass参数。ASAP只有7,784条训练记录，
自由分布是否增加样本需求是待验证假设。E2用共享特征预测归一化位置和尺度：

$$
\mu(x)=\sigma(f_\mu(q)),\qquad
b(x)=0.01+0.49\sigma(f_b(q)).
$$

潜变量服从位置为中位数的Laplace族。对完整归一化原生档位
$t_0<\cdots<t_{K-1}$，相邻档位中点形成区间，首尾吸收无限尾部：

$$
c_j=(t_j+t_{j+1})/2,\qquad
p_j=F_{\mu,b}(c_j)-F_{\mu,b}(c_{j-1}),
\quad c_{-1}=-\infty,\ c_{K-1}=+\infty.
$$

因此每个合法档位都有概率，预测中位数对应最接近$\mu$的原生档位；
输出随合法评分范围变化，不依赖固定类别数。训练目标为：

$$
\mathcal L_{E2}=\mathcal L_{\mathrm{CDF-log}}(p,y)
+|\mu-y_{\mathrm{normalized}}|.
$$

绝对误差的总体最优点是条件中位数，直接对齐MAE；CDF-log同时学习概率的
阈值事件。这是**中位数回归**，不同于此前R1的预测均值平方误差。
尺度保留每条作文的分散程度，但强制位置/尺度族会牺牲分布形状灵活性；
如果实际条件分布不对称或复杂，可能不如自由CDF。不会保证涨分。
这是连续latent Laplace加原生区间离散化，不声称离散端点质量一定单峰。

有序概率输出的统计背景见
[Vargas et al., Cumulative link models for deep ordinal classification](https://arxiv.org/abs/1905.13392)。
E2是该类有序分布建模思想与MAE目标的具体探索，不声称组合目标在任意
错设分布下仍是无偏的全分布估计。E3在E2前加入E1池化，检验证据表示与
输出约束是否互补。

## 公平性、验证与监控

复用 `data/score_specialist_asap_fold0_20261104_v1` 全部数据/编码与SHA-256，
对照 `runs/formal_score_numeric_asap_20261105_v1`。新增模块前的完整数值head
hash、原始D2 head hash、LoRA/pointer/base hash均由supervisor检查；新增模块
随机初始化，所有LoRA B初始为零。E1/E3的pool、E2/E3的输出头分别使用
独立fork RNG保持相同初始化。输出头共享投影复用同一**未训练**初始化。

27项CPU测试通过：`tests/test_asap_research_modules.py`、
`tests/test_numeric_score_distribution.py`。覆盖完整支持的有限概率、概率
归一化、任意原生范围下中位数与最近档位一致、仿射范围变换、边界CDF梯度、
padding不影响结果、逐行/分批一致、池化梯度以及回归/尺度分支梯度。

每组还必须通过真实两卡预检：正常分布式反向传播、完整最长训练作文、
开发/测试最长输入、密集批次和第一完整global batch；零优化更新，显存
上限64GiB。预检失败会停止而不是截断/跳过样本。训练仍为半层checkpoint、
8192 padded-token预算、最多4行微批。保存各组真实吞吐和峰值显存。
输出logit诊断只衡量CDF部分，不能冒充中位数辅助或backbone梯度。

Python：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
新增模块/worker/supervisor/准备脚本位于`scripts/formal/*asap_exploration*`
及`asap_research_modules.py`。根/各arm manifest记录代码/config/数据hash、
依赖源码快照、初始化、环境和启动/恢复命令；没有改动原运行的冻结源文件。

03:17:57 +08:00已启动三条tmux队列
`deciform_asap_explore_g{45,67,23}_20261106`；精确命令见根`launch.json`。
每组自动preflight→train→完整dev/test。定期保存本运行resume.pt，失败需先
查看原因，仅能恢复本运行，不借用其他训练权重。status/alerts记录异常与
15分钟无进展；GPU保活监管继续。

## 覆盖式统一记录

`docs/experiment_record_final_d2_20261105.md`已经删除被替代版本与追加式
状态快照，Score专项每个方法只保留一个有效结果，明确两套独立训练人口。
修正主表普通LoRA/Kev的Score为已冻结median口径0.5196/0.4256；此前该表
错误混入旧argmax值，原始概率重放报告是权威依据。

2–3卡100k最终结果已覆盖填入：Choice 0.7240、Noul 0.8770、Score median
MAE 0.4400；1563步/100,000访问，训练14909.00秒、完整评测746.61秒。
25k/50k/100k/200k的新版D2与Kev曲线已齐，不保留旧D2曲线重复行。

`scripts/formal/update_final_record.py --watch`在独立tmux运行，完成标记或
阶段变化时覆盖相应表格，不追加历史结果。ASAP生成式复跑完成后自动替换
现有生成式单元格，E1/E2/E3完成后替换Placeholder。原始报告、checkpoint和
旧执行日志保留溯源，不删除。与人工修改并存时只改带标记的当前结果区域。

## 完成结果

2026-11-06 10:05，三组均完成122步、7,784条训练访问、2,593 dev和
2,598 test，exit code 0；ASAP生成式LoRA独立复跑也完成，2,598条输出全部
有效。固定主输出为完整原生档位的median：

| 运行 | Test MAE ↓ | 八题等权 NMAE ↓ | 训练/评测秒 |
|---|---:|---:|---:|
| E1 evidence＋CDF | 0.980370 | 0.102086 | 3268.20 / 636.89 |
| E2 median＋Laplace | 0.910316 | 0.099332 | 3228.19 / 626.81 |
| E3 evidence＋median＋Laplace | **0.890685** | **0.097932** | 3238.21 / 627.92 |
| JSON生成式LoRA（新复跑） | **0.803695** | **0.086374** | 3164.31 / 1303.63 |

E3相对原数值CDF（0.954196/0.100579）、R1 CDF＋均值回归
（0.911855/0.100677）和R2纯CRPS（0.931486/0.102575）均改善，但仍未超过
生成式LoRA。E1单独增加全文汇聚没有改善；E2直接对齐中位数更有效，E3
显示证据表示与中位数分布存在互补。以上为单seed结果，不作显著性结论。

ASAP生成式重跑的证据为
`runs/formal_score_recheck_20261106_v1/asap_json_replay/`；三组探索结果为
`runs/formal_asap_exploration_20261106_v1/{e1_evidence_cdf,e2_median_laplace,e3_evidence_median_laplace}`。
所有训练和自动测评已结束，GPU 0–7 当前空闲，保活监管仍在运行。
