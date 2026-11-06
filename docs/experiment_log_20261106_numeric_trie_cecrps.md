# Score 默认建议与前缀 CE/CRPS 小对照

## 决策

暂定默认：Decimal-prefix sharing + soft CE/CDF-log，主预测为合法支持上的分布中位数；不默认加 R1 均值回归或 numeric-sequence supervision。本建议仅针对 RQ-Score，不自动更换 R26 主实验或已完成 D2 的训练配方。现有证据不支持所有来源均最佳。

已有小实验来源：runs/pilot_numeric_trie_r1_20261106_v1/results_summary.json。R1相比匹配前缀CE/CDF：ASAP测试MAE 1.226563→1.156250，但开发集跨题NMAE 0.119375→0.125347；STS-B测试MAE 0.471363→0.468629，WMT英中11.830729→11.942708。小样本单种子，配对区间跨0，不足以默认加回归。已检查的测试属于探索性结果，不能称未触碰测试。

本轮仅补一个固定参数的小对照：保留前缀结构和soft CE，将CDF阈值对数损失换成CRPS平方损失。之前纯CRPS使用另一种数值读出且没有此CE混合，不能替代本次对照。CRPS是已有评分规则，不宣称发明新loss。不扫参、不做全量训练。

## 公式

完整合法支持$s_1<\cdots<s_K$，$t_i=(s_i-s_1)/(s_K-s_1)$。保留原始标签$y$；相邻档位线性插值得到$q$，$F_j=\sum_{i\leq j}p_i$，$G_j=\sum_{i\leq j}q_i$，$w_j=t_{j+1}-t_j$。

$$
\mathcal L_{CE}=-\sum_iq_i\log p_i,\quad
\mathcal L_{CDF}=-\alpha\sum_{j=1}^{K-1}w_j[G_j\log F_j+(1-G_j)\log(1-F_j)].
$$

现有默认：

$$
\mathcal L_{default}=0.5\mathcal L_{CE}+0.5\mathcal L_{CDF},\quad\alpha=3.0383850790577127.
$$

本次候选：

$$
R=\sum_{j=1}^{K-1}w_j(F_j-G_j)^2,\quad
\mathcal L_{trial}=0.5\mathcal L_{CE}+\alpha R.
$$

混合实现为0.5CE+0.5×(2αR)。2α仅使F=0.5的单阈值导数尺度与αBCE相同，不代表模型全程梯度范数匹配。按分数间隔归一化，正仿射变换不改变目标；不因增加K累计更多等权项。

非档位标签：R与原始标签CRPS相差仅与标签和支持相关的常数ΣwG(1−G)，参数梯度相同。CPU测试核对能量形式、常数及梯度，不四舍五入标签。不依据测试切换期望或argmax。

## 固定实验

| 面板 | 训练 / Dev / Test | GPU | 步数 |
|---|---|---|---|
| ASAP八题 | 512 / 64 / 128 | 0–1 | 32 |
| STS-B＋WMT英中 | 2048 / 256 / 512 | 6–7 | 128 |

复用data/pilot_numeric_trie_20261106_v1完整记录、顺序、缓存及合法档位；无截断、重采样、Score位置实验。相同fresh Qwen3.5-9B-Base、seed20261101、epoch1、batch16、world2、优化器/调度/微批。已完成对照：runs/pilot_numeric_trie_20261106_v1/{asap_control,continuous_control}，不重复训练。

新根目录runs/pilot_numeric_trie_cecrps_20261106_v1，分支asap_cecrps及continuous_cecrps。为匹配对照，保留因果后缀前向，数字序列loss权重严格0，Score查询在数字前。真实骨干校验prompt-only/原后缀/改后缀一致；不是新增序列监督。初始化LoRA/head/pointer哈希、优化器组须匹配对照。自动Dev/Test，native median；ASAP八题等权NMAE，连续源等权NMAE。

R26 KEV2–5不动；不恢复旧ASAP RLCD，不触发主实验或其他队列。

## 执行记录

项目/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm；Python /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python。

- 准备：python scripts/formal/prepare_numeric_trie_cecrps.py，冻结代码/配置/数据哈希，审计配置差异。
- 持久tmux：python scripts/formal/schedule_research_queue.py --root runs/pilot_numeric_trie_cecrps_20261106_v1 --gpus 0,1；另一路--gpus 6,7。
- 报告：python scripts/formal/report_numeric_trie_cecrps.py --root runs/pilot_numeric_trie_cecrps_20261106_v1 --watch；锁保护覆盖主记录NUMERIC_TRIE_CECRPS_SMALL区块。
- 每步心跳、checkpoint16；各experiment_manifest.json保存resume_command。失败需检查，不擅改冻结来源。

numeric_trie/consistency/pilot/R1/CECRPS共56项CPU测试通过，涵盖能量形式、小数、不等距支持、正仿射不变、适当性、极端logits及梯度、无新参数或额外骨干前向。

实际状态以preparation_verified.json、launch_verified.json、分支status/progress/complete.json和results_summary.json为准。结果Placeholder，不预先改变默认。本地提交与冻结，不进行远程备份。

理论来源：Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation, JASA102(477), DOI10.1198/016214506000001437，§4.2 CRPS/能量形式；本实现采用最小化损失方向。

## Trajectory — 2026-11-06T23:02:22.065780+08:00

已提交代码81a8c2c，冻结168份源文件；56项CPU测试通过。GPU preflight两组通过，LoRA/pointer/head/base配置及可训练参数逐项与已完成对照一致；无新增参数。

- asap_cecrps，GPU 0,1，实际优化器step 1/32，finite loss=1.532390832901001；GPU前置检查峰值20.56GiB。
- continuous_cecrps，GPU 6,7，实际优化器step 35/128，finite loss=1.9874557256698608；GPU前置检查峰值20.33GiB。

下一步：现有持久队列自动完成训练和原生Dev/Test，reporter覆盖本实验区块。尚无本次结果，不提前宣布优于CE/CDF或换默认。曾有两次工具封装语法错误，均在shell执行前失败，已通过独立重发修复；训练本身无失败。
