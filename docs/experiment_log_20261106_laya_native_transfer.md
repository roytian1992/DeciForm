# Laya 原生路径上的机制迁移修正 — 2026-11-06

用户要求检查并改进机制迁移。该实验是上一轮迁移的整体修订，保留所有旧结果；不修改已冻结的正式 D2、A1–A3 或 Score 专项。

## 1. 诊断依据与边界

上一轮完整结果在 `runs/formal_laya_mechanisms_20261106_v2`，原 Laya 在 `runs/formal_laya_20261106_v1`。原生 Test15000，Choice/Noul/Score 各5000：

| 布局 | 方法 | Choice Acc | Noul Acc | Score median MAE |
|---|---|---:|---:|---:|
| sequential | 原 Laya | 0.6848 | 0.8066 | 0.5024 |
| sequential | 上一轮迁移 | 0.6758 | 0.8126 | 0.5062 |
| parallel | 原 Laya | 0.6918 | 0.8054 | 0.4992 |
| parallel | 上一轮迁移 | 0.6960 | 0.8106 | 0.5046 |

未发现任务编号、初始化继承或梯度断开错误。迁移方式未能保留源机制的作用位置和优化条件：

1. **全参数微调下共享分支重复参数化。** 上一版用 `W + 2B(I+R_t)A`，其中共享 `2BA` 可吸收到本已全量训练的 W，增加计算但不扩展共享线性函数的表达集合。任务差异部分仍有意义。
2. **几何变换放在旁路。** 原 D2 在主要评分路径内应用任务几何；上一轮保留原 Laya scorer，再另加一个 bilinear scorer。其几何项与 QF 的直接影响很小。
3. **模块优化强度不同。** 原 D2 `runs/formal_independent_20261105_v1/optimizer_groups.json` 的 residual/router LR 为1e-3；上一轮迁移统一使用 head LR1e-4，低了10倍，且未移植任务矩阵结构正则。
4. **迁移范围有限。** 上一轮迁移了 A1–A3 风格模块，没有移植 D2 的独立候选编码和 ordinal CDF loss。因此不能把它称作完整 D2 在 ModernBERT 上的等价实现。

只读 Dev6000 诊断保存在 `runs/laya_transfer_audit_20261106_v1/{sequential,parallel}.json`；脚本及当时副本为 `scripts/formal/audit_laya_transfer_readout.py` 和该目录的 `code/`。在 parallel 的 Choice 上，每条候选分数极差的均值分别为原 scorer 8.84162、共享 bilinear 0.258696、直接几何项0.015971、QF 0.002723。推理时移除 QF，整体 Choice Acc 仍为0.727；移除整个新增 readout 后为0.725。**这是固定已训练权重的干预，不是重新训练消融，不能据此否定模块对训练过程的影响。** 输出 `no_geometry` 只移除了 bilinear 的直接几何项，未移除几何对 QF 的间接影响。

诊断脚本也计算过 Score 反序，但不使用或汇报这些结果；正式位置实验仅 Choice/Noul。新的 worker 在预检和正式评测中均保持 Score 原序。当前诊断脚本也已修正后续运行范围；当时实际运行的原始源码副本和结果保留在诊断目录，未重算或覆盖。

额外40条固定 Dev 分类样本的 parallel 精度探针记录于 `precision_probe.json`：BF16 最大概率变化0.021286，FP32约1.01e-6，均无改判。这表明该小样本上的变化主要受数值精度影响；不能推断全部 Test 改判均由数值误差造成。parallel 仍有共享 token 的间接集合交互，并非候选独立编码。

## 2. 修正后的数学结构

保留 Laya 的 parallel 布局、原编码器、原 scorer、全参数微调与 RLCD 数据目标。只运行一个修正版 parallel。

### 编码器：只建模任务差异

$$
W_{j,t}=W_j+\frac{\alpha}{r}B_jR_{j,t}A_j,\qquad r=16,\quad\alpha=32.
$$

共享适配由 W 学习；低秩分支只表达任务差异。任务顺序为 choice/score/noul，权重为0.25/0.375/0.375。每个投影的 raw maps U 独立中心化和约束：

$$
\widetilde U_{j,t}=U_{j,t}-\sum_s\pi_sU_{j,s},\qquad
R_{j,t}=\frac{0.5\widetilde U_{j,t}}{\max(1,\max_s\|\widetilde U_{j,s}\|_F)}.
$$

A/B 非零正交初始化，R 初始为零，因此初始分支严格为零且 R 在第一次反向传播就有梯度。A/B 在 R 非零后学习；不声称所有因子在第一步都应有非零梯度。作用位置仍为112个 QKV/O/MLP 输入输出投影。

### Head：在原 scorer 内应用任务坐标变换

固定正交行基 P（256×1024），保持其为 checkpoint buffer。对原 scorer 的归一化输入应用任务变换：

$$
n_i=\operatorname{LN}(h_i),\qquad
\widetilde n_i=n_i+P^\top H_tPn_i,\qquad PP^\top=I.
$$

H 使用同样的中心化和0.5范数约束。因此变换的扰动谱范数至多0.5，恒等加扰动的奇异值介于0.5和1.5之间。该约束针对 readout 坐标变换，不是对整个编码器分支的全局范数保证。

原 Laya 的 Linear–GELU–Linear scorer 接收变换后的输入；不再叠加第二个 bilinear scorer。H 初始为零，原评分函数完全保留，但 H 可以从第一步收到数据梯度。固定子空间减少可训练参数，并由可训练的编码器与 scorer 学习其坐标。

### Choice QF

$$
z_i=\operatorname{LN}(P_q\widetilde n_i-V_q\operatorname{LN}(h_{\mathrm{CLS}})),\qquad
\phi_i=\frac{z_i\odot z_i-\mathbf1}{\sqrt2},
$$

$$
s_i=\operatorname{scorer}_{\mathrm{MLP}}(\widetilde n_i)
 +\mathbf1[t=\mathrm{choice}]\,4\tanh\left(\frac{a^\top\phi_i}{4\sqrt{64}}\right).
$$

a 初始为零；中心化平方特征使用标准高斯平方特征的方差归一化约定，并不假定学习后的特征实际上服从高斯。CLS 仍是原 Laya 的集合条件表示，没有候选独立性承诺。

## 3. 损失与优化

$$
\mathcal L=\mathcal L_{\mathrm{RLCD}}+10^{-3}\left[
\frac1{3J}\sum_{j,t}\|\widetilde U_{j,t}\|_F^2+
\frac13\sum_t\|\widetilde H_t\|_F^2\right].
$$

RLCD、sigma0.4、samples4、spherical权重0.75、RPS权重1均保持原版。新增的是任务矩阵结构正则，不应称整个 loss 完全不变。原 RLCD 的 advantage normalization 仍在本地 microbatch 中进行，和原对照一致。

| 参数 | 初始 LR | weight decay |
|---|---:|---:|
| 原编码器与新增 A/B | 2.5e-5 | 0.01 |
| 原 head | 1e-4 | 0.01 |
| 编码器任务矩阵、head任务矩阵 | 1e-3 | 0（显式结构正则） |
| Choice QF | 1e-3 | 0.01 |

保留 cosine 到1e-6、clip1、原 batch/packing。该修订同时改变接入方式、学习率和正则，最终结果只能支持整套方案对比，不能隔离某一项的因果收益。

## 4. 固定协议、文件与启动

- 运行根：`runs/formal_laya_native_transfer_20261106_v1`，parallel 使用 GPU4–5。
- 同一冻结200000条训练视图，seed20261101，1epoch，global batch64，3125更新。每个样本负选项视图与原 Laya 完全一致。
- 从相同原始 ModernBERT-large 和随机原 head 初始化；不继承已训练 checkpoint。校验 encoder/head 初始化 SHA256，新增初始化保留原 RNG 状态。
- Dev6000/Test15000，全原生候选、全文，不截断。Choice/Noul accuracy、Score原生median MAE。
- 训练后自动运行七种顺序的 Choice/Noul10000；Score不置换。不同顺序复用相同原始完整行分批，以减少改变 batch shape 带来的数值混杂。Score在这些 batch 中保持原序但不计入位置报告。
- 原基线位置实验曾以分类子集重新分批，新实现修正这一点；两者微小改判率不能不加说明地视作纯架构差异。质量主指标与原版使用同样完整 original 分批。
- 原模型/数据/cache哈希继承自原 Laya manifest；本次代码/config/源码副本冻结在新 root。所有原始日志、失败记录与上一轮结果保留。

关键文件：`scripts/formal/laya_native_transfer.py`、`worker_laya_native_transfer.py`、`prepare_laya_native_transfer.py`、`report_laya_native_transfer.py`；测试 `tests/test_laya_native_transfer.py`。保存模型需通过新模块的 `load_transfer_checkpoint` 加载，原 upstream loader 不包含新增结构。

```bash
PYTHON=/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python
$PYTHON scripts/formal/supervise_laya.py --manifest runs/formal_laya_native_transfer_20261106_v1/parallel/experiment_manifest.json
$PYTHON scripts/formal/report_laya_native_transfer.py --root runs/formal_laya_native_transfer_20261106_v1 --watch
```

具体 tmux shell 命令、时间、日志路径见 `launch_proof.json`。中断恢复仅用该 arm 自身 `resume.pt` 和 manifest 的 `resume_command`。每10步进度、周期checkpoint、10分钟停滞告警、30分钟无进展终止保留现场。报告器只更新主记录 `LAYA_NATIVE_TRANSFER` 区块并使用共享文件锁。

## 5. 执行记录

2026-11-06 16:27 Asia/Shanghai：11项 CPU 测试通过，覆盖初始输出/RNG/原权重保留、任务梯度、checkpoint反传、变量K、非零模块下 parallel 置换等变、导出重载、参数组与Choice作用范围。语法检查通过。源码/config冻结后在持久 tmux 启动2卡零更新预检；通过后 supervisor 自动开始训练，再进行全量评测。实际训练进展、GPU预检证据后续补入；此时不声称已获得新结果。

2026-11-06 16:34 Asia/Shanghai：实际双卡预检通过，三任务初始 logits 与 act 输出差异均为0；所有最长训练样本反传、完整最大K及最长评测输入均通过。首次任务路由、head任务矩阵、QF输出梯度范数分别为0.03855、0.15859、0.13734，均非零且有限。正式训练超过170/3125步，已处理10880条；最近100步约2.08秒/步，纯训练预计再约1.7小时，另计checkpoint保存与自动评测。该时间是早期估计，不是完成承诺。新增参数7,610,432，总参数428,904,259，可训练428,640,321；比上一版少475,136个新增参数。训练峰值显存约9.24GiB/卡。校验所有冻结源码与配置哈希仍一致；证据保存在 `training_started_verified.json`、`parallel/preflight.json`、`parallel/optimizer_groups.json` 和逐步日志。主记录第14节已加入待出结果占位，报告器持续运行。
