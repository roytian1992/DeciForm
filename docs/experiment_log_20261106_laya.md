# Laya 同数据重训与候选位置对照

## 目的和授权

2026-11-06 用户授权在 GPUs 4–7 下载 Laya、使用原来的 200,000 条训练并比较效果与位置变化。
本次实现两个上游布局：sequential（4–5）及 parallel（6–7）。这是补充跨骨干的决策模型基线，
不是修改已经冻结的 D2、A1–A3 或 RQ-Score。未来双向迁移先分析，不自动启动新矩阵。

## 固定输入

- 根目录：`runs/formal_laya_20261106_v1/`。
- 上游：<https://github.com/NandhaKishorM/laya>，下载并固定 PyPI **0.4.26** 源码。
- 原始 wheel：`vendor/laya-0.4.26-py3-none-any.whl`，SHA256
  `0c3bf5416187b6fdf74706c80b95a9b8e90ce90f39e1f4be818ed936aa31e4bf`。
- 本地解包：`vendor/laya_release/`；上游 `DecisionModel`、`parallel_layout`、`rlcd_loss` 未修改。
- 编码器：`answerdotai/ModernBERT-large`，revision
  `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`，本地 `base/`。
- 原始权重 SHA256：`44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f`。
- **没有下载或继承已训练 Laya decision checkpoint**，也不继承 D2、Kev、static-full。
- 数据：`data/formal_200k_20261101_v1/cache/{train,dev,test_in_domain}.views.jsonl`。
- 训练原文件 SHA256：`5fe588cf4deb75588e910868987163ff51dafaecad21cee8af6e6142a31080b9`。
- 完整输入、源码、配置、版本与校验和：`inputs.json`、各 arm 的 `experiment_manifest.json`。

| Split | 总条数 | Choice | Noul | Score | 候选 |
|---|---:|---:|---:|---:|---|
| Train | 200,000 | 50,000 | 75,000 | 75,000 | 原固定视图；Choice 正例 + 至多 3 个负例 |
| Dev | 6,000 | 2,000 | 2,000 | 2,000 | 原生完整候选 |
| Test | 15,000 | 5,000 | 5,000 | 5,000 | 原生完整候选 |

候选没有重新采样，没有新建 split，也没有为 Laya 另抽校准子集。校准温度为 1，
headline 使用与本地概率模型相同的未事后校准口径；不能将此实验结果称为发布权重自带校准效果。

## 模型、训练与长度处理

ModernBERT-large 双向编码器，Laya 两层 Transformer head、task embedding、marker scorer。
总参数 421,293,827，可训练 421,029,889；无标签的 act/escalation head 保留但冻结。
两组初始 encoder/head hash 完全相同，详见 `launch_verified.json`。

复用上游目标：soft CE 加 Gaussian-logit REINFORCE；4 个噪声样本、sigma 0.4、
spherical 权重 0.75、Score RPS 权重 1。由于只有一个 epoch，上游按 epoch 退火的 sigma 保持 0.4。
RPS 使用有序支持上的累计概率平方差。这不是 D2 的有序 CDF-log 目标。

- 一轮、3,125 updates、global batch 64、seed 20261101；顺序与原正式实验的 shuffle 一致。
- 上游 AdamW，encoder LR 2.5e-5、head LR 1e-4、weight decay 0.01、gradient clip 1。
- 上游 CosineAnnealingLR，末端 1e-6；与 D2 OneCycle 配置不同，明确作为本方法训练配置记录。
- 每卡最多 8 rows，padded token budget 16,384；每个 microbatch 按实际条数 / global batch 加权，
  各 rank 梯度求和一次。上游 RLCD advantage 的标准差仍按 microbatch 计算，故该正则项本身仍依赖分组。
- FP32 master weights + BF16 autocast，encoder 与 head 均 non-reentrant checkpoint。
- 每 100 steps 原子保存 optimizer、scheduler、各 rank RNG 和完整模型，可精确续训。

SDK 默认会限制每候选 48 tokens、候选头预算和 state 长度；本地适配层
`scripts/formal/laya_adapter.py` 使用同一模板但保留全部 token，并且记录完整支持。
实际分词没有丢弃任何一条，也没有裁短正文、指令或选项：

| Split | 最大 token 数 | 超过原始 8,192 的条数 |
|---|---:|---:|
| Train | 9,204 | 4 |
| Dev | 9,662 | 1 |
| Test | 9,512 | 2 |

ModernBERT 当前实现按位置计算 RoPE，可完整处理这些输入；未改原始频率或重写文本。
这是少量输入的长度外推，不声称其超过预训练长度后的质量已经得到保证。
最长训练输入反向、最长 dev/test 前向均已在实际显卡预检通过。
导出的参数保留 FP32 master 数值；后续复现须使用本适配层，不能换回 SDK 的默认截断路径。

## 自动评测与输出

训练结束依次测完整 Dev、Test，并对 Test 中 Choice/Noul 各 5,000 条执行七种顺序：
original、reverse、rotate_left_1、rotate_right_1、shuffle_11/29/47。
种子与每条 ID 的映射规则复用最终 D2 的位置审计，任何条件都保留完整候选。
输出概率先按语义 ID 还原，再计算 accuracy、flip rate、any-order flip 和概率变化。
Score 不加入分类改判率；质量主指标为 native median MAE，同时保留期望值、argmax 和 nearest-mean。

- 质量原始输出：各 arm `evaluations/{dev,test_in_domain}.original.rank*.jsonl`。
- 质量统计：各 arm `evaluations/{dev,test_in_domain}.original.json`。
- Score 解码：各 arm `evaluations/{dev,test_in_domain}.score_rules.json`。
- 位置：各 arm `position_sensitivity.json` 与各条件完整预测文件。
- 总汇：`results_summary.json`；维护脚本 `scripts/formal/report_laya.py --watch`。
- 全部当前结果统一写入 `docs/experiment_record_final_d2_20261105.md` 第 12 节，占位单元格完成后覆盖。

## 环境、运行与恢复

Conda：`qwen35-train`；Python：
`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
精确包版本与环境变量在各 manifest。运行目录为 DeciForm 项目根。

```bash
python scripts/formal/supervise_laya.py --manifest runs/formal_laya_20261106_v1/sequential/experiment_manifest.json
python scripts/formal/supervise_laya.py --manifest runs/formal_laya_20261106_v1/parallel/experiment_manifest.json
```

命令在上述 Conda 环境执行；manifest 固定实际解释器路径。tmux sessions：
`deciform_laya_sequential_20261106`、`deciform_laya_parallel_20261106`。
恢复在同一命令末尾增加 `--resume`；只恢复本次运行的 `resume.pt`。
监管器记录各 rank 心跳；10 分钟无进度产生 alerts，30 分钟无进度终止卡死进程并保留恢复状态。
失败写入 `status.json`、`alerts.json`、`failure.rank*.json`，不伪装成功或自动更改输入。

## 后续双向迁移：待本次结果后决定

本次两个 Laya 布局只回答相同监督量下的跨骨干质量与顺序效应，不能隔离骨干预训练差异。
现阶段可研究的可迁移机制为：

1. **D2 到 ModernBERT-large**：先保留 ModernBERT 的原生 attention，考虑迁移 task-conditioned
   readout geometry、Choice QF、Score CDF-log，以及适合 encoder 的 task-conditioned adapter。
   ModernBERT 的投影命名/融合方式和 Qwen 不同，不能直接复制 DeltaNet 的适配位置。
   若迁移 D2 独立候选思想，需要另外定义 query 与候选隔离；上游 parallel 是排列等变，
   仍允许通过共享正文和后续 head 间接交互，不能等同于 D2 独立候选编码。
2. **Laya 到 Qwen3.5-9B**：RLCD 和概率评分目标可以接入 Qwen 决策 logits；这不要求改变 backbone。
   两层全序列 bidirectional head 可以作为独立模块研究，需实测新增成本。
   parallel 的 encoder mask 不可直接套在 Qwen 的 DeltaNet 上：DeltaNet 有顺序递归状态，
   需要候选状态分支/重置才能实现相应独立性，单独重置 position ID 不足以保证。

双向迁移必须在各自 backbone 内设置对应基线，固定数据和解码后才能归因模块收益。
以上是分析方向，**尚未实现或训练迁移模型**；不把潜在可迁移性写成已验证结果。

## 轨迹

### 2026-11-06 — 下载、适配、预检和启动

- GitHub git transport 超时、源码 zip 下载也超时；随后成功下载官方 PyPI 0.4.26 wheel。
  没有使用失败下载中的不完整源码。HF 直连不可达，通过公共镜像下载固定 revision，并核对权重 hash。
- 完整分词缓存完成：200,000 / 6,000 / 15,000；truncated=0、dropped=0。
- `tests/test_laya_baseline.py`：4 passed，包括真实微型 ModernBERT 全排列等变检查。
- 两个 arm 零更新显卡预检均通过，峰值约 4.98 GiB；初始参数 hash 匹配。
- 正式首步和 step 30 已检查，loss、gradient norm 有限；自动训练/测评链进入运行。
- 后续以实时 `status.json`、`progress.json` 与完成结果为准。备份仓库更新未请求。
