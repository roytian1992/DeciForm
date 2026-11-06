# Laya 多档位 Score 专项 — 2026-11-06

## 用户确认的口径
用户要求与已有 Score 专项对齐。核查 D2、Kev、JSON生成式LoRA的实际配置，均为 fresh_base_only、parent_checkpoint=null、专项训练1轮；没有继承20万条三任务 checkpoint。

本轮 Laya 因此从原始 ModernBERT-large 重新初始化，每次加载同一预训练 encoder，随机初始化 Laya decision head。**不继承20万条主实验权重，也不加载发布版 Laya 的决策权重。** 先前“主实验权重直接迁移/续训”的建议撤回；先前异步选择问题由本次明确对齐要求取代。

主 D2、正在4–7卡运行的 Laya＋机制迁移不变。本轮只补原生 Laya Score baseline；不加入我们的 A1–A3 模块，不改原生 RLCD 目标。

## 数据、训练与测评
| 面板 | train | dev | test | 完整分数支持 |
|---|---:|---:|---:|---|
| ASAP全部8题 | 7,784 | 2,593 | 2,598 | 原生4/5/6/11/31/61档 |
| STS-B＋WMT英中 | 8,129 | 2,500 | 2,379 | 51/101点既有数值网格 |

- ASAP原数据：data/score_specialist_asap_fold0_20261104_v1。
- 连续原数据：data/score_specialist_stsb_enzh_20261105_v1；训练含STSB4,882/WMT英中3,247，测试分别1,379/1,000。
- 每种布局独立训练ASAP一个模型、连续面板一个模型；后一个不会从前一个继续训练。
- sequential用0–1卡，parallel用2–3卡，各队列先ASAP后连续面板。
- 1 epoch，seed20261101，global batch64；ASAP122步，连续128步，末批保留全部真实记录、不补重复样本。
- encoder LR2.5e-5、head LR1e-4、AdamW weight_decay.01、CosineAnnealingLR至1e-6，沿用原生Laya主对照的训练设置。
- 两种布局保留原生Laya结构，全参数微调；预训练encoder和随机head起点相同。与Qwen/LoRA属于不同backbone/训练方式的方法对比，不称只改变一个模块。
- 原始RLCD：soft CE＋高斯logit采样REINFORCE，sigma.4、samples4、spherical.75、RPS1，代码从固定Laya0.4.26直接导入，不修改vendor。
- 整数原生标签使用one-hot；连续标签沿用专项协议映射到相邻网格两点的线性插值软标签。原始标签不取整；RLCD原本支持soft target，不新增私有loss。
- 训练、开发、测试都保留完整档位和原文；预编码最长ASAP6,788tokens、连续800tokens，零裁剪/零丢弃。
- 主解码固定native median，指标原生MAE；ASAP另报equal-prompt NMAE。mean/argmax/mean_nearest保留诊断，不按测试结果切换。用epoch1终点，不选择最优测试checkpoint。
- 历史Kev连续模型含额外英德训练数据；用户此前允许忽略这个差异，但表述仍不把该历史组称为严格同训练人口。当前D2/生成式复跑/本轮Laya均使用冻结STSB＋英中数据。

## 文件与复现
- 根：runs/formal_laya_score_20261106_v1。
- 数据缓存/来源SHA/counts：cache/{asap,continuous}/audit.json。
- 四组配置：{sequential,parallel}_{asap,continuous}/config.json。
- 输入/backbone/vendor/代码SHA与快照：experiment_manifest.json、code/。
- base：runs/formal_laya_20261106_v1/base；ModernBERT-large revision45bb4654a4d5aaff24dd11d4781fa46d39bf8c13。
- 新实现：scripts/formal/laya_score.py、prepare_laya_score.py、worker_laya_score.py、schedule_laya_score.py。
- worker基于原始Laya worker的分布式训练逻辑，扩展为可变数据总量和连续Score；动态微批按实际样本数/global batch加权，梯度跨rank求和，末批不丢失。原生RLCD advantage normalizer仍依赖microbatch，沿用其口径。
- 环境：qwen35-train；Python=/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python。
- 启动/恢复同一tmux命令：上述Python scripts/formal/schedule_laya_score.py --root runs/formal_laya_score_20261106_v1。
- tmux：deciform_laya_score_20261106。两队列依次执行真实零更新preflight、重新初始化训练、完整dev/test。
- 每40步保存optimizer/scheduler/RNG/model可恢复状态；原始参数精度保存最终checkpoint，并逐tensor核对与实际测评内存权重一致。
- 汇总results_summary.json。逐样本evaluations/*.rank*.jsonl，指标test_in_domain.original.json与score_rules.json。
- scheduler检查来源/配置/数据SHA，检查重复活跃进程，写实时progress/status与失败/停滞alerts，不隐藏错误或丢弃失败记录。
- 自动更新总MD独立LAYA_SCORE区块，使用runs/final_record_update.lock；不覆盖其他并行实验区块。
- 未请求远端备份；仅本地git记录本轮新增代码及日志。

## 检验与轨迹
- 2026-11-06：12项CPU测试通过，覆盖与原5档Laya模板完全一致、两布局编码、4至101完整支持、超过8192token不裁剪、连续标签保期望/不泄漏到输入、原生RLCD软标签反向有限。证据tests.log。
- 数据准备完成，四组初始化/训练配置均parent_checkpoint=null；两条GPU队列已启动preflight。实际训练启动以progress.json和launch_verified.json为准。

### 2026-11-06T15:38:28.047521+08:00 — 真实训练启动确认
- 两组ASAP零更新GPU预检通过，原生最长全文/最大K反向和完整dev/test前向通过，峰值约6.29GiB。两组已到10步以上；实际训练峰值约9.56GiB。
- encoder与随机head初始SHA与原20万条实验的“训练前初始状态”一致，未读取其训练后checkpoint。证据launch_verified.json；代码commit432f992。
- 两条队列会在各自ASAP训练和完整评测成功后，从原始backbone独立启动连续专项。主MD的LAYA_SCORE区块已加入placeholder并由队列自动更新。
