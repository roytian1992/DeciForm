# Choice/Noul 小规模 RLCD 筛选实验

## 范围

用户于2026-11-06要求先使用小训练集和小测试集快速验证RLCD是否有助于Choice/Noul。
Score不参与。本实验不重训或替换正式20万条D2，也不改变运行中的ASAP任务。

- 项目：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
- 运行根：`runs/pilot_classification_rlcd_20261106_v1`。
- 数据根：`data/pilot_classification_rlcd_20261106_v1`。
- 父数据：`data/formal_200k_20261101_v1`，沿用原train/dev/test分割。
- Python：`../anaconda3/envs/qwen35-train/bin/python`。

## 数据与公平性

| 分割 | Choice | Noul | 总计 |
|---|---:|---:|---:|
| Train | 5,000 | 5,000 | 10,000 |
| Dev | 500 | 500 | 1,000 |
| Test | 1,000 | 1,000 | 2,000 |

在每类任务内部按来源和标签进行比例分层，使用固定seed及ID哈希选记录。
各来源配额、标签、原行索引、缓存哈希、候选数量分布写入数据manifest。
三个分割的ID/group交叉检查；不新划分父数据，不根据结果挑题或过滤长文本。
三组共用完全相同的样本与顺序。训练Choice沿用父缓存的1个正例＋至多3个负例；
开发和测试每题保留完整原生候选，包括Banking77的77个和CFPB的21个。
Noul保持完整两个选项。没有Score，也没有截断文本或删候选。

## 三组并行

| 名称 | GPU | Choice目标 | Noul目标 |
|---|---|---|---|
| ce | 2–3 | 原CE | 原Bernoulli NLL |
| choice_rlcd | 4–5 | Laya原生RLCD | 原Bernoulli NLL |
| noul_rlcd | 6–7 | 原CE | Laya原生RLCD |

每组从同一原始Qwen3.5-9B-Base及相同随机初始化D2模块开始；不载入已训练
LoRA/head。固定独立候选编码、路由、Choice QF、任务几何、结构正则及优化器。
保留完整参数结构，Score专属路径无数据监督。原D2初始权重和优化器分组哈希
逐项核对。每组2卡、global batch64、1epoch、157步，末批16条全部保留。
OneCycleLR按157步构建，原warmup0.3、半层checkpoint与token预算不变。
小规模CE对照同样重新训练，不能与20万条模型的分数直接归因比较。

## 数学与实现

对指定任务，沿用固定Laya0.4.26的softCE＋带噪logits策略梯度。

$$
R(p,y)=max(log p_y,-9.21)+0.75rac{p_y}{lVert pVert_2}.
$$

Choice/Noul不启用RPS；sigma0.4、4次噪声采样、每条有效候选噪声去均值。
奖励和优势停止梯度，优势先按每行4次采样中心化，再除以当前microbatch中
所有被指定为RLCD的行和采样的标准差；每行loss最后按global batch实际样本
数归一化。保留微批依赖性，不声称任意分批等价。

RLCD噪声在fork_rng上下文中产生，防止改变后续backbone dropout的RNG序列。
与原生Laya的损失/梯度计算相同，但随机数流管理不同，这是显式对照约定。
RLCD只作用于一次模型前向得到的logits，无额外backbone前向。

入口：`classification_rlcd.py`、`worker_classification_rlcd.py`、
`prepare_classification_rlcd.py`、`supervise_classification_rlcd.py`、
`report_classification_rlcd.py`，均在`scripts/formal/`。
新脚本隔离，运行期间冻结源码和config，不修改其他活跃任务源码。

## 验证与运行

- CPU：6项测试通过，覆盖上游值/梯度/RNG路径、padding梯度为零、CE完全一致、
  指定任务隔离、loss诊断不改梯度、单次评分、分层抽样稳定及拒绝Score。
- CUDA：各组正式启动前进行零optimizer更新预检，覆盖最长完整训练样本、
  两类任务反向、最大完整评测样本、dense/首批及梯度聚合。
- tmux持久托管，每步心跳，每25步保存可恢复checkpoint；15分钟无进展告警，
  30分钟停滞终止并保留恢复点。
- 启动/恢复的完整命令在各组`experiment_manifest.json`。
- 训练后自动Dev/Test；仅固定epoch1，不按测试选checkpoint或改变解码。
- 主指标分任务Accuracy，来源明细及NLL/Brier作为诊断；保存每条概率预测。
- 自动报告器写`results_summary.json`，并只更新主记录独立pilot区块。

每类Test1,000条只能快速筛选方向。输出来源分层配对bootstrap区间及新增/丢失
正确数，不把小差距或单seed结果称为稳定提升。Dev和Test均保留，优先看方向
是否一致，不能通过反复挑测试结果获得正式结论。没有自动扩展到20万条的队列。

## 轨迹

### 2026-11-06 19:12 Asia/Shanghai — 构建小规模分类对照

已实现独立loss/worker/supervisor与自动报告，6项CPU测试通过；数据冻结和
CUDA预检待完成。0–1的ASAP继续运行，2–7分配给三组小对照。
本轮不要求远程备份或额外训练seed；所有原结果保留。


### 2026-11-06T19:15:51.414760+08:00 — 三组已进入正式训练

三个双卡预检均通过，峰值显存62.89 GiB、零optimizer更新；三组训练初始化完全一致，与原D2 fresh初始化哈希一致。
- ce：GPU 2,3，已验证step 16/157，平均7.28秒/步。
- choice_rlcd：GPU 4,5，已验证step 16/157，平均7.23秒/步。
- noul_rlcd：GPU 6,7，已验证step 16/157，平均7.23秒/步。
完整证据：runs/pilot_classification_rlcd_20261106_v1/launch_verified.json。预计剩余20–30分钟含自动Dev/Test，以后续心跳修正。
数据冻结已完成：10,000/1,000/2,000条；三分割ID/group无交叉；10个来源；评测Choice完整K为3/4/5/21/77，Noul为2。
ASAP 0–1继续运行。自动报告器只更新CLASSIFICATION_RLCD_PILOT区块，不替换正式D2结果；本轮无自动扩大训练队列。
