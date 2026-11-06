# 固定 D2＋Laya RLCD：Score 数据损失迁移

## 目的和状态

用户于2026-11-06授权使用空闲算力执行前一轮提出的两个损失对照。
研究问题：固定专项D2参数化与输入表示，Laya的分布奖励和带噪logits优化
是否改善多档位评分？本轮不是Laya架构迁移、不是新的主模型或A4。

运行根：runs/formal_d2_rlcd_score_20261106_v1。
启动时间：2026-11-06 17:20 Asia/Shanghai。GPU0–1为ASAP，GPU2–3为连续面板。
GPU4–5的Laya原生路径迁移继续独立运行，未被停止或改动。

## 严格对照与数据

| 面板 | 原D2运行 | Train | Dev | Test | 更新数 |
|---|---|---:|---:|---:|---:|
| ASAP | formal_independent_asap_score_20261105_v5/independent_d2_asap | 7784 | 2593 | 2598 | 122 |
| STS-B＋WMT英中 | formal_score_numeric_20261105_v1/d2_batched | 8129 | 2500 | 2379 | 128 |

上表原运行均位于项目 runs/ 下，其CDF-log结果复用。每个新面板从同一个
原始Qwen3.5-9B-Base及随机初始化的D2模块独立开始，不载入任何已训练
adapter/head或20万条模型。两个面板不是互相续训，也不混合ASAP与连续来源。

数据目录分别为 data/score_specialist_asap_fold0_20261104_v1 与
data/score_specialist_stsb_enzh_20261105_v1。连续测试是STS-B1379条、WMT英中1000条。
全文与全部合法档位保留；ASAP为4/5/6/11/31/61档，连续表示网格为51/101点。
小数标签用原相邻网格线性插值分布，不四舍五入为硬标签；无数据增加、删减或重采样。

固定：seed20261101、1epoch、global batch64、world2、全部原D2模块和
46,304,262个可训练参数、原AdamW/OneCycle调度/clip/结构正则。
LoRA和pointer初始最大LR2e-5，residual/router1e-3，保留原热身比例0.3。
ASAP保留全层checkpoint、4096 token预算、每次最多8个候选分支的执行路径；
连续面板保留半层checkpoint、16384 token预算、最多4行微批和批量独立候选路径。
ASAP的历史config中schedule描述写了3125，实际worker以total_steps122构建；
新config修正描述为122，未改变有效调度。

## 损失实现

原版本固定为本地vendor Laya0.4.26，使用 laya/train.py:499 rlcd_loss
和 laya/common.py:720 proper_reward。新增 score_rlcd.py 沿用其逐元素运算，
唯一接口变化是返回每行loss，以接入现有按全局样本数归一化的D2训练循环。
同一mask、logits、targets和RNG下，逐行loss均值及梯度与上游相同。

$$
\mathcal L=\mathrm{softCE}(z,t)
-\frac{1}{M}\sum_{m=1}^{M}\mathrm{stopgrad}(A_m)
\log\mathcal N(\widetilde z_m;z,\sigma^2 I)
+\mathcal R_{\mathrm{D2}},\quad M=4,\;\sigma=0.4.
$$

噪声在每行有效选项上均值为零，扰动中心取stopgrad(z)。分布奖励为
截底对数评分＋0.75球面评分－1.0归一化RPS。优势先对每条样本的4次
扰动中心化，再除以整个当前microbatch的advantage标准差，epsilon1e-6。
reward和advantage停止梯度；不是直接反传负reward。分类任务不参与本研究。

softCE和奖励均在FP32小张量上计算；一组logits只来自一次原模型评分，
没有四次backbone前向。原模型自身独立候选分支/chunk计算量仍然存在。
原D2正则每条样本一份，loss.sum()/实际global batch行数之后跨rank求和，
末批不满64也按实际行数归一化。保留原微批划分；Laya的advantage归一化
本身依赖微批，不能称为任意分批等价。随机loss会消耗新的随机数，无法
同时声称每一步dropout的随机实现与CDF-log对照完全相同。

## 验证与运行机制

6项CPU测试已通过：单行/混合4、11、61档/51、101档的上游loss值、梯度、
RNG精确路径对照；padding梯度为零；RLCD额外项生效；诊断不改变loss、
梯度或RNG；只调用一次scores；连续target插值与越界拒绝。测试日志在
运行根 unit_tests.log。语法检查通过。

实际每组双卡预检零optimizer更新：验证初始化SHA和优化器参数组与原D2
完全一致、正常批和最长输入反传、最大完整评测输入、dense和first-global
batch、有限梯度/显存限制。全部通过后supervisor才启动fresh训练。
整个训练/评测由tmux持久托管，逐步心跳、每25步checkpoint、15分钟无进展
告警、30分钟停滞终止并保留恢复点。所有源码/config/cache哈希冻结；未修改活跃共享代码。

源码入口：scripts/formal/score_rlcd.py、models_score_rlcd.py、worker_score_rlcd.py、
prepare_score_rlcd.py、supervise_score_rlcd.py、report_score_rlcd.py。
Python环境：/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python。
项目cwd：/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm。
每组experiment_manifest.json的launch_command和resume_command为准确命令；
launch_proof.json记录tmux命令和启动时间，code/存放冻结源码。

主评测为固定epoch1、完整Dev/Test、中位数解码、原生MAE及等题目/来源NMAE；
保留mean/argmax作为诊断，不按测试成绩更换解码。Score不做顺序实验。
结果来自各arm的evaluations/epoch_1.test_in_domain.json及逐行jsonl；只有
完整完成才填入主记录D2_RLCD_SCORE区块。报告器使用共享锁，不覆盖别的实验。

## 耗时边界

原严格同架构对照实际记录：ASAP训练15990秒、评测2079秒；连续面板训练
2394秒、评测372秒。因ASAP保留长正文及逐档位候选编码，预计明显慢于
一次文本编码的数值CDF/数字前缀树模型。此轮先保证只换loss，不能为了
追求几分钟跑完而换架构或截断输入。实际ETA以新训练日志为准。

本次未请求远程备份；所有既有结果和失败证据保留。单seed探索不能证明
跨seed稳定性；此前多次查看测试集，结果须标为探索性证据。

## 执行轨迹

2026-11-06 17:20：冻结两个arm，源码与原参考哈希均核对；从持久tmux
启动真实双卡预检，自动报告器已接主记录。此时尚无新训练或测评结论。
