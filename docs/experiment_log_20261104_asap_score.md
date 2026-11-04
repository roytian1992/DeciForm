# ASAP S1/S2 原生多档位评分实验

## 当前状态与边界

2026-11-04 21:57 Asia/Shanghai：用户明确要求用空出的 4–7 卡启动 S1/S2。
原始数据和完整题目材料现已下载、审计并冻结；两组在 tmux 中启动零更新
GPU 预检；截至 22:06，两组均已通过并训练到 10/122 步，结束后执行完整
原生支持集的 dev/test。
S1 使用 4–5 卡、S2 使用 6–7 卡。D2 full、已完成 S3/S4 的模型和结果不变。

项目：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
运行根：`runs/formal_score_asap_20261104_v1/`。
数据：`data/score_specialist_asap_fold0_20261104_v1/`。
接收原件：`data/score_specialist_asap_intake_20261104_v2/`。
代码提交：`ecda189`；新源码隔离，原 S3/S4 源文件与哈希保持不变。

## 之前失败的原因与本次下载来源

之前尝试下载的是作者 GitHub 固定 revision 的 fold_0 TSV，错误是 TLS/
读取超时、网络不可达；Git 克隆也曾连接超时。并不是 Hugging Face 的一次
数据内容校验失败，也不是 GPU 不足。旧失败证据保留在
`data/score_specialist_research_20261104_v1/asap_author_fold0/`。

本次重新探测后，GitHub raw 恢复可达，Hugging Face 官方域名仍出现网络
不可达；通过 `hf-mirror.com` 成功取得 Hugging Face 仓库的原始文件。

- 原始作文与题目说明：
  [TasfiaS/ASAP-AES](https://huggingface.co/datasets/TasfiaS/ASAP-AES/tree/a46733395585b56c3bbe88f1957360a632ac3e9e)，
  revision `a46733395585b56c3bbe88f1957360a632ac3e9e`。
  `training_set_rel3.tsv` 为 16,337,165 bytes，SHA-256
  `cf04127eaff3224ce4aab64654e62e5738d3d672c61bb998546dd1674c60d312`；
  `Essay_Set_Descriptions.zip` 为 218,787 bytes，SHA-256
  `1ee7dac2284c983fc9e4aaa8ba915b079d5d7b67e2624d58ec0c35652f7c2211`。
  两项均与仓库 LFS 指纹匹配。
- 预先指定划分：
  [AES-NPCR fold 0](https://github.com/CarryCKW/AES-NPCR/tree/7a89a331be1f81871764b3eb4257f6732cabe9fb/data-set/asap/fold_0)，
  revision `7a89a331be1f81871764b3eb4257f6732cabe9fb`。
  三份 `*_ids.txt` 下载成功；作者 train/dev TSV 也成功，test TSV 本次仍超时。
  使用原始全量作文按固定 ID 重建三份文件，不更换 test 划分。
  这是作者固定 fold 0，并非 Kaggle 隐藏测试或五折复现。
  重建 train/dev 共 10,380 条的作文、prompt ID 和 domain1_score 与作者副本逐条完全一致。
- 题目、评分规则、阅读材料和评分合并规则从原始 ZIP 内 8 份 ReadMeFirst.docx
  的段落提取。保留全文任务相关段落，只去掉开头的语料规模等管理元数据。
  Set 2 明确预测 Domain 1；Set 1/7/8 使用最终 resolved score 的原生量程，
  不把单个阅卷人的 rubric 范围误当成最终分数范围。

本地下载 URL、revision、SHA 和失败尝试分别在接收目录的
`download_manifest.json`、`fold_ids_download_manifest.json`、
`hf_download_manifest.json`；原始压缩包与 TSV 均保留。

## 数据口径与完整性

| 面板 | 原始 fold 0 | 准入 | 说明 |
|---|---:|---:|---|
| train | 7,787 | 7,784 | 清除 2 条与 dev/test 相同正文及 1 条 train 内重复正文和标签 |
| dev | 2,593 | 2,593 | 全部保留 |
| test | 2,598 | 2,598 | 全部保留 |

原始 TSV 与全部划分 ID 的总量均为 **12,978**，ID 无交叉且覆盖原文件全部
记录。构建代码最初引用常见的 12,976 统计触发断言，已根据实际原件修正；
不存在两条记录丢失。口头进度中的“移除 3 条跨划分重复”不够准确，实际是
2 条跨划分 + 1 条训练内重复，具体见 `exclusions.json`。

原始 TSV 混有 Windows-1252 未定义的 `0x9d` 字节，严格解码会失败。
本次复现作者公开 `preprocess_asap.py` 的 cp1252/errors=replace 策略，
20 篇出现 78 个 U+FFFD，与作者 train/dev 副本逐字一致；不添加人工改写、
纠错或隐式截断。`manifest.json/source_encoding` 记录所有受影响 essay ID，
原始字节保留供追溯。这是原数据既有编码瑕疵，不声称无损 Unicode 转换。

8 个 prompt 全部纳入，原生量程分别为 2–12、1–6、0–3、0–3、0–4、0–4、
0–30、0–60，对应 11/6/4/4/5/5/31/61 档，训练和评测都不采样分数档位。
每篇原文保留，完整阅读材料、rubric 和 adjudication rules 随请求编码。
最长 train/dev/test 为 **6,774 / 6,826 / 6,670 tokens**，无硬截断。
缓存来源与长度审计见 `cache/audit.json`；27 条覆盖每个 prompt 和最长输入
的真实 tokenizer 无缓存/有缓存逐 token 核对通过。

## 公平对照与指标

两组保持相同 D2 架构、原始 Qwen3.5-9B base、r16 全模块 LoRA、任务路由、
QF、head geometry 和 ordinal readout。仅 Score 有数据监督，不能以本实验
证明多任务路由或 QF 的贡献。不继承 S3、D2、Kev 或 static-full 训练权重。

- S1：原生类别 CE；实现复用 `soft_ce` 入口，但原生整数目标使 q 严格 one-hot。
- S2：原 D2 ordered-CDF loss，系数固定 `3.0383850790577127`；`soft_cdf`
  入口在原生整数目标上与原 D2 值和梯度相同，8 个量程分别通过测试。
- seed 20261101、global batch 64、两卡、1 epoch、122 updates、相同顺序和 LR。
- 两组采用相同 4096 tokens 的微批次打包预算。它控制一次并行的记录数，
  单条超过预算时完整单条执行，不截断到 4096。保持 half checkpointing，
  预检覆盖每个 prompt 最长训练输入、最长 dev/test 和完整 global batch。
- 主指标：固定中位数解码、各 prompt 原生 MAE，以及等 prompt 权重 NMAE。
  RMSE、Spearman、QWK、NLL、normalized RPS 作为补充；其他解码规则保留
  为预注册诊断。不同量程的 pooled raw MAE 不作总体结论。
- 固定唯一 epoch 终点，不按 dev/test 换 checkpoint、不补额外 seed。

## 命令、监控和恢复

环境：`qwen35-train`，绝对 Python 路径：
`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
下列命令实际均使用该路径，工作目录为项目根：

```bash
python scripts/formal/build_asap_score_data.py --intake data/score_specialist_asap_intake_20261104_v2
python scripts/formal/cache_asap_score.py --data data/score_specialist_asap_fold0_20261104_v1 --workers 8
python -m pytest -q tests/test_asap_score.py tests/test_score_specialist.py tests/test_score_distribution.py
python scripts/formal/prepare_asap_score.py
python -u scripts/formal/supervise_asap_score.py --root runs/formal_score_asap_20261104_v1
```

最后一条位于 tmux `deciform_asap_score_20261104`，根日志 `scheduler.log`。
每组 `experiment_manifest.json` 保存环境、source/config/data 哈希、启动命令。
每组 `status.json`、`progress.json`、`preflight*.json`、`train.log` 记录状态；
异常写 `alerts.json`/`failure.rank*.json`，超过 15 分钟无进度另写警报。
每 25 步保存优化器、调度器、rank RNG 与 trainable 参数到 `resume.pt`。

异常后确认该组遗留进程已退出，用相同 Python 执行：
`scripts/formal/supervise_score_specialist.py --manifest <该组/experiment_manifest.json> --resume`。
不要重复启动整个根调度器来覆盖已有结果；检查失败原因后显式恢复对应组。

自动评测输出为各组 `evaluations/epoch_1.{dev,test_in_domain}.json`、
`.score_rules.json` 和 rank JSONL；两组成功后根目录写 `results_summary.json`。

## 轨迹

### 2026-11-04 21:57 Asia/Shanghai

从先前网络失败恢复：GitHub raw 取得指定 fold ID，HF 镜像取得哈希验证的
原始作文/题目 ZIP；重建划分并核对作者 train/dev。构建过程中显式修正
原始文件条数与既有 cp1252 解码问题，发现 3 条训练重复并记录移除原因。
原生完整支持集实现与全量缓存就绪，65 项测试、27 条真实 tokenizer 检查
通过；6 个新 Python 文件编译通过。提交 `ecda189` 后冻结两卡对照，
4–5/6–7 卡预检已启动。尚无本次训练结果。未请求远程备份。

### 2026-11-04 22:06 Asia/Shanghai：两组训练核验

S1、S2 均通过零更新预检并进入训练，已观测 10/122 步，初始 LoRA、head、
pointer 及 base config 哈希完全相同，且匹配已冻结 D2 的初始哈希。
当前源码与冻结清单一致，无 failure/alert。证据保存在
`runs/formal_score_asap_20261104_v1/launch_verified.json`。
前十步分别平均 26.86/26.60 秒，剩余训练约 50 分钟，另加后续自动评测；
训练峰值 allocated 显存约 46.32 GiB/卡。没有最终训练或测评结论。
