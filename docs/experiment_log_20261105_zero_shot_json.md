# 未经任务微调的 LLM JSON 基线

## 目的与版本

用户要求补充无需训练的 LLM 推理基线，覆盖正式主实验、Score 专项及位置变换。
当前执行口径为 **官方 Qwen3.5-9B 指令模型 zero-shot**：用户已明确选择“只补官方指令模型”。它没有使用我们的任务数据微调，不加载 LoRA 或自定义决策 head，直接产生 JSON。
这是现成指令 LLM 的零样本能力参照，不预设指标一定是能力下限。它已有官方后训练，不能表述为“只有预训练”或严格同 Base 权重消融。

原 Base 队列 `runs/formal_zero_shot_json_20261105_v1` 已按用户选择取消，取消时没有分配 GPU、启动推理 worker 或产生预测。原 manifest 与 `cancellation.json` 保留为过程记录，不进入结果表，不会自动恢复。

## 数据与推理次数

运行根目录：`runs/formal_zero_shot_instruct_20261105_v3`，唯一模型 ID 为 `instruct`。

| Panel | 原始测试题目 | 显示顺序 | 推理输出数 |
|---|---:|---|---:|
| 20 万训练条目主实验对应的 Test | 15,000 | 原序 | 15,000 |
| ASAP fold 0 | 2,598 | 原生分数范围 | 2,598 |
| STS-B + WMT | 3,379 | 原生连续范围 | 3,379 |
| Choice 位置实验 | 5,000 | 两个冻结设计的并集，共 10 个命名条件 | 50,000 |
| Noul 显式列表位置实验 | 5,000 | [false,true] / [true,false] | 10,000 |
| 合计 | 主面板与位置面板共享题目，不能相加为独立题目数 | | **80,977** |

主面板数据：`data/formal_200k_20261101_v1/cache/test_in_domain.views.jsonl`。
Score：`data/score_specialist_asap_fold0_20261104_v1/test_in_domain.jsonl` 和
`data/score_specialist_continuous_20261104_v1/test_in_domain.jsonl`，沿用
`data/score_baseline_encodings_20261105_v1/{asap,continuous}/audit.json` 验证的输入。

不评训练集；不改训练数据；不重新训练已完成模型。此请求不恢复延后的三任务外部来源面板。

## 比较协议

- 官方指令权重路径：`../models/Qwen3.5-9B`；四个权重 shard、config、tokenizer、model card 等实际文件 SHA256 全部保存。
- 保留生成式 LoRA 的任务正文和 JSON 字段要求，使用该模型自带 chat template、`enable_thinking=False`、greedy、每条最多生成 64 tokens。不使用 few-shot，不加 schema 约束或 JSON 修复。64 是新增指令模型的输出预算，不是输入截断；相对于生成式 LoRA 的原始 completion prompt 和 15/17/30-token 预算，模板与输出预算不同，因此是实用零样本参照，不宣称只有 LoRA 权重一个因素不同。
- 所有原始候选、文本和规则允许的分数范围完整保留。Score 不取整原始小数标签，ASAP 整数范围不变；未输出合法值时按原 baseline 全范围误差处罚，留在总分母。
- JSON parse validity 与 Choice/Noul accuracy 一起报告。Score 主要报告每源 native MAE、跨范围 equal-source NMAE。生成的一个数值不是预测概率分布，不构造 NLL/Brier/expected-score。
- Choice 的 10 个命名条件由新版 D2 的 7 个顺序和旧基线的 5 个顺序合并得到，原序/倒序共用命名。报告分别提取 **相同 7 条件**、**相同 5 条件** 子集，不把不同顺序数量的 any-order flip 直接比较。个别题目中命名条件可能碰巧形成相同排列，仍按冻结设计执行并保存 input hash。
- 候选 ID 始终绑定原语义，输出按原始 ID 解析。Noul 使用原提问后追加 `Allowed answers: ...` 的已冻结干预，不把没有候选列表的原生布尔接口报告为 0 翻转。
- Noul prompt-list 干预与新版 D2 的 native candidate-branch 干预属于不同接口层；结果必须标注，不能作为相同输入干预直接归因比较。旧生成式/普通 LoRA 的完整 Noul prompt-list 对照可直接复用对应协议。
- Score 不新增置换/改判实验。两种顺序都解析失败时，记录格式失败并报告双方均有效的配对数量/flip rate；不将其宣传为正确决策稳定。

## 实现、环境、检查

新增 `scripts/formal/zero_shot_protocol.py`、`worker_zero_shot_json.py`、
`prepare_zero_shot_json.py`、`schedule_zero_shot_json.py`，不改正在训练的 Score 源码/配置。

Python：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。
cwd 为 DeciForm。每个模型分 4 个固定 shard，每个进程只占 1 张 GPU；模型全部 requires_grad=False，零 optimizer 更新。
原始 outputs、token 数、输出预算是否耗尽、解析失败原因和 prompt token hash 保留在 JSONL。

CPU：`python -m pytest -q tests/test_zero_shot_json.py tests/test_score_baselines.py tests/test_classification_order_audit.py`，55 passed（39.18 秒）。覆盖不泄露标签、完整 K、稳定 ID、实际提示词换序、两种冻结 order 设计的精确一致、无效输出计入分母及不伪造概率。
原 Base 版本以真实 tokenizer 检查 99 条输入与训练基线缓存逐 token 相同；该历史检查不代表指令模型。新版本在同样 99 条真实输入上验证官方 user/assistant 边界及空 thinking 前缀，完整输入保留；具体校验文件为新目录的 `tokenizer_validation.json`。
GPU 释放后，每个 shard 先检查模型文件 hash、无 adapter、零可训练参数及最长完整输入真实推理，再进入批量评测。该检查只验证运行/解析路径能执行，不根据结果修改模型或提示词。

## 调度、状态与恢复

启动：manifest 的 `launch_command`，tmux session `deciform_zero_shot_instruct_20261105`。旧 session `deciform_zero_shot_json_20261105` 已停止。
原 Score 任务占用的 GPU 只有在其 status 为 complete/exit_code 0、complete.json 存在且显存释放后才能分配。当前 0–1 ASAP D2、2–3 ASAP CE、4–5 continuous CDF、6–7 continuous CE 均受保护。

根 `status.json` 区分 waiting_for_gpu/running/failed/complete；每个模型的 shard 目录保存 allocation、worker.log、progress、各面板报告和 complete/failure。
预测逐批落盘；恢复保持原 batch/padding，已落盘记录不会覆盖。scheduler 使用文件锁，并拒绝与仍存活的同运行 worker 重复启动。模型或源码 hash 不匹配即停止，禁止默默切换。
一张卡上的任务失败会写 alerts/failure，并阻止进一步新启动；其它已运行进程可以完成。不通过工具向外部用户发送消息。

完成的 shard 会自动合并为 `instruct/{main,asap,continuous,choice_order,noul_order}.json`；所有测试行数与语义 ID 完整性必须通过断言。没有实测结果时保持 Placeholder。

## 轨迹

- 2026-11-05：确认用户新增零样本参照；识别当前 backbone 是 Base，而本地官方指令版是另一套权重；先准备严格同骨干 Base。
- 新建隔离实现和 55 项通过的 CPU 检查，冻结同一测试数据与输出预算。没有新增训练或中止既有任务。
- 准备 manifest、原始权重/代码/input hashes、真实 tokenizer parity 与常驻推理队列；实际队列状态以 status.json 为准。没有在 GPU 推理发生前声称已获得结果，也没有进行 test-based prompt selection。
- 未请求或执行远程备份。
- 2026-11-05 用户明确选择只评官方指令模型：停止尚未分配 GPU 的 Base scheduler，确认无 allocation/预测文件，写入取消记录。原配置不覆盖；新建 instruct-only v2，核验本地官方权重与 chat 模板，使用 64-token 无 thinking JSON 生成并替换自动队列。此选择同时适用于主面板、Score 和位置实验，不能再自动补 Base。
- 指令 v2 的启动复核发现真实 tokenizer 的 `apply_chat_template` 在 Transformers 5 默认返回 BatchEncoding，`len()` 是字段数 2，不是输入 token 数。虽然 decode 能接受该对象，batch padding 不能把它当 token 列表。v2 在任何 GPU worker 分配前撤下并保留 cancellation；修复为显式 `return_dict=False` 及非空整数列表断言，新增回归检查，重新冻结到 v3。没有截断或丢弃输入，也没有产生错误的模型结果。真实 tokenizer 的 99 条验证及 token 长度以 v3 文件为准。
- 19:28:59 +08:00：v3 instruct-only tmux 队列启动。针对修改运行 `tests/test_zero_shot_json.py`，12 passed（14.07 秒）。99 条真实 tokenizer 检查通过，各面板检查到的最长输入 main/ASAP/continuous 分别为 5580/6584/286 tokens；全部为完整整数 token 列表，官方 assistant 空 thinking 前缀正确。当前等待受保护 Score GPU 释放；启动记录在 v3 的 launch.json。
