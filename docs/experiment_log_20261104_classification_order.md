# Choice/Noul 改判实验：范围纠正与 Choice 补测

日期：2026-11-04，Asia/Shanghai。

用户最新明确：改判实验只研究 Choice 和 Noul，不需要 Score 改判实验。
此前实际执行的是 Score/Noul，Choice 尚未纳入，因此上一轮结果不能称为
Choice/Noul 的完整实验。Score 的分数变化与 MAE 属于数值评分诊断，不再
列入当前改判表，也不追加 Score 换序计算。

旧目录 `runs/formal_answer_order_20261104_v1` 的原始 protocol、预测及
report 保留溯源。新增 `classification_report.json` 为当前汇总，只包含
Choice/Noul；Choice 未完成时明确标为 partial，不能把 Noul 结果冒充 Choice。
完成的 Noul 直接复用相同 checkpoint 的旧结果，不重新花费 GPU。

## Choice 补测

输入是正式冻结测试的全部 5,000 条 Choice：AG News 1,200 条/K=4、
Amazon 1,200/K=5、CFPB 1,200/K=21、MNLI 800/K=3、Banking77 600/K=77。
每题保留完整原生候选列表，测原始顺序、倒序及三个由 row ID 固定的不同
随机排列。共 5 个视图/题；不是 5 个候选，也不采样、删除或截断候选。
五个视图不是全部排列，也不声称每个候选位置均衡。

四个已完成模型：ordinary LoRA、Kev、JSON-generative LoRA、当前 D2 full。
每个模型 25,000 个预测视图，共 100,000 个，实际测试人口仍为 5,000 条。
沿用前一轮 checkpoint 哈希、原始测试 cache、推理精度与解码配置。
不训练、不修改 D2，不使用任何新标签。

输入处理按真实接口执行：

- ordinary LoRA：实际重排逐候选编码序列，将返回概率和所选位置映射回
  原候选 ID。保持实际 argmax 的 tie-breaking，不人为改成零改判。
- Kev/D2：实际重排因果输入中的候选 token 段，读取位置按稳定候选语义
  对齐。不会把送入 backbone 的候选再排序回原位。
- JSON 生成式 LoRA：调换候选文本的显示顺序，原 `c0` 等 ID 始终绑定
  同一个候选内容；解析仍用原语义 ID，不把标签换号算作模型改判。

主要指标是 accuracy、倒序改判率、多视图任一改判率；另外报告概率 TV/JS、
最坏视图错误率和解析失败。失败保留在分母。保存逐条预测和每个实际输入
的哈希。按来源/group 的 2,000 次配对 bootstrap 只反映固定模型评测不确定性。

Noul 沿用原生 5,000 条及共同指令 1,000 条两个面板。普通 LoRA 与生成式
LoRA 的原生 Noul 接口没有候选列表输入，继续报告 N/A；共同指令列表仍是
另外一种干预。不能混合这两种面板，也不能把 N/A 算成零改判优势。

## 文件和运行

- 项目：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
- 环境：`../anaconda3/envs/qwen35-train/bin/python`。
- 新 root：`runs/formal_choice_order_20261104_v1`。
- 数据：`data/formal_200k_20261101_v1/cache/test_in_domain.views.jsonl`。
- 编码/语义：`scripts/formal/classification_order_audit.py`。
- 准备：`scripts/formal/prepare_choice_order_audit.py`。
- worker：`scripts/formal/worker_choice_order_audit.py`。
- 监管：`scripts/formal/supervise_choice_order_audit.py --root runs/formal_choice_order_20261104_v1`。
- 当前汇总：`scripts/formal/report_classification_order.py`。

仍使用两张空闲卡：6 卡 ordinary → generative，7 卡 D2 → Kev。
监管程序核对先前 A2 已完成和显存释放后启动，不占用 A3 所在的 0–3 卡。
每个 checkpoint 严格恢复全部可训练张量，记录零训练更新，自动记录异常。
Choice 全部完成后自动生成 Choice/Noul 当前汇总。

## 轨迹

读取实际 protocol 确认旧实验 tasks 为 Score/Noul，检查正式 Choice 源分布。
新增隔离的补测入口，没有修改历史输入或旧报告；当前汇总过滤 Score。
33 项编码/排列/语义映射/失败统计测试通过。真实 tokenizer 的 cache 一致性
与换序检查写入新 root 的 `cpu_validation.json`（完成后复制）。
原始 checkpoint、配置、代码快照、输入和输出哈希由新 manifest 记录。
是否实际启动以 root `status.json`、各模型 `progress.json` 和
`checkpoint_loaded.json` 为准，不把 CPU 测试当成 GPU 结果。未请求备份。
