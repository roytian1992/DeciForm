# 新版独立候选 D2：补齐 Score 专项对照

## 授权与状态

用户要求 A1–A3 结束后补齐缺少的 Score 实验。三组消融于 2026-11-05 17:10 Asia/Shanghai 完成完整 dev/test 和 Choice/Noul 倒序检查；各自 `evaluation_recovery_20261105_v2/status.json` 为 complete、exit_code 0。原失败 status 保留，不用其作为 GPU 释放条件。

本次冻结 `runs/formal_independent_score_completion_20261105_v1`，在释放的三对卡上启动常驻 tmux supervisor，依次执行零更新 GPU 预检、从原始 base 训练和完整 dev/test。实际阶段以各 `status.json` 为准。

| Arm | GPUs | Train / Dev / Test | Steps | 目标 |
|---|---|---|---:|---|
| asap_ce | 2–3 | 7784 / 2593 / 2598 | 122 | 新版结构 + categorical CE |
| continuous_cdf | 4–5 | 11457 / 3500 / 3379 | 180 | 新版结构 + 插值标签 soft-CDF |
| continuous_ce | 6–7 | 11457 / 3500 / 3379 | 180 | 新版结构 + 同一插值标签 soft-CE |

已运行的 ASAP D2 继续占用 0–1，路径 `runs/formal_independent_asap_score_20261105_v5/independent_d2_asap`。该任务的源文件、配置、进程保持原样。旧因果候选编码的 S1–S4 和既有 Kev/JSON LoRA Score baselines 保留。

## 冻结的比较条件

- 数据：`data/score_specialist_asap_fold0_20261104_v1`、`data/score_specialist_continuous_20261104_v1`；读取现成 passed cache/audit.json，并核对原 manifest 和每个 split 的 view/cache hashes。
- ASAP 保留完整原生 4/5/6/11/31/61 档；连续面板 STS-B 51 点、WMT 101 点。小数标签采用邻点插值，不四舍五入；WMT 两语向训练不变，en-zh 为报告重点。
- 单 epoch、seed 20261101、global batch 64、OneCycle 30% warmup，fresh Qwen3.5-9B base，继承 adapter/head 为 null。
- 全部新版模块保持启用，46,304,262 可训练参数；初始化 LoRA、pointer、head、base config hash 必须与新版主实验一致。
- 与正在训练的 ASAP D2 相同的 branch chunk 8 和全层激活 checkpointing，4096 为分批预算而非截断长度；任何超长单条仍完整输入。连续两臂也使用相同设置。
- 配对实验只改 loss；主结论用 native median MAE、跨量纲 equal-source NMAE；mean/argmax 等固定解码仅保留为诊断。没有新增 Score 置换、普通 LoRA Score 或外部三任务面板实验。

## 代码、环境与验证

新增隔离脚本：`models_independent_score_specialist.py`、`worker_independent_score_specialist.py`、`prepare_independent_score_completion.py`（位于 scripts/formal）。不修改已运行任务的模型、worker 或 supervisor 源码。

新 worker 显式绑定 panel/model/metrics，不依赖运行时替换 sys.modules。Score 在三任务模型中仍使用 task route 2；支持原生任意递增数值档位。预检使用真实模型预测后调用 prediction_row、metrics、score_rule_report，训练前检查整条评测路径，防止再次出现缺失结果函数的问题。完成文件按具体 panel 记录支持集，不沿用 generic worker 的固定 [51,101] 元数据。

Python：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`；cwd：DeciForm。精确包版本、环境变量、代码快照及 SHA256 见各 `experiment_manifest.json`。

CPU 检查：`python -m pytest -q tests/test_independent_score_specialist.py tests/test_score_specialist.py tests/test_asap_score.py tests/test_score_distribution.py`，67 passed。包括与正在运行的 ASAP chunk 函数在 dropout 下逐值相同的前向/梯度、连续支持集不遗漏、Score 路由正确、CDF 在整数节点与原 D2 目标/梯度一致、标签插值与数值指标。最初测试收集缺少 vendor import path，已在首次 GPU 启动前修复；原预冻结测试与 manifest 留存在 `prelaunch_history`。生产模型和数据未变。

启动/恢复命令见各 manifest 的 `launch_command` / `resume_command`。启动外壳为 tmux session `deciform_ind_score_{arm}_20261105`；supervisor 在预检失败时记录 alerts 并停止该臂，成功后自动开始训练及原生完整评测。日志包括 supervisor.log、preflight.log、train.log、progress.json、status.json；训练每 25 步保存恢复点。

注意当前 generic worker 每 10 步写一次训练 progress；长篇 ASAP 可能超过 supervisor 的 15 分钟 stale 提示阈值。该提示不终止任务，也不能单凭它判定卡死；结合进程、GPU、日志及 checkpoint 时间检查。

## 轨迹

- 17:10：A1–A3 恢复评测完成，每臂 6k dev、15k test、10k 分类倒序对；无额外 optimizer 更新。
- 本次：核对空闲 GPU，冻结上述三组代码/数据/配置；67 项 CPU 检查通过后启动三组独立 tmux 预检→训练→评测。真实启动时间及命令见根目录 launch.json；不把预检/启动等同训练完成。
- 下一步：以 preflight_passed.json、progress.json 和最终 complete.json 跟进。结果仍为空，不能提前声称 ordinal 比 CE 更好。未创建远程备份。
