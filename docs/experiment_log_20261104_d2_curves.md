# D2 full 数据量敏感实验（2026-11-04）

## 范围和固定条件

用户授权：当前 D2 full；0–1 卡依次运行 25k、50k；2–3 卡运行 100k。
三次均从原始 Qwen3.5-9B fresh base 独立初始化，50k 不继承 25k。
已完成的 D2 200k 作为第四个曲线点复用；不重新训练 Kev。

- D2 参考：`runs/formal_score_distribution_20261104_v1/d2_ordinal_main`。
- 数据：`data/formal_200k_20261101_v1`，原始 200,000 条冻结训练视图。
- 子集：直接复用 `runs/formal_curves_20261102/indices/indices_{25000,50000,100000}.json`。
- Kev 对照：`runs/formal_next_20261102/kev_{25k,50k,100k}`。
- 子集按 task/source 分层、稳定 hash 前缀，严格嵌套；不更改样本或负候选。
- seed 20261101，1 epoch，global batch 64，学习率及 OneCycle 比例沿用 D2。
- 更新数：391、782、1563；最后不足 64 条的 batch 按实际条数归一化。
- 模型和 loss 保持 D2：task routing、QF、head geometry、ordinal readout；Choice CE、Noul BCE、Score 原始 ordered-CDF loss（系数 3.0383850790577127）。
- 自动评测固定开发集 6,000 条和域内测试集 15,000 条，完整原生候选、不截断。
- Choice/Noul accuracy；Score predictive median 的原始量纲 MAE 为主，保留四种固定解码规则。
- 来源外面板继续推迟；4–7 卡已有 ASAP S1/S2 不受影响。

| 训练记录数 | Choice | Noul | Score | GPU |
|---:|---:|---:|---:|---|
| 25,000 | 6,251 | 9,374 | 9,375 | 0,1 |
| 50,000 | 12,500 | 18,750 | 18,750 | 0,1，25k 训练评测成功后 |
| 100,000 | 25,000 | 37,500 | 37,500 | 2,3 |

来源：三个冻结 indices JSON 的 task_counts，经实际训练视图重新核对。

## 实现和启动

项目根目录：`/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`。
Conda：`qwen35-train`；Python：`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`。

`scripts/formal/worker_score_curve.py` 是冻结 D2 worker 的隔离版本，只开放训练子集规模、校验子集 hash/来源/分层计数，以及修复开发集评测后恢复时的 report 读取；不改变模型数学或训练 loss。原始运行文件保持不变。

```bash
python scripts/formal/prepare_score_curves.py
python scripts/formal/schedule_score_ablations.py --root runs/formal_d2_curves_20261104_v1 --arms d2_25k d2_50k
python scripts/formal/schedule_score_ablations.py --root runs/formal_d2_curves_20261104_v1 --arms d2_100k
```

两个队列在独立 tmux 中运行。每臂先零更新 GPU preflight，验证初始 base/LoRA/pointer/head hash 与 D2 相同，再独立加载 fresh base 正式训练。每 100 步保存恢复点；异常保存 `alerts.json`/`failure.rank*.json`，无进度超过 15 分钟报警文件可见。队列失败则停止，不错误启动后续臂。
恢复命令：`python scripts/formal/supervise_score_specialist.py --manifest <arm>/experiment_manifest.json --resume`；仅用于同一个实验的自身恢复点。

输出：`runs/formal_d2_curves_20261104_v1`；逐臂 `config.json`、`experiment_manifest.json`、`status.json`、`progress.json`、`train.log`、`evaluations/epoch_1.{dev,test_in_domain}.json` 和 `.score_rules.json`。
队列：`queue_01.json` / `queue_23.json`。
历史 Kev 保存概率由 `replay_curve_references.py` 在 CPU 重算同一 median MAE，写入本次 `reference_score_rules/`，保留原始 argmax 报告不覆盖。

## 可比性说明

每档均为一轮，所以同时改变独立记录数和优化步数；本实验是固定 epoch 的数据规模曲线，不是固定计算量的因果消融。三个新点各用 2 张卡；既有 D2 200k 用 4 张卡，因此不将其直接解释为匹配硬件的速度曲线。
Kev 旧 config 中的 continuation 字段是遗留元数据；对应 worker_next 不读取该字段，仅在显式 `--resume` 时读取当前实验自己的恢复点。三档的 initialization.json 都是 fresh base、无继承 adapter/head，完成记录为完整一轮。新 D2 config 明确禁止 continuation。

## 轨迹

### 2026-11-04 22:24 Asia/Shanghai — 准备

完成冻结子集、既有 Kev 完成记录与 fresh 初始化检查。50 项子集契约/原始 D2 loss 测试通过；编译检查通过。准备三个隔离配置和自动队列，尚无新模型结果。未请求远程备份。

### 2026-11-04 22:33 Asia/Shanghai — 启动已验证

- 代码提交 `a6139ce`；两个 tmux 队列于 22:30 启动。
- `deciform_d2_curves_g01_20261104`：25k 已通过 preflight，观察到正式训练 1/391；50k 等待其训练及评测成功后启动。
- `deciform_d2_curves_g23_20261104`：100k 已通过 preflight，观察到正式训练 1/1563。
- 两臂预检峰值均约 54.81 GiB、优化器更新数 0；base/LoRA/pointer/完整 head 初始化 hash 与 D2 相同。正式训练重新加载 fresh base，没有继承训练 checkpoint。
- BGE 8081–8084 服务已按精确 PID 正常 SIGTERM，GPU0–3 显存释放后启动新队列；重启参数仅保存于运行目录的 0600 私有记录，不公开凭据。GPU4–7 的 ASAP S1/S2 正常运行。
- Kev 25k/50k/100k/200k 和 D2 200k 的保存概率已按同一个协议重放；全任务行 ID、标签和候选支持完全匹配，开发集和测试集报告位于 `reference_score_rules/`。
- 启动证据：`launch.json`、`launch_verified.json`、逐臂 `preflight_passed.json` / `initialization.json` / `progress.json`。尚无新模型评测结果；初始单步耗时不作为稳定 ETA。
- 下一步由队列自动完成三个训练与完整候选评测。恢复时使用同臂 `--resume`；代码/config/hash 在运行期间冻结。
