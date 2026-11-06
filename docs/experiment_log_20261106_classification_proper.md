# Choice/Noul: normalized proper scoring and scaled RLCD pilot

## Purpose and scope

User authorizes the next small loss study. This isolates loss changes on the completed independent-candidate D2 classification pilot. It is separate from R26 planning and does not start a new main-data experiment. Score, formal main checkpoints and other running experiments remain separate.

Three fresh-base treatments use both Choice and Noul:

| Arm | GPUs | Data objective |
|---|---|---|
| rlcd_scaled | 2,3 | CE + 0.1 × (native RLCD − CE) |
| proper | 4,5 | (CE + 0.75√K × spherical loss) / 1.75 |
| rlcd_both | 6,7 | Native RLCD on both tasks; joint-scope control |

The completed CE/BCE pilot is the matched reference. Earlier task-specific RLCD arms changed one task at a time; the new joint native control avoids attributing a task-scope change to the proposed objective. No grid search or extra seeds. Coefficients fixed before training.

## Mathematical treatment

With p=softmax(z), observed label y and current candidate count K:

$$
L_{proper}=rac{-log p_y+0.75sqrt K(1-p_y/|p|_2)}{1.75}.
$$

The log and spherical mixture is strictly proper for the conditional categorical distribution. At uniform predictions its logit gradient equals CE for every K. This does not establish calibration on full candidates after sampled-candidate training. Stable log-softmax/logsumexp/expm1 computation; train loss FP32. No probability floor, extra backbone pass, extra parameter or changed candidate interface.

RLCD controls preserve sigma=0.4, four draws and upstream spherical weight=0.75. Advantage normalization remains over selected rows and draws in each actual microbatch; both tasks are selected. Only the increment is multiplied by 0.1. The structural penalty is unchanged. RNG isolation preserves backbone dropout RNG.

The proposed parameter-space task-protection constraint is not part of this screening. Log+spherical proper scores and lowering policy-gradient weights are existing ideas; results cannot establish algorithmic novelty by naming alone. See rlcd_principled_optimization_20261106.md for derivations and boundaries.

## Frozen inputs and protocol

- Project: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
- Data: data/pilot_classification_rlcd_20261106_v1 (reused directly; no rebuild).
- Train 10,000; dev 1,000; in-domain test 2,000; each population half Choice, half Noul.
- Original training negative IDs retained; all dev/test candidates and text retained.
- Base: RSI_Research/external/baseline_models/qwen35_9b_base, no trained parent.
- Same seed 20261101, two GPUs per arm, batch64, epoch1, 157 steps, optimizer groups, half-layer checkpointing and regularization as CE.
- Expected trainable parameters: 46,304,262; actual initial LoRA/head/pointer hashes must equal the existing fresh D2 reference.
- Sole epoch1 endpoint; no checkpoint selection using test. Accuracy primary by task/source; NLL/Brier, paired source-bootstrap intervals and training time diagnostic.
- This repeatedly examined small test is exploratory, not untouched confirmatory evidence or cross-seed stability.

## Code, commands and outputs

Python: /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python

New files: scripts/formal/{classification_proper,worker_classification_proper,prepare_classification_proper,supervise_classification_proper,report_classification_proper}.py; tests/test_classification_proper.py. Frozen old worker/model files remain byte-identical.

Run the prepare script once, then each exact launch_command from the arm experiment_manifest.json under persistent tmux. Automatic preflight, training, full-native dev/test, heartbeat, own-run resume checkpoints and stall alerts are inherited. No automatic full-scale continuation.

Root: runs/pilot_classification_proper_20261106_v1. Per-arm config.json, experiment_manifest.json, preflight.json, progress.json, status.json, train.log, loss_diagnostics.jsonl, complete.json and evaluations/.

Reporter: scripts/formal/report_classification_proper.py --watch. It compares identical prediction IDs/supports to CE and the joint native RLCD, writes results_summary.json/report.md, and updates only CLASSIFICATION_PROPER_PILOT in the master experiment record under the shared lock. Failed arms remain visible while other arms continue reporting.

## Trajectory

### 2026-11-06 — implementation and launch preparation

- GPUs2–7 verified free; ASAP occupies0–1.
- All101 inherited source hashes match the completed pilot; input files are reused directly.
- 22 CPU tests pass, including native RLCD parity from the prior suite, gradient/regret identities, uniform CE scale, candidate permutations, extreme BF16/FP32/FP64 logits, scaled-gradient endpoints, unchanged RNG, and one-forward diagnostics.
- Two malformed client transport calls were rejected before execution; retried as bounded edits, with no partial experiment writes or GPU runs from those calls.
- GPU preflight and real optimizer progress will be recorded in launch_verified.json before reporting training as started.
- No remote backup or push requested.
