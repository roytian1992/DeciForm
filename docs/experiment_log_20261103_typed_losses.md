# 2026-11-03 — isolated noul/score loss experiments

## Authorization and scope

The user approved stopping Qwen on port 8001 and using GPUs 0--3 for two
additional task-loss controls. The existing L1/L2 choice runs on GPUs 4--7
continue unchanged. All four controls share completed A4 as their reference;
none loads A4 weights. The frozen v6 architecture and historical results remain
preserved. External evaluation stays deferred.

| Arm | GPUs | Only changed data objective |
|---|---|---|
| L3 `l3_noul_brier` | 0,1 | noul: 0.5 Bernoulli NLL + 0.5 * 2(p-y)^2 |
| L4 `l4_score_ordinal` | 2,3 | score: CE + 0.1 weighted cumulative-threshold log loss |

L3 retains choice CE and score CE. L4 retains choice CE and noul Bernoulli NLL.
All architecture components and the 0.001 structural penalty remain enabled;
46,304,262 trainable parameters. Score CE's coefficient stays exactly one in L4.

## Fixed contract and stable mathematics

Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
Data: `data/formal_200k_20261101_v1`, same audited cached inputs and fixed
negative IDs as A4. Fresh base:
`../RSI_Research/external/baseline_models/qwen35_9b_base`; no inherited head,
LoRA, or parent checkpoint. One epoch, 200k requests, batch 64, 3125 updates,
seed 20261101, original OneCycle schedule, two GPUs per arm, fixed half-layer
checkpointing and 8192-token packing budget. Full text is retained even when
a singleton exceeds the packing budget. No hard truncation or candidate dropping.

For L4, score support s_1<...<s_K supplies weights
`w_k=(s_(k+1)-s_k)/(s_K-s_1)`. The threshold target is `1[y<=k]`.
The ordinal term is `-sum_k w_k [t_k log F_k+(1-t_k)log(1-F_k)]`.
The implementation uses left/right logcumsumexp and the global logsumexp
directly, retaining corrective gradients when a CDF saturates numerically.
It never clips probabilities or substitutes a guessed label distribution.
The training term is normalized by native score range; headline MAE remains
in native score units. Current formal score supports have unit spacing.

Automatic evaluation remains 6000 dev and 15000 in-domain test examples,
using full native supports. Existing argmax outputs and metrics are preserved.
An additional **dev-only** CPU median-decoding diagnostic is saved separately
from the headline report, without selecting checkpoints or changing test
predictions. The same diagnostic can be replayed on saved probabilistic
baseline outputs. JSON-only generated answers do not define such distributions.

## Service change

Stopped only the identified Qwen 8001 process tree (API PID 2978324 and its
vLLM workers) and the `llm_keepalive_8001` tmux session. All recorded Qwen
processes exited after SIGTERM and port 8001 was verified closed. BGE on 8081
remains running. The service action record is
`runs/service_changes_20261103_loss/stop_8001.json`.

The existing idle watchdog recognizes `scripts/formal/worker_` and yields
GPU use to training. No unrelated process was terminated.

## Validation, commands and outputs

Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.

```bash
python -m pytest -q tests/test_typed_loss_experiments.py tests/test_choice_loss_experiments.py tests/test_v6_ablations.py tests/test_kev_ours_v6.py tests/test_kev_ours_v5.py
python scripts/formal/prepare_typed_loss.py
```

94 tests passed. Coverage includes analytical/autograd agreement, Bernoulli
boundary-gradient matching, expected-risk stationarity, native nonuniform
score intervals, affine support normalization, finite extreme-logit gradients,
mixed-task isolation, microbatch weighting and nonmutating native median replay.

Output root: `runs/formal_typed_losses_20261103_v1`.
New code: `scripts/formal/{typed_losses,models_typed_loss,score_decoding,
worker_typed_loss,supervise_typed_loss,prepare_typed_loss}.py`.
Running L1/L2 source files are untouched. Each manifest records source/config
hashes, exact command and environment, and the root contains a source snapshot.

Tmux sessions: `deciform_loss_noul_20261103`, `deciform_loss_score_20261103`.
Each supervisor first verifies a zero-update distributed GPU preflight and
fresh initialization hashes against A4, then launches training and internal
evaluation. Save/resume every 100 updates; local failure/stall alerts enabled.
Detached output-logit diagnostics are recorded at step 1 and every 50 steps;
they are not shared-backbone gradient-conflict measurements.

## Trajectory

2026-11-03: Qwen 8001 and its keepalive stopped; BGE retained. Implemented and
tested the isolated objectives and dev decoding diagnostic. Both existing
choice arms remained in training with no failure files. Launch/preflight
results are appended after verification below.
