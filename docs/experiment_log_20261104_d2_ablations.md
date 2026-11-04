# D2 full architecture ablations (2026-11-04)

## Decision and reference

The user designates the completed D2 ordinal-main model as the current full
method. Its exact checkpoint, configuration, initialization, code and results
are under `runs/formal_score_distribution_20261104_v1/d2_ordinal_main`.
Earlier A1/A2/A3 were trained against v6's mixed CE/CRPS objective and
argmax evaluation, and therefore do not isolate the same components in D2.
Those historical results remain in place but are not substitutes for these
new comparisons. D2 itself is reused; no identical full retraining is needed.

## Fixed three-arm comparison

Each new run starts from the original Qwen3.5-9B base, with no D2 checkpoint
inheritance. D2's 200,000 frozen training views (50k choice, 75k noul,
75k score), candidate IDs, seed 20261101, batch 64, one epoch/3,125 updates,
optimizer, learning-rate schedule, 4-GPU world size and half-layer activation
checkpointing are unchanged. Each automatically evaluates the same 6,000 dev
and 15,000 in-domain test requests with complete native candidates. Choice
uses CE, noul uses Bernoulli BCE, and score uses D2's frozen pure ordinal
threshold loss with global scale 3.0383850790577127. Score primary output is
the native predictive median with MAE; all four decoder reports are retained.
External-source evaluation remains deferred.

| Arm | Architecture change from D2 | GPUs | Trainable parameters |
|---|---|---|---:|
| A1 | Disable rank-space task routing | 0--3 | 46,113,798 |
| A2 | Disable choice quadratic feature | 4--7 | 46,025,670 |
| A3 | Disable task-conditioned bilinear head geometry | 0--3 after A1 | 46,107,654 |

The frozen D2 full has 46,304,262 trainable parameters. A1/A3 share one
sequential queue on GPUs 0--3; A2 runs concurrently on GPUs 4--7. The
scheduler launches A3 only after A1's supervisor exits successfully. Every
supervisor runs a zero-update distributed preflight before fresh training,
saves resumable checkpoints every 100 steps, logs status/failures, and then
performs full-native dev/test automatically. No training or eval item is
silently skipped. The complete manifests, source snapshots, hashes, queue
commands, and authoritative status files are under
`runs/formal_d2_ablations_20261104_v1`.

The retained modules in a single ablation may compensate during retraining,
so its score difference is a conditional component effect, not an additive
contribution that can be summed across A1/A2/A3. A1/A2/A3 do not separately
test the ordinal readout, which remains enabled in every arm.

## Validation and trajectory

The model retains D2's `losses` and `predict` methods. Its initializer passes
one architecture mask through the v6 installer, then restores the already
replaced D2 auxiliary-loss flag in the manifest and inventory. The new worker
is an otherwise exact copy of the completed D2 worker, with only the model
binding and ablation guards changed. Source/config snapshots are frozen before
launch. The project environment is `qwen35-train` at
`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.

2026-11-04: Reviewed D2's complete run, architecture flags, frozen objective,
and older A1/A2/A3 parameter counts. Added isolated D2 ablation model,
worker, manifest preparer and sequential GPU queue. Fifty-five focused
score-distribution/ablation/v6 tests passed. No backup requested.
