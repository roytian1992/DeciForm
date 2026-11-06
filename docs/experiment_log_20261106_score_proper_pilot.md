# Score small-data direct proper-scoring pilot

## Scope and user steering

User authorizes testing the classification log+spherical objective on Score using GPUs6–7, then explicitly narrows this to small training/testing populations for fast screening. No full-data Score run was launched. This separate pilot does not alter the formal full method or R26 main planning.

Two matched losses per panel: original D2 CDF-log control and direct log+spherical with alpha0.75. Both train fresh from Qwen3.5-9B-Base; no trained model/checkpoint inheritance. Architecture, trainable inventory, optimizer groups, seed, row order and complete output support match within each pair.

| Panel | Train | Dev | Test | Epoch | Global batch | Updates per arm |
|---|---:|---:|---:|---:|---:|---:|
| STS-B + WMT English-Chinese | 2,048 | 256 | 512 | 1 | 16 | 128 |
| ASAP all eight prompts | 512 | 64 | 128 | 1 | 16 | 32 |

Continuous subsets have equal source quotas and proportional ten-bin label stratification. ASAP samples proportionally by prompt and original integer score. Stable ID hashing with seed20261101 chooses records within original train/dev/test splits. No short-prompt filtering, candidate sampling, label rounding or text truncation. Reuse the corresponding original encoded request exactly. Verify no cross-split ID/group overlap.

Batch16, shared by both arms, provides a useful number of updates at small data size. These runs must be compared to their newly trained small-data controls; full-data batch64 results are not matched baselines. Epoch1 is the only endpoint. Existing repeatedly examined source tests are exploratory.

## Objectives and interpretation

Direct target is the expectation of the normalized classification log+spherical loss under the existing adjacent-grid target distribution q. For integer labels q is one-hot. For fractional labels q has mass on the two bracketing grid values and preserves the original numeric expectation. It uses soft CE plus alpha times square-root-K times (1 − q dot p / norm(p)), divided by (1+alpha). No random noise or added model parameters.

The CDF reference keeps the original ordinal scale3.0383850790577127 and full numeric supports. Its values/gradients are checked against the original D2 objective on integer labels, and use the original specialist implementation for fractional labels.

Direct log+spherical has no ordinal distance term. Whether its categorical probability improvement lowers Score MAE is the actual hypothesis; properness alone does not guarantee MAE improvement. Native predictive median remains primary; mean/argmax are diagnostics, not post-hoc decoder selection. Score is never reordered.

## Inputs and provenance

- Project: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
- Small data: data/pilot_score_proper_20261106_v1/{continuous,asap}
- Source continuous data: data/score_specialist_stsb_enzh_20261105_v1
- Source ASAP data: data/score_specialist_asap_fold0_20261104_v1
- Original architecture/init references: runs/formal_score_numeric_20261105_v1/d2_batched and runs/formal_independent_asap_score_20261105_v5/independent_d2_asap
- Baseline initial LoRA/pointer/head/config hashes and optimizer inventories are checked before real updates. No trained parent is loaded.
- Small manifests store original source/cache hashes, source indices, IDs, source counts, grids and zero text/label changes.

## Execution and outputs

Environment: qwen35-train; Python /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python.

New code: scripts/formal/score_proper.py, models_score_proper.py, worker_score_proper.py, supervise_score_proper.py, prepare_score_proper_pilot.py, queue_score_proper_pilot.py, report_score_proper_pilot.py; tests/test_score_proper.py. Old active source files stay immutable.

Root: runs/pilot_score_proper_20261106_v1. Prepare once, then queue_score_proper_pilot.py --root ROOT in persistent tmux; reporter uses report_score_proper_pilot.py --root ROOT --watch.

Queue on GPU6–7 only: continuous_cdf → continuous_proper → asap_cdf → asap_proper. Each automatically performs distributed zero-update preflight, epoch1 training, complete-native small Dev/Test. Next arm starts only after successful completion and released GPU memory. Failure stops the queue, writes explicit alerts and preserves own-run checkpoints. Per-step progress and stall checks remain active.

Automatic results_summary.json/report.md updates compare only matched small populations, including source-stratified paired NMAE intervals. Only SCORE_PROPER_SMALL in docs/experiment_record_final_d2_20261105.md is updated, under the shared document lock.

## Trajectory

### 2026-11-06 — implement and prepare

- GPUs6–7 verified idle;0–5 occupied and untouched.
- User clarification arrived before any training launch; full-data plan replaced with this matched small pilot.
-38 CPU checks passed in10.79s: original one-hot loss parity, soft-target expectation, analytical/autograd gradients, proper regret identity, finite extreme logits, RNG preservation, single forward, baseline CDF parity and deterministic stratification.
- No remote backup or push requested. Actual launch/preflight/update proofs will be saved in launch_verified.json.

Preparation audit correction: inherited continuous source Dev/Test share215 groups, while train-vs-evaluation share0. The first small draw had10 Dev/Test group overlaps and was rejected before training. Preparation artifacts are retained under a prepare_group_overlap timestamp. New dev draws exclude all original test groups; test sampling and original source files stay unchanged. Eligible/excluded counts are recorded per split. This affects both loss arms identically.

### 2026-11-06 21:20 +0800 — parallel scheduling authorized

User released GPUs0–1 and authorized their use. Verified both at4MiB before launch. New-objective lane now uses0–1: continuous_proper then asap_proper; original-CDF lane remains6–7: continuous_cdf then asap_cdf. Each run still has world_size2 and identical frozen config, data, seed and numerical execution. Only unused proper-arm manifests changed physical GPU assignments. Prior scheduling manifests are preserved under scheduling_before_01_release.

The old dispatcher PID3737631 was SIGSTOPed without signaling its existing child supervisor/torchrun/workers. The new CDF lane adopts the ongoing control, then retires that stopped dispatcher after control completion. Thus no optimizer restart or discarded steps. New lane scheduler is scripts/formal/queue_score_proper_lanes.py, tracked by scheduler_sha256; active worker/source hashes remain unchanged. Persistent lane status/alerts and original per-arm supervision remain enabled. GPU2–5 workloads untouched.

Independent data_verification.json checks every selected row and encoded request exactly equals its frozen source, and all small splits have zero cross-group overlap. First control passed real two-GPU preflight at52.365GiB and began optimizer updates; full proof in launch_verified.json. New proper arm preflight/training is being verified; no finished results claimed.
