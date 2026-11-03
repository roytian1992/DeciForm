# 2026-11-03 — choice loss controls on the frozen architecture

## Scope and authorization

The user approved proceeding with the two choice-only loss comparisons from
`docs/loss_research_20261103.md`. The reference is completed A4, whose score
data loss is pure CE. All architecture components, noul Bernoulli NLL, score
CE, structural penalty, original argmax decoding and dataset views are retained.
Frozen full v6 remains a preserved reference. This is a loss study, not a new
architecture or dataset. External-source evaluation remains deferred.

| Arm | GPUs | Choice objective |
|---|---|---|
| L1 `l1_choice_pair` | 4,5 | 0.5 CE + 0.5 (2/K) sum_j softplus(z_j-z_y), j != y |
| L2 `l2_choice_focal` | 6,7 | gamma=1 focal divided by its CE-gradient multiplier at p_y=1/K |

Both objectives match the CE logit gradient at a uniform prediction. This
does not equate gradients throughout training or add any supervision.
All 46,304,262 trainable parameters remain enabled in both arms.

## Fixed inputs and training contract

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
- Base: `../RSI_Research/external/baseline_models/qwen35_9b_base`.
- Initialization: fresh base, no inherited adapter/head or parent checkpoint.
- Data/cache: `data/formal_200k_20261101_v1`; the same audited views and
  encodings used by A4. Hashes are recorded in each manifest.
- Training: 200,000 rows (50k choice, 75k noul, 75k score), one epoch,
  global batch 64, 3,125 updates, seed 20261101, existing OneCycle schedule.
- Choice training: one identical fixed view, positive plus at most 3 negatives.
- Runtime: two GPUs per arm, fixed half-layer checkpointing, token budget 8192,
  max 16 rows per microbatch, complete long inputs; no truncation.
- Automatic evaluation: 6,000 dev and 15,000 in-domain test rows, complete
  native supports throughout. Choice/noul accuracy and score native MAE.
- Reference: `runs/formal_v6_ablations_20261103_v1/a4_ce_only` (also two GPUs).

## Code, validation and execution

New source files under `scripts/formal`: `choice_losses.py`,
`models_choice_loss.py`, `worker_choice_loss.py`, `supervise_choice_loss.py`,
`prepare_choice_loss.py`. Historical training/model files are unchanged.

The worker preserves the A4 distributed sample-weighted engine. At step 1 and
every 50 steps, detached diagnostics record per-task data loss and output-logit
gradient L2 norms relative to CE. These are per-row output-gradient proxies,
not backbone/task-conflict measurements; no extra backbone backward is used.
Regularization is excluded from these diagnostic norms, and remains included
exactly once per global optimizer step through sample-weighted aggregation.

Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
Targeted validation command:

```bash
python -m pytest -q tests/test_choice_loss_experiments.py tests/test_v6_ablations.py tests/test_kev_ours_v6.py tests/test_kev_ours_v5.py
```

Result: 61 passed. Tests cover actual mixed-task loss isolation, unequal
microbatch aggregation, finite extreme-logit gradients, uniform-gradient
matching, permutation/shift invariance of the loss, binary pair=CE,
diagnostic/autograd agreement, and retained model components.

Preparation: `python scripts/formal/prepare_choice_loss.py`.
Output root: `runs/formal_choice_losses_20261103_v1`.
Each arm has `experiment_manifest.json` with the exact supervisor/worker
command, environment, code/config hashes and a source snapshot. Supervisors
first run zero-update distributed preflight, verify fresh initialization against
A4, then launch fresh training followed by internal evaluation. Checkpoints
are saved every 100 steps. Failures and stale progress produce local alerts.

Tmux sessions: `deciform_loss_pair_20261103`, `deciform_loss_focal_20261103`.
The existing idle watchdog recognizes the new `scripts/formal/worker_` command
and yields to training. GPU 0--3 services are outside this run.

## Trajectory

2026-11-03: implemented the two objectives, retained the A4 non-choice paths,
passed 61 tests and prepared the traceable launch workflow. GPU preflight and
initial live training status are appended below after verification.
