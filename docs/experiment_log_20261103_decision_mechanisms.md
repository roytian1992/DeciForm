# 2026-11-03 — Isolated score regression and conditional choice geometry

## Authorized scope

The user approved launching the two proposals in `docs/loss_postmortem_20261103.md`
and explicitly fixed noul to CE/BCE. Both initialize fresh from the original
Qwen3.5-9B base; no trained adapter/head is inherited from A4 or any POC. The
completed A4 serves as an objective/architecture control, not a checkpoint parent.

Run root: `runs/formal_decision_mechanisms_20261103_v1`.

| Arm | GPUs | Change | Trainable parameters |
|---|---|---|---:|
| m1_score_regression | 0,1,2,3 | Score CE + 0.5 native MAE, separate rank-64 numeric head | 46,566,535 |
| m2_choice_metric | 4,5,6,7 | Query-conditioned positive QF precision, CE unchanged | 46,320,646 |

Counts above are expected from the audited A4 inventory plus the new modules;
each preflight verifies the actual parameter count before training.
Both retain noul Bernoulli NLL, task routing, task bilinear geometry, and the
ordinal score distribution head. Structural regularization is unchanged.

## Mechanisms

Score shares the decision hidden state across the existing categorical head
and an additional layer-normalized 4096 -> 64 -> 1 SiLU regression head. Its
last linear layer starts at zero, giving the midpoint of the declared range.
For support endpoints a,b, its unbounded coordinate r gives m=a+(b-a)r. Loss is
CE + 0.5*|m-y| in native score units. Training never clips m and never replaces
the distribution mean with a median target. It adds no backbone forward.
The 0.5 auxiliary coefficient is fixed before the run, not selected on test.

Choice retains the query-relative layer-normalized QF features z_i. Instead of
the old static signed quadratic readout it uses static plus query-generated
log precisions v(x), transformed to a=softplus(2*tanh(v/2)), then a/mean(a).
The new query map is 256 -> 64, bias-free and zero-initialized. The unbounded
correction before the existing tanh cap is

    delta_i = 0.5 * mean_r [(1-a_r(x))*z_ir^2].

This is negative anisotropic quadratic energy relative to the isotropic
reference; its precisions are strictly positive and have mean one. Subtracting
the isotropic reference gives exactly zero at uniform initialization, including
finite LayerNorm epsilon. The existing final cap of 4 is retained. This changes
the QF parameterization only for choice; noul/score forwards are unmodified.
Head equivariance holds conditional on fixed candidate embeddings; the complete
causal encoder is not claimed to be permutation invariant.

## Fixed inputs and schedule

Data: `data/formal_200k_20261101_v1`, 200,000 requests (choice 50,000, noul
75,000, score 75,000), one epoch, seed 20261101, global batch 64, 3,125 updates.
Choice uses the same frozen positive plus at most three negatives and one view.
Dev is 6,000 and in-domain test 15,000, always with complete native candidates.
There is no hard input truncation and no external/source-held-out evaluation.

Data manifest SHA256: 7519680ae972f040eb71e6f97d0da98651f17c346572dcad4d782437283aa2e3

Cache audit SHA256: cb186c69bf247ed0c6abf599902be5c0d9db355d5d7f2fcb421846724ef9ea2e

Original base: `../RSI_Research/external/baseline_models/qwen35_9b_base`.
AdamW and OneCycle schedule (30% warmup), optimizer groups, clipping, dynamic
packing budgets and the frozen 16/32-layer activation-checkpoint policy remain
those of A4. New head parameters use the existing residual group, max LR 1e-3.
LoRA/pointer max LR remains 2e-5. No multiple seeds or extra epochs.

Each arm now uses four GPUs to use the released eight-card capacity. A4 used
two. Global sample order/batches and initialization are matched, but local
microbatch partitions, dropout RNG use and floating-point reductions differ.
This is a disclosed comparison limitation; runtime is not a matched-hardware
speed comparison against A4. Per-run costs and world size are saved.

## Evaluation and diagnostics

Primary predictions retain categorical argmax on native support; choice/noul
accuracy and native score MAE remain headline metrics. No decoder selection on
test. All three tasks are evaluated for each arm to detect cross-task effects.

The score arm stores its regression output alongside categorical predictions.
Dev-only files separately report continuous raw/clipped and nearest-native
regression MAE (ties go to lower support). Neither replaces primary results.
Both arms retain dev-only predictive-median replay. Automatic evaluation is dev
then in-domain test, immediately after training.

## Code, environment and execution

New isolated sources under `scripts/formal/`: `decision_mechanisms.py`,
`models_decision_mechanism.py`, `worker_decision_mechanism.py`,
`supervise_decision_mechanism.py`, `prepare_decision_mechanism.py`.
Existing frozen A4 and prior loss workers are not edited. Exact source/config
hashes and copies are stored in run manifests and `code/`.

Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
Working directory: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.

Preparation command:

```bash
python scripts/formal/prepare_decision_mechanism.py
```

Use the exact manifest `launch_command` within tmux sessions
`deciform_score_reg_20261103` and `deciform_choice_metric_20261103`.
Resume with that command plus `--resume`; source/config/world size must match.
Each run saves `resume.pt` every 100 steps, status/progress, failure records,
staleness alerts, training/preflight logs, and full evaluation prediction files.
Qwen8001 remains stopped; BGE8081 and the yielding idle watchdog remain available.

## Verification and trajectory

76 tests passed across the new mechanism suite and the existing typed-loss,
v5/v6 and ablation suites. Checks include exact initial categorical equivalence,
RNG preservation, task isolation, positive normalized query-dependent precision,
fixed-embedding candidate permutation at K=5/21/77, native numeric MAE and its
unclipped gradient, unequal microbatch gradient equivalence, optimizer inventory,
and unchanged argmax predictions with only one backbone forward.

Distributed preflight performs zero optimizer updates, checks mixed-task
gradients including nonzero new-module gradients, longest full training inputs,
largest complete evaluation requests, dense/first global batches, actual memory,
and initial LoRA/pointer/base hashes against A4. Actual launch evidence is to be
appended after both preflights and first optimizer updates pass. No backup was
requested.
