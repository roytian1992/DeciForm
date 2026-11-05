# Final independent D2 curves and numeric Score research

## Scope and allocation

User authorization on 2026-11-05: stop the abnormally slow continuous Score
runs, cancel future English–German work, allocate four GPUs to final D2 size
curves and four to separate D2-based Score optimization. No new main-model
training or A4, and no repeated A1–A3 suite.

| GPUs | Job | Execution |
|---|---|---|
| 0–1 | final independent D2 25k, then 50k | sequential persistent queue; 25k training verified |
| 2–3 | final independent D2 100k | queued after already-trained ASAP CE completes full evaluation |
| 4–5 | D2 batched complete Score branches | preflight passed, training verified |
| 6–7 | single-encoding numeric CDF Score | preflight passed, training verified |

ASAP is essay-quality Score prediction, with prompt-specific legal integer
supports of 4/5/6/11/31/61 levels. Its current evaluation is finishing, not a
new training allocation. Queue state is authoritative; do not describe 100k
as already training while ASAP holds GPUs 2–3.

## Inputs, counts and provenance

Curves: `runs/formal_independent_curves_20261105_v1`, prepared by
`scripts/formal/prepare_independent_curves.py`. They reuse exact Kev nested
25k/50k/100k indices from the frozen formal 200k population, with identical
candidate-negative IDs and one training view. Global batch 64, one epoch,
seed 20261101, respectively 391/782/1563 updates. Final independent D2
architecture and losses are fixed. Each run starts from original base weights,
not a learned model. Dev/test are the unchanged 6,000/15,000 full-native rows.
The 200k endpoint is reused from `runs/formal_independent_20261105_v1`.

Score: `data/score_specialist_stsb_enzh_20261105_v1`, built by
`scripts/formal/prepare_score_no_german.py`. Remove only `wmt_en_de` from the
already audited continuous panel; retain exact rows, order, text, fractional
labels, numeric grids and token encodings. Do not relax the original overlap
exclusions. Training versus held-out parent-group overlap is zero. The
retained dev/test group relationship is recorded in the data manifest, not
silently redefined as a new split.

| Source | Train | Dev | Test | Full score support |
|---|---:|---:|---:|---|
| STS-B | 4,882 | 1,500 | 1,379 | 0–5 at 0.1 spacing, 51 points |
| WMT en-zh | 3,247 | 1,000 | 1,000 | 0–100 at 1 spacing, 101 points |
| Total | 8,129 | 2,500 | 2,379 | no candidate sampling |

These grids are representations, not natural classes or rounded targets.
Data/cache manifest SHA-256 values are frozen in each run manifest;
cache audit SHA-256 is
`f73f1255a02dd745e07873dc59b81fc429a8005aafe013d8306f0abaec8fdb8f`.
The old continuous panel had 11,457 training rows, including 3,328 en-de rows;
its dev/test counts were 3,500/3,379. Historical mixed-language Kev and JSON
LoRA runs remain references, not matched training-population controls for
this new pair. No new en-de job is queued.

## Slow-run diagnosis and cancellation

`runs/formal_independent_score_completion_20261105_v1/continuous_cdf` and
`continuous_ce` were explicitly stopped. Their `cancellation.json`, preserved
status/progress snapshots, logs and resume checkpoints identify the interruption;
automatic restart is disabled. They must not be resumed into the new study.

The old execution serialized one row and eight candidate branches at a time,
recomputing the shared prefix for each candidate chunk. A 51/101-level row
required 7/13 complete calls. Across 11,457 rows this was 119,649 calls per
epoch, with all 32 decoder layers checkpointed, about 19.65 GiB allocated and
440–446 seconds per update. The audited full inputs were at most 838 tokens,
with prefix maximum 240: long text alone did not explain the slowdown.

The new reference batches complete candidate branches across rows and uses
the already-tested half-layer checkpoint policy. It retains the independent
D2 architecture and all score levels. Different batching changes BF16
roundoff and training dropout sharing: chunked execution resampled the prefix
dropout, whereas one shared call reuses it. Do not claim bitwise-equivalent
training trajectories or attribute a new score to a loss change.

## Matched Score pair and mathematical hypothesis

Root: `runs/formal_score_numeric_20261105_v1`.

Both arms use the same 8,129 ordered training requests, base initialization,
LoRA rank/targets and task routing, seed, one epoch, batch 64, 128-update
OneCycle schedule, original fractional targets, soft-CDF objective and
complete-support predictive median. They automatically evaluate all 2,500
dev and 2,379 test rows. Equal-source NMAE is the cross-range summary; raw
native MAE remains the per-source primary metric. The main D2 is unchanged.

`d2_batched` preserves the original ordinal Score readout and candidate
representations, with 46,304,262 trainable parameters, four rows per microbatch.

`numeric_cdf` retains broad LoRA, task rank routing, query projection and
query-side head geometry. It encodes the complete state/rubric/query once,
then predicts a conditional numeric distribution. The unused candidate key,
Choice-QF-only and old ordinal-readout parameters are frozen. This arm has
44,734,978 trainable parameters and 16 rows per microbatch. It replaces the
Score readout/encoding path, not merely its loss. This Score-only study does
not establish multi-task routing or Choice QF contributions.

Normalize a complete output support to increasing points
$t_i=(s_i-s_0)/(s_{K-1}-s_0)$ on $[0,1]$. The new head predicts 32 positive
bin widths and 34 probability masses: one uniform-density mass per bin and
one atom at each boundary. Widths sum to one, with minimum width $10^{-4}$;
masses sum to one. With bin boundaries $a_j,b_j$, its integrated CDF is

$$
I(t)=m_0t+\sum_{j=1}^{32}m_j\left[
\frac{\min(\max(t-a_j,0),b_j-a_j)^2}{2(b_j-a_j)}
+\max(t-b_j,0)\right].
$$

Project onto every legal output grid point by averaging the CDF over each
adjacent interval:

$$
A_i=\frac{I(t_{i+1})-I(t_i)}{t_{i+1}-t_i},\qquad
p=(A_0,A_1-A_0,\ldots,A_{K-2}-A_{K-3},1-A_{K-2}).
$$

This is linear barycentric projection of the numeric distribution, not
nearest-level rounding. In exact arithmetic, for any legal decision $s_i$,
it preserves the expected absolute error of the continuous distribution:

$$
\mathbb E_{\operatorname{Proj}(Z)}|S-s_i|
=\mathbb E_Z|Z-s_i|.
$$

Thus a native-grid median is an MAE-optimal legal decision for the projected
distribution. This is a representation/decision property, not evidence that
the fitted distribution is accurate. Integral arithmetic uses float64 and a
$10^{-12}$ finite-probability floor before log normalization.

For an observed fractional label, both arms use the same adjacent-level
barycentric target $q$, with cumulative $Q_i$ and model cumulative $P_i$:

$$
\mathcal L=\alpha\sum_{i=0}^{K-2}
\frac{s_{i+1}-s_i}{s_{K-1}-s_0}\operatorname{BCE}(P_i,Q_i),
\qquad \alpha=3.0383850790577127.
$$

The same existing D2 structural regularizer is retained. Prediction always
includes all 51/101 points; original labels are never rounded for scoring.
Hypothesis: numerical structure can share statistical strength across nearby
levels while avoiding candidate-dependent backbone cost. It may lose
useful candidate text interactions, and must be judged by actual MAE and
runtime rather than assumed to win.

## Verification and measured execution

- Curve/index and independent-candidate tests: 13 passed.
- Numeric projection and Score-panel tests: 21 passed, including MAE-risk
  preservation, full-grid and affine-range behavior, finite gradients and gradcheck.
- Actual 9B CUDA benchmark: longest retained examples and a fixed mixed batch,
  forward/backward, finite gradients and real prediction/metric code, zero
  optimizer updates. Final evidence under
  `runs/score_execution_preflight_20261105_v1/{d2_final_batch4_half,numeric_final_batch16_half}/complete.json`.
- D2 batch4 fixed mix: 1.67595 s, 30.49 GiB peak. Longest en-zh batch4:
  54.05 GiB peak. Numeric batch16 fixed mix: 1.01929 s, 29.75 GiB peak;
  longest en-zh batch16: 33.68 GiB peak. These are microbatch measurements,
  not end-to-end training/evaluation speed claims.
- Initial numeric CUDA probe exposed unsupported backward in the single-token
  FLA recurrent branch. Fixed by encoding the complete prefix plus query in
  the differentiable full-sequence path. The failed probe log is retained;
  final probes and distributed preflight passed. No hard truncation or CPU
  surrogate replaced the real GPU computation.
- Distributed preflight verifies shared fresh base/LoRA/pointer/head hashes;
  numeric head hash is checked before intentional Score-head replacement.
  Both configurations passed, then performed actual optimizer updates.
  Exact status, steps and validation hashes are in `launch_verified.json`.

## Commands, environment and recovery

Working directory: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
The run manifests include environment variables, code/config/data hashes,
reference initialization and full commands; root `code/` holds source snapshots.

Score launch entry is `scripts/formal/supervise_score_research.py --manifest
<arm>/experiment_manifest.json`; each arm runs under its own tmux session
`deciform_score_rq_{d2_batched,numeric_cdf}_20261105`. The supervisor performs
preflight, training and automatic full evaluation, writes status and failure
alerts, and does not silently restart. Recovery from that arm's own checkpoint
uses the same manifest plus `--resume`, only after investigating the failure.
Do not resume cancelled historical runs or mutate live frozen code/configs.

Curve queue sessions are `deciform_final_d2_curves_g01_20261105` and
`deciform_final_d2_curves_g23_20261105`, controlled by
`scripts/formal/schedule_research_queue.py`. Successful dependency completion
and actual GPU memory release are required before starting the next job.

No remote backup/push was requested. Local version control and frozen snapshots
record this implementation; unrelated workspace changes are excluded.

## Trajectory

### 2026-11-05 22:25 Asia/Shanghai — allocation and launches verified

Stopped the two explicitly cancelled slow runs; preserved their evidence.
Filtered only the excluded language source, froze hashes and validated counts.
Started final-D2 learning curves with exact existing nested subsets. Implemented
and tested batched D2 and conditional numeric CDF Score paths; fixed the actual
CUDA-backward failure before launch. Both new Score arms passed distributed
fresh-initialization checks and reached real training updates. Main full and
A1–A3 remain frozen. Next: collect complete evaluations, compare the matched
Score pair and complete the queued learning-curve points.
