# Loss study follow-up: estimation, decisions and optimization

## Scope and reproducibility

The four loss treatments completed. This follow-up is an analysis, not a new
training launch or a change to the frozen argmax evaluation protocol. Formal
data, native complete candidate inventories, source predictions and checkpoints
remain unchanged. External-source evaluation remains deferred.

Command from the DeciForm project root:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/analyze_loss_alignment_dev.py
```

Output: `runs/loss_postmortem_20261103_v1/diagnostics.json`. The output records
the executable, package versions, source-code hash, every input file hash and
the exact metric sources. Five models each have the same 6,000 unique dev IDs;
each diagnostic uses all 2,000 native score rows. No rows were skipped. Existing
argmax MAE is reproduced before computing median MAE. The analytic ordinal
derivative is checked against float64 autograd on 16 rows per model (80 checks).
The script refuses to overwrite an existing report. Backup not requested.

## Findings and limits

| Model | Dev score argmax MAE | Dev score median MAE |
|---|---:|---:|
| A4 | 0.4275 | 0.4150 |
| L1 choice pairwise | 0.4245 | 0.4165 |
| L2 choice focal | 0.4210 | 0.4120 |
| L3 noul Brier | 0.4250 | 0.4150 |
| L4 score ordinal | 0.4235 | 0.4135 |

These are development-only diagnostics, not replacement test results, a
preselected winner, or a claim that our model outperforms Kev. The separately
saved reference dev replay at
`runs/formal_typed_losses_20261103_v1/reference_dev_decoding.json` gives Kev
0.4200 argmax / 0.4125 median on the same score population. Any future protocol
adoption must use the same rule for every model with comparable probabilities.

At the L4 final model's saved dev predictions, the median ratio
`||grad_z L_ordinal|| / ||grad_z CE||` is 0.2488098. Its coefficient is 0.1,
so the median auxiliary-to-main gradient norm ratio is only 0.0248810.
The median cosine between `grad_z CE` and
`grad_z (CE + 0.1 L_ordinal)` is 0.99999890; even the minimum is 0.99948802.
All 2,000 CE/ordinal logit-gradient dot products are positive.

Thus this diagnostic does not support an output-level gradient-conflict
explanation. It shows a small directional change from this particular weighted
mixture at the final dev predictions. It does NOT measure initial/throughout-
training gradients, shared-adapter gradients, or prove the cause of its test
performance. The Jacobian from logits to shared parameters can change both
angles and relative magnitudes.

## What CE does and does not imply

Categorical CE estimates a conditional distribution; under correct population
optimization and sufficient capacity, argmax is Bayes-optimal for nominal
choice accuracy. Bernoulli NLL for noul is exactly binary CE. Different task
names do not logically require different losses. In particular, neither focal
nor the tried noul BCE/Brier mixture adds new supervision: for each example
their logit gradients are scalar multiples of CE's gradient.

CE can also estimate a valid ordered score distribution. The action minimizing
predicted absolute error is a median, not generally its mode. Hence CE itself
is not inconsistent with MAE; the whole estimation-plus-decoding rule matters.

## A distinct score hypothesis, not another small auxiliary mixture

For increasing native scores s_1,...,s_K, define

    F_k = sum_{j<=k} p_j,
    t_k = 1[y <= s_k],
    w_k = (s_{k+1}-s_k)/(s_K-s_1).

Use the ordered probability objective as the score MAIN objective:

    L_order = sum_{k=1}^{K-1} w_k BCE(F_k, t_k).

Its matching discrete decision is the first native support value whose
cumulative probability reaches 0.5. For any native prediction a,

    |a-y| = sum_k (s_{k+1}-s_k) |1[a<=s_k]-1[y<=s_k]|.

Every positive-weight binary log score is strictly proper for its threshold
event. The population minimizer therefore recovers the true CDF (with enough
capacity); the median is an MAE-optimal action. More specifically, at fixed x,
let F* be the true CDF, F the predicted CDF, and R=s_K-s_1. Then

    excess MAE / R <= sum_k w_k 2|F_k-F*_k|
                   <= sqrt(2 sum_k w_k KL(Ber(F*_k) || Ber(F_k))).

The last term is sqrt(2 * excess conditional ordered log risk), by Pinsker and
weighted Cauchy-Schwarz. This is a population consistency argument, not a
finite-budget improvement guarantee. CE also has decision-consistency bounds;
the proposed distinction is the explicit ordered event geometry/inductive bias.
The ordered loss and median rule are established ideas, not a standalone new
paper contribution.

A next controlled score run would replace score CE with this objective rather
than retain CE plus a 0.1 term. Hold the other two objectives, architecture,
negative IDs, initialization, budget and optimizer schedule fixed. Use at most
one positive global scale, estimated on training-only initial gradients and
frozen, to avoid accidentally shrinking score's effective task weight. Do not
normalize separately by label-dependent realized gradient norms: that can
change the proper objective. Primary argmax reports stay preserved; adoption of
median as a formal headline rule is a separate explicit protocol decision.

## Shared optimization: diagnose before adopting

For the three per-task losses on shared parameters theta, measure
g_t=grad_theta L_t, their magnitudes, and pairwise cosine similarities on fixed
training batches. Saved output-gradient norms are not substitutes. Our task
rank maps still share A/B matrices; task routing is not complete isolation.
If destructive interference is present, gradient balancing/projection is a
separate mechanism to test, with actual runtime and parameter-scope reporting.
Do not claim that these four final metric tables prove negative transfer.

## Sources

- Gneiting, Making and Evaluating Point Forecasts:
  https://arxiv.org/abs/0912.0902
- Pedregosa, Bach, Gramfort, On the Consistency of Ordinal Regression Methods:
  https://www.jmlr.org/papers/v18/15-495.html
- Yu et al., Gradient Surgery for Multi-Task Learning:
  https://papers.nips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
- Liu et al., Conflict-Averse Gradient Descent for Multi-task Learning:
  https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- Du et al., Adapting Auxiliary Losses Using Gradient Similarity:
  https://arxiv.org/abs/1812.02224

## Trajectory — 2026-11-03 (Asia/Shanghai)

The user requested further principled options after the four loss trials.
Inspected the operative v6/A4/loss code and existing dev score diagnostics;
reviewed primary ordinal/decision-theoretic/multitask literature. Added and ran
a CPU-only diagnostic with input hashes and independent derivative checks.
The initial scratch calculation stopped on a NumPy scalar JSON serialization
error; the saved diagnostic uses native Python scalars and completed normally.
No new model training, original-report replacement, or test decoder change.
Next proposal: a genuine ordinal main objective plus matched decision analysis,
and training-batch shared-gradient measurements before any conflict mechanism.

## User steering: focus choice and score; freeze noul BCE

The latest user instruction narrows optimization to choice and score. Keep
noul Bernoulli NLL fixed; do not queue further noul loss sweeps. The user asks
whether score should be regression or regression plus classification. The
preferred next score hypothesis is now a shared-representation, separate-output
classification/median-regression model; a pure ordinal main objective remains
an alternative, not an automatically queued extra run.

Score proposal:

    p = categorical_head(h, native_support)
    m = numeric_median_head(h, score_range)
    L_score = lambda_cls * (-log p_y) + lambda_reg * |m-y|.

The positive coefficients must be fixed using training/development information.
At population optimum and sufficient function capacity, these heads can
simultaneously estimate the label distribution and a conditional median. Do
not call MSE a direct MAE surrogate: its optimum is the conditional mean.
Likewise, using sum_j p_j s_j as the regression prediction can force a conflict
between calibrated probabilities and a median-seeking absolute-error objective.
Separate outputs avoid imposing that identity, though shared-parameter
optimization can still interfere. No arbitrary probability-mean consistency
term should be added.

Normalize coordinates using each declared native range [a,b], with R=b-a,
u=(y-a)/R and m=a+R*m_normalized. Raw MAE is R*|m_normalized-u|; silently removing
R changes weighting across score ranges. Preserve the intended metric weighting
and disclose any task scale. A discrete score task must still emit one of its
native values; a continuous head does not authorize changing its output domain.
For the first isolated comparison, preserve the original classification argmax
headline and treat the regression head as auxiliary training. Regression/native
projection and probability-median decisions can be compared on dev separately;
any adopted new headline decision rule needs an explicit matched protocol.

Choice proposal: improve question-conditioned geometry instead of another
scalar reweighting of CE. The current QF in scripts/kev_ours_v5.py uses
query-relative features but a shared static quadratic readout vector. A
candidate mechanism is

    u_i = b_i - gamma * sum_r a_r(x) (v_ir - mu_r(x))^2,
    a_r(x) > 0, with mean_r a_r(x) = 1.

The question generates dimension weights, giving a positive diagonal precision
in a learned low-dimensional space. Equal-prior Gaussian compatibility with a
shared query-conditioned covariance gives this quadratic ranking energy; the
log determinant is common across candidates and cancels in their softmax.
This is an inductive interpretation, not a claim that real candidate embeddings
are Gaussian. The low-rank PSD metric and small residual gate retain a stable
independent score and need no second backbone pass. This extends the existing
QF's conditioning; it is not the claim that CE lacks ranking gradients or that
metric learning itself is novel. Final quality and speed require measurement.

Implementations and training for these proposals have not been started. Keep
the frozen data/negative IDs, native full evaluation, backbone and budget. First
isolate the score dual-output change and the choice metric change; combining
them before either is measured would obscure which mechanism helps.
