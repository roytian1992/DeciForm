# 2026-11-04 — Score distribution learning and matched decision rules

## Authorization and question

After M1/M2 completed, the user requested further improvements and new training,
specifically suggesting predictive distributions and expectation-based score
evaluation, while allowing a justified alternative. Noul remains fixed to BCE.
This authorizes the explicitly versioned decision-rule study below; it does not
authorize overwriting legacy reports. Choice uses the existing A4 QF and CE.
No external-source work is resumed.

The model already outputs probabilities. The preceding runs primarily used
categorical argmax; M1's independent scalar regression head was auxiliary to
the shared representation and did not determine the reported categorical
decision. Neither fact proves CE is inadequate, or that a more elaborate loss
must improve finite-data performance.

## Development evidence before freezing the new runs

CPU replay of 2,000 identical native score development examples:

| Model | Argmax MAE | Median MAE | Continuous mean MAE | Nearest-native mean MAE |
|---|---:|---:|---:|---:|
| A4 | 0.4275 | 0.4150 | 0.4556903 | 0.4105 |
| Kev | 0.4200 | 0.4125 | 0.4575138 | 0.4125 |
| M1 independent regression | 0.4245 | 0.4120 | 0.4576744 | 0.4175 |
| M2 conditional choice | 0.4215 | 0.4100 | 0.4539730 | 0.4090 |
| Full v6 | 0.4245 | 0.4135 | 0.4562889 | 0.4135 |

A4 continuous-mean MSE is 0.4274160 versus argmax MSE 0.6165, illustrating why
the preferred decision depends on the metric. A nearest-native mean may improve
empirical MAE even though it is not generally the distribution's Bayes action
for absolute error. These are dev findings, not a test-selected rule.

## Frozen evaluation protocol

Protocol `score_distribution_20261104_v1` is frozen in the run root before
any alternative test decoding is replayed. Primary score decision is the
lower predictive median on the complete native support, with native MAE.
Choice and noul retain argmax/accuracy. Both new runs and every probability
baseline receive the same four reports: argmax, median, continuous expectation,
and expectation projected to the nearest native score (ties to lower value).
Continuous expectation is explicitly a diagnostic; discrete primary outputs
remain within the original support. All original prediction/metric files stay
preserved. This is a protocol change, not an improvement in the trained model.

Reference replays include A4, Kev, ordinary LoRA, full v6, L4 ordinal auxiliary,
M1 and M2 on both dev and in-domain test. The JSON-generative model is excluded
from probability-decoder replay because one generated answer is not a complete
predictive distribution. No test-guided rule/weight/checkpoint selection.

## Two new, isolated training objectives

Both use the unchanged A4 architecture, 46,304,262 trainable parameters, a fresh
original Qwen3.5-9B initialization and no inherited trained adapters/heads.
No additional regression head or extra backbone pass is introduced.

**D1 mean-moment supervision**, GPUs 0--3:

    mu_p = sum_j p_j s_j, R = s_K-s_1
    L_score = -log p_y + 0.5*(mu_p-y)^2/R.

The regression target supervises the mean of the SAME predictive distribution
used for decisions. This is squared error of the expectation, not expected
per-candidate squared error and not absolute error of a mean. The positive
native span R gives the auxiliary linear scale transformation under score-unit
changes, matching MAE units rather than allowing squared ranges to dominate.
The CE coefficient remains one and the moment coefficient 0.5 is fixed.

At fixed input with true distribution eta, the expected score objective is

    H(eta) + KL(eta||p) + 0.5/R * ((mu_p-mu_eta)^2 + Var_eta(Y)).

Thus the CE and mean-regression optima are compatible and the combined scoring
rule remains strictly proper on the complete distribution. This contrasts with
an absolute-error penalty on its mean, whose optimum need not equal the mean
of the true distribution. Strict propriety is not a finite-budget guarantee.

**D2 pure ordinal main objective**, GPUs 4--7:

    F_k = sum_{j<=k} p_j
    L_score = alpha * sum_k ((s_{k+1}-s_k)/R) * BCE(F_k, 1[y<=s_k]).

There is no score categorical-CE term in D2. This is distinct from L4's
CE + 0.1*ordinal auxiliary, whose measured final-dev logit-gradient direction
was almost unchanged from CE. The positive-weight threshold log scores jointly
identify the full ordered distribution; its median is MAE-optimal under that
distribution. The single frozen alpha is approximately 3.038385079, calculated
from ALL 75,000 training score labels to match the RMS CE/ordinal output-logit
gradient norms at uniform probabilities. It is one global constant, not
per-label/per-sample reweighting or a measurement of backbone-gradient matching.
Exact value, input hash and counts are in `ordinal_scale.json`.

Both retain choice CE, noul BCE, the original structural penalty, native numeric
score supports, original QF/task routing/head geometry and ordinal readout.
These are established statistical loss ingredients; no claim that simply
combining them constitutes a new algorithm or guarantees accuracy gains.

## Inputs, runtime and recovery

Run root: `runs/formal_score_distribution_20261104_v1`.
Data: `data/formal_200k_20261101_v1`; 200,000 requests (50k choice, 75k noul,
75k score), one epoch, seed 20261101, batch 64, 3,125 updates. Score ranges are
0--4 (45,000 train rows) and 1--5 (30,000 train rows). Choice retains the exact
fixed positive plus at most three negative IDs and one view.
Dev/test remain 6,000/15,000 full-native rows; no truncation or sampled eval.

Data manifest SHA256: 7519680ae972f040eb71e6f97d0da98651f17c346572dcad4d782437283aa2e3

Cache audit SHA256: cb186c69bf247ed0c6abf599902be5c0d9db355d5d7f2fcb421846724ef9ea2e

Base: `../RSI_Research/external/baseline_models/qwen35_9b_base`.
Four GPUs per arm, fixed half-layer activation checkpointing, AdamW/OneCycle
with 30% warmup and unchanged optimizer groups/packing/clipping. A4 used two
GPUs: record the rank/RNG/reduction caveat, not a matched-hardware speed claim.
BGE8081/idle watchdog remain; Qwen8001 is not restarted.

Working directory: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.

```bash
python scripts/formal/prepare_score_distribution.py
python scripts/formal/replay_score_decisions.py --root runs/formal_score_distribution_20261104_v1 --split dev
python scripts/formal/replay_score_decisions.py --root runs/formal_score_distribution_20261104_v1 --split test_in_domain
```

Launch each manifest's exact `launch_command` in tmux
`deciform_mean_moment_20261104` / `deciform_ordinal_main_20261104`. Resume with
that command plus `--resume`, preserving code/config/world size. Supervisors
run zero-update preflight, fresh training, dev, and test automatically. Save
optimizer/RNG checkpoints every 100 steps; status, progress, stale/failure
alerts and logs remain per-arm. Full source/config hashes and snapshots are
frozen before launch. Separate references live in `reference_score_rules/`;
new runs save `evaluations/epoch_1.{split}.score_rules.json` beside predictions.

## Validation and trajectory

100 tests passed in 5.54 seconds across score-distribution, prior typed-loss,
v5/v6 and ablation suites. Tests cover population stationary points, analytic
versus autograd derivatives, extreme wrong predictions, native/nonuniform
supports and units, global calibration, exact noul/choice isolation, uneven
microbatch aggregation, MAE/MSE Bayes actions, old/new decoder contracts and
unchanged single-backbone-forward behavior. Runtime preflight and first-step
evidence are recorded separately after successful launch. No backup requested.

Primary sources: Gneiting, *Making and Evaluating Point Forecasts*,
https://arxiv.org/abs/0912.0902; Pedregosa et al., *On the Consistency of Ordinal
Regression Methods*, https://www.jmlr.org/papers/v18/15-495.html.

### 05:36 Asia/Shanghai — launched and initial training verified

Both tmux supervisors launched at 05:25:29. Distributed preflights passed with
zero optimizer updates and peak allocated memory 54.651176 GiB. At 05:36:23
both were training at step 100/3125, with 6,400 visits. Step times were
4.9674 / 4.9629 seconds and training peak memory 55.304055 GiB. No failure or
alert files were present. All trainable counts are 46,304,262. The complete
initial head hash matches between D1/D2; their LoRA/pointer/base hashes match
A4, and each training initialization exactly matches its own zero-update
preflight. Inherited adapter/head fields are null. First-step reference CE
matches exactly across arms for all three tasks on the same 64 rows.

The frozen ordinal scale is 3.0383850790577127. Launch verification, tests and
initial diagnostics are saved in root `launch_verified.json`; source commit
is `02eef73`. Both jobs will automatically evaluate dev and in-domain test.

After objective/decoder freeze and training launch, CPU reference replay
completed for 7 models with identical 2,000/5,000 score dev/test IDs. Every
original argmax MAE was reproduced; original files were not changed. Test MAE:

| Model | Argmax | Median (new primary) | Mean (continuous) | Nearest-native mean |
|---|---:|---:|---:|---:|
| A4 | 0.4296 | 0.4148 | 0.4801932 | 0.4246 |
| Kev | 0.4396 | 0.4256 | 0.4808593 | 0.4262 |
| Ordinary LoRA | 0.7836 | 0.5196 | 0.5777439 | 0.5286 |
| Full v6 | 0.4320 | 0.4214 | 0.4813024 | 0.4238 |
| L4 ordinal auxiliary | 0.4350 | 0.4194 | 0.4777161 | 0.4238 |
| M1 regression auxiliary | 0.4340 | 0.4228 | 0.4794334 | 0.4284 |
| M2 conditional choice | 0.4320 | 0.4210 | 0.4797832 | 0.4310 |

Source: `reference_score_rules/dev.json` and `test_in_domain.json` under the
run root, including input hashes, exact counts and source breakdowns. These
decoder changes improve some point metrics without changing weights. A4
remains the lowest median MAE among these references; no improvement from the
new training can yet be claimed. Neither training config was changed after
this alternative test replay.
