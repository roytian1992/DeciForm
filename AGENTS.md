# DeciForm ownership

DeciForm owns the decision model architecture, supervised training, optimizer
and LoRA implementations, inference, calibration, data preparation and model
benchmark comparisons. Implement and test model changes here.

IUC-RSI consumes this package and explicit checkpoint manifests; its compatibility
repository path is `JEV_RSI/`. Keep recursive controllers, self-edit decisions,
rewards, episode orchestration and reset-transfer experiments there. Do not import
jev_rsi from the active deciform package.

Preserve released weights and historical experiment artifacts. New training
runs use new output directories and record their parent checkpoint and hashes.
Historical source files under legacy/ are provenance, not supported launchers.

# Research evaluation protocol (user requirement, 2026-10-31)

Candidate sampling is a training-compute control only. Every development,
calibration, validation and test example must retain its original complete
candidate inventory. Do not run sampled-K evaluation views, including auxiliary
diagnostics. Candidate permutation audits retain every candidate and remap stable
semantic IDs. Report unsupported or failed examples; never silently truncate text,
drop candidates, or omit failures to make a benchmark fit.

The active research plan is `../papers/JEV_QF/notes/research_plan.md` (formal reset,
2026-11-01; working method name TCDL). All previous experiments are internal POC,
not evidence for the formal paper or instructions to resume old runs. Formal
comparisons include ordinary LoRA, Kev, JSON-generative LoRA and TCDL, plus
new-protocol ablations. Preserve the old 100k/10k files; the old validation is an
internal regression set, not a new untouched test. Freeze any expanded formal
dataset separately. Architecture comparisons require matched supervision,
negative IDs and initialization rules; distinguish equal data exposure from
equal compute and report actual cost. Separate hypotheses from measured findings.

The user clarified that the paper's main contribution is the general-purpose
decision model: its architecture, task-specific objectives, quality and measured
efficiency across choice/noul/score. Rubric/criterion interaction is optional
supporting analysis, not a replacement research topic or a prerequisite for
the main experiments. Its proposed new loss/module is not in the active method.

We study and improve the existing decision-model paradigm; do not claim to have
introduced it. Attribute prior Jev/Kev work accurately. Formal baselines also
include a generative LLM that outputs task JSON followed by parsing: same-base,
same-data JSON-generative LoRA is a matched training comparison; prompted instruction
LLMs are separate practical references. Preserve native candidate/score support,
include parse failures in label-metric denominators, and do not invent comparable
probability distributions from a single generated answer or self-rated confidence.

The user confirmed approximately 200k training requests for the paper's main
experiments, with smaller nested learning-curve subsets. Do not expand this study
to 500k or full-corpus training by default. Broader release-model training and a
global unique-data audit are separate future work, not blockers for this study.

The user explicitly rejected multiple training seeds and the extra LoRA-All arm.
Use one fixed training seed and one nested-subset sequence; do not automatically
add seed repeats or restore LoRA-All. Learning curves include only Kev and TCDL;
ordinary LoRA and JSON-generative LoRA participate only at 200k. The remaining formal
matrix has 16 training configurations/runs, starting with four main models.
The user cancelled C1 (existing multi-task adapter) and C2 (matched MLP)
replacement controls; do not restore them as requirements or deferred jobs.
Retain four core ablations and limit component claims to measured comparisons.
Report parameter coverage;
evaluation bootstrap intervals are conditional on the single trained model and
must not be represented as evidence of cross-seed stability.

The user subsequently changed the formal main budget to one epoch (2026-11-01).
Active run: `runs/formal_main_20261101_v2_one_epoch/experiment_manifest.json`.
All four main models use 200k requests, batch 64, 3125 updates and a complete
OneCycle schedule with 30% warmup. Each resumes its own current fresh-base step-200
checkpoint; the completed LR/momentum prefix is identical and audited. This is
not inheritance from a POC model. Do not automatically add epochs 2 or 3. Default
curves/ablations also use one pass; the two explicitly planned 50k repeated-update
controls are separate. Dev and both test panels always keep native full candidates.

Naming: call the generative baseline “JSON 生成式 LoRA” (Generative LoRA, JSON
output). SFT is shared by all four arms and is not a distinguishing model name.
Keep the internal `json_sft` ID/config/checkpoint hashes stable during active runs.

After each main model completes full-native evaluation, the four TCDL component ablations may be queued automatically on the released matching GPU pair. Use `runs/formal_ablations_20261102_one_epoch/` and preserve fresh-base initialization.

Latest user override (2026-11-02): stop the four ablation trainings; do not resume
them automatically while improving the method. Kev 25k/50k/100k controls and
TCDL-PAC have now completed. The authorized active optimization is
runs/formal_geometry_20261102_v1/experiment_manifest.json, fresh-base 200k,
one epoch on GPUs 4--7, with automatic full-native dev/test evaluation.
See docs/experiment_log_20261102_geometry.md for dev-only diagnostic evidence,
method changes and the limits on causal/novelty claims.

Latest authorized continuation (2026-11-02): v5 completed. Current run is
runs/formal_ordinal_20261102_v1/experiment_manifest.json on GPUs 4--7.
It retains the v5 data/geometry/losses, adds the ordinal CDF-gap score readout,
and uses 16/32 checkpointed decoder layers after measured preflight selection.
See docs/experiment_log_20261102_ordinal.md. Content-dependent routing was
not implemented in this run; old ablation trainings remain cancelled.

Latest evaluation override (2026-11-02): defer all external/source-held-out
dataset construction, evaluation, and routine result reporting until the other
experiments are complete. Then prepare a new, carefully audited external panel
covering choice, noul and score, freeze it, and evaluate matched models together.
The existing source-held-out panel omits choice and must not represent overall
three-task generalization. Preserve its old files for provenance, but do not
queue additional external evaluations or include its results in progress reports.
Current v6 continues from its own optimizer/RNG checkpoint under
runs/formal_ordinal_20261102_v1/experiment_manifest.internal_only.json;
automatic evaluation now includes only dev and test_in_domain, always with
complete native candidates. Model, training data, losses and schedule are unchanged.

The v6 model is now frozen as the formal proposed version after its matched
200k source-internal evaluation. For score tasks, native numeric MAE is the
primary selection and headline metric; exact-label accuracy, NLL, Brier and RPS
remain secondary diagnostics and cannot overturn a lower-MAE conclusion. The
next evidence order is component ablations, then nested Kev/TCDL data-size
curves. These use the frozen v6 reference, data views and source-internal test.

Latest execution (2026-11-02): the user authorized two v6 ablations now on GPUs
4--7. Active root: runs/formal_v6_ablations_20261102_v1. A1 on 4--5 fixes only
LoRA rank-space task matrices to zero (shared A/B); head task geometry, QF,
ordinal readout and losses stay enabled. A2 on 6--7 disables only the choice
QF contribution and freezes its dedicated parameters. Both start fresh from
the base, 200k/one epoch/batch 64/3125 steps, with the frozen half-layer
checkpointing policy and dev/in-domain automatic evaluation. Do not resume
legacy v2 ablation jobs. The previous four-item list incorrectly retained a
"set term": frozen v6 has no independent set-summary branch. Only A1/A2 are
implemented and queued; revise remaining component controls against v6 before
launching more. The full reference used four GPUs, these arms use two each;
record rank/RNG/numerical differences and do not claim a wall-time speedup from
this hardware comparison. See docs/experiment_log_20261102_v6_ablations.md.

Latest continuation (2026-11-03): the user authorized the two remaining valid
v6 controls in runs/formal_v6_ablations_20261103_v1. A3 on GPUs 4--5 disables
the task-conditioned bilinear head geometry; A4 on GPUs 6--7 removes only the
score CRPS auxiliary term (TCDL-CE). Both passed fresh-base and fixed-half
checkpoint preflight and are training one epoch. The stale set-term control is
not revived because no independent set branch exists in v6.

Completion (2026-11-03): A3/A4 training and internal evaluation both finished
successfully. Results: runs/formal_v6_ablations_20261103_v1/results_summary.json.
A4 is slightly better on all three in-domain headline metrics, so this result
does not establish a benefit from the score CRPS mixture. V6 remains frozen;
no subsequent training or external evaluation was automatically queued.

Latest authorized execution (2026-11-03): the user approved the choice-loss
study in docs/loss_research_20261103.md. Active root is
runs/formal_choice_losses_20261103_v1: L1 CE/pairwise likelihood (lambda=0.5)
on GPUs 4--5 and L2 uniform-gradient-normalized focal (gamma=1) on GPUs 6--7.
Both retain all v6 architecture components and use A4's unchanged noul
Bernoulli NLL, score CE and structural penalty. They initialize fresh from
the base (no A4 checkpoint inheritance), with the same 200k/one epoch/fixed
negative IDs/global batch 64 and two-GPU partition as A4. Both distributed
zero-update preflights passed. Supervisors in tmux automatically train and
evaluate full-native dev/in-domain test; external evaluation remains deferred.
Score decoding remains original argmax, with MAE primary. No median-decoding
change or noul-loss experiment is included. See each arm's status.json and
progress.json and docs/experiment_log_20261103_choice_losses.md. The frozen
v6 and completed A4 results are preserved; loss-gradient diagnostics are only
output-logit proxies, not backbone gradient-conflict evidence.

Latest authorized expansion (2026-11-03): the user approved stopping Qwen 8001
and using GPUs 0--3 for the remaining two task-loss controls. Qwen's identified
process tree and llm_keepalive_8001 were stopped; BGE 8081 remains available.
New root: runs/formal_typed_losses_20261103_v1. L3 on GPUs 0--1 changes only
noul to 0.5 BCE + 0.5 binary Brier (2*(p-y)^2); L4 on GPUs 2--3 changes only
score to CE + 0.1 normalized-support threshold log loss. Both retain A4's full
architecture, identical data/negative IDs, fresh-base initialization and
200k/one epoch/batch 64/3125-step schedule. Both zero-update GPU preflights
passed and supervisors transitioned to fresh training. L1/L2 on 4--7 remain
unchanged. Primary inference/evaluation still uses full native candidates and
original argmax; score MAE remains primary. L3/L4 additionally save dev-only
median-decoding diagnostics, not replacement test metrics. The same CPU replay
for A4/full v6/Kev/ordinary LoRA is in reference_dev_decoding.json under the new
root. No external-source evaluation or dataset change. See
docs/experiment_log_20261103_typed_losses.md and both active roots' status.json.

Latest research scope (2026-11-03): all four task-loss controls have completed.
The user now fixes noul to Bernoulli BCE and requests focus on choice and score,
including a regression-plus-classification score formulation. Do not queue
additional noul-loss treatments. Current analysis/proposals are recorded in
docs/loss_postmortem_20261103.md; its latest user-steering section supersedes the
earlier preferred pure-ordinal proposal. These proposals are not launched runs
or an adopted change to the frozen evaluation/score decoding protocol.

Latest authorized launch (2026-11-03 23:15 Asia/Shanghai): the user approved
the two choice/score proposals with noul fixed to BCE. Both are now training
under runs/formal_decision_mechanisms_20261103_v1. M1 on GPUs 0--3 adds a
separate numeric score head with CE + 0.5 native MAE supervision. M2 on GPUs
4--7 changes choice QF to positive normalized query-conditioned precisions,
with CE unchanged. Both use fresh base initialization, identical 200k fixed
views/negative IDs, global batch 64 and one epoch (3125 steps), and automatic
full-native dev/in-domain test. Four GPUs per arm differs from A4's two; record
the rank/RNG/reduction caveat. Both zero-update preflights passed and both were
verified at step 20, with exact matching initial categorical losses and no
inherited trained weights. Primary argmax decoding remains unchanged; numeric
regression/median decoding are separate dev-only diagnostics. See
docs/experiment_log_20261103_decision_mechanisms.md and launch_verified.json.
Do not edit frozen live-run source/config files or queue further noul losses.

Completion (2026-11-04): both M1/M2 decision-mechanism runs finished training and
full-native internal evaluation successfully. Results are in
runs/formal_decision_mechanisms_20261103_v1/results_summary.json. Neither exceeds
A4 on the original-argmax test metrics; M1 score MAE is 0.4340 and M2 0.4320 vs
A4 0.4296. Regression and median outputs remain separate dev-only diagnostics.
Eight training GPUs were released. No further run or test decoder change was
automatically queued; noul remains fixed to BCE for future authorized work.

Latest authorization (2026-11-04): the user requested new training and explicitly
suggested distribution/expectation score evaluation, allowing a justified
alternative. The next root is runs/formal_score_distribution_20261104_v1:
D1 CE plus squared-error supervision of the predictive mean, and D2 a pure
ordered-CDF log objective. Both retain A4 architecture, choice QF/CE and noul
BCE, fresh base and the identical 200k/one-epoch fixed views. Protocol
score_distribution_20261104_v1 explicitly uses native predictive median MAE
as score headline and reports argmax/median/mean/nearest-native-mean for all
comparable probability baselines. This supersedes the former argmax-only
restriction for these new versioned reports; preserve all original reports.
Continuous means are diagnostics, not new allowed native labels. The protocol
and objective coefficients are frozen before alternative test replay. See
docs/experiment_log_20261104_score_distribution.md. External evaluation stays
deferred, and no additional noul-loss experiments are authorized.

Launch verified (2026-11-04 05:36 Asia/Shanghai): both distribution objectives
are training under runs/formal_score_distribution_20261104_v1. D1 on 0--3,
D2 on 4--7; both passed zero-update preflight and reached step 100/3125 with
matching full initial heads, no inherited trained weights, and no failures.
The fixed ordinal scale is 3.0383850790577127. Seven old probability models
were replayed under the same four-rule protocol after freeze; A4's median
test MAE is 0.4148, Kev's 0.4256. Use reference_score_rules/test_in_domain.json
for matched comparisons to these new median-primary runs, not legacy argmax
MAE. See launch_verified.json and docs/experiment_log_20261104_score_distribution.md.
Keep the running source/config hashes immutable; no new run is queued beyond
these two, and their supervisors automatically perform full-native internal eval.

Completion (2026-11-04 10:07 Asia/Shanghai): D1/D2 score-distribution runs both
finished training and full-native dev/in-domain test with exit code 0, at
09:45/09:46. See runs/formal_score_distribution_20261104_v1/results_summary.json.
Test choice/noul/median-score-MAE: D1 0.7218/0.8898/0.4182; D2
0.7252/0.8884/0.4180; matched A4 0.7262/0.8916/0.4148. D2 improves the
diagnostic argmax score MAE to 0.4278 versus A4 0.4296, but does not beat A4
under the frozen median-primary protocol. Preserve all four decoder reports
and do not switch the primary rule based on these test results. No further
run was queued; training GPU allocations are released, with existing services
and idle watchdog retained. External evaluation remains deferred.

Latest method designation (2026-11-04): the user adopts completed D2
`runs/formal_score_distribution_20261104_v1/d2_ordinal_main` as the current
full TCDL method. It includes task-conditioned rank routing, choice QF,
task-conditioned head geometry, ordinal score readout, choice categorical CE,
noul Bernoulli BCE and score pure ordered-CDF threshold loss. Median native
score MAE remains the frozen headline rule; report other fixed rules alongside.
Old v6 A1/A2/A3 ablations cannot establish D2 component contributions because
the score objective and decoder changed. New D2 A1/A2/A3 are documented in
docs/experiment_log_20261104_d2_ablations.md. D2 itself is not retrained.
Fresh-base four-GPU A1/A2/A3 use the exact 200k D2 data/config/protocol, each
disabling only one architecture component. Historical v6/A4 runs and paper
draft remain provenance; update draft claims only after matched D2 results.

Launch verified (2026-11-04 11:35 Asia/Shanghai): new D2 A1/A2 are training
in tmux queues `deciform_d2_ablate_g03_20261104` and
`deciform_d2_ablate_g47_20261104`. Both passed zero-update full-native GPU
preflight, matched D2 full's initial base/LoRA/head/pointer hashes, and were
observed at 60/3125 and 50/3125 steps with no failures. A3 is queued after
A1 on GPUs 0--3 and launches automatically only upon A1 success. Current
authoritative progress is under runs/formal_d2_ablations_20261104_v1;
source/config hashes are frozen. Do not edit live-run source files or infer
new results before complete internal evaluation.

Answer-order audit authorization (2026-11-04): user requests ordinary LoRA,
Kev, JSON-generative LoRA and current D2 full score/noul option-order tests
using two spare GPUs. All eight GPUs currently run A1/A2, so the new audit
waits for A2 completion and memory release on GPUs 6/7. It does not resize or
interrupt either ablation. The frozen in-domain test's 5k score/5k noul rows
form the native interface panel; a deterministic source-stratified 20% subset
forms the 2k common instruction-list panel. Preserve all candidates/meanings;
ordinary scalar native interfaces and generative noul have no order input and
must be reported N/A rather than zero-flip wins. Both pointer models gather
readouts by numeric/boolean semantics after genuinely permuted causal input.
No guarantee of end-to-end invariance or D2 superiority is assumed. See
docs/experiment_log_20261104_answer_order.md and
runs/formal_answer_order_20261104_v1 for the frozen design/queue. External
evaluation remains deferred; live D2 ablation sources stay immutable.

Order-audit queue verified (2026-11-04 12:49 Asia/Shanghai): tmux
`deciform_answer_order_20261104` is waiting for A2 success and GPU 6/7 release.
The full frozen scope is 233,000 unique model/panel/row/order predictions:
ordinary LoRA 22k, Kev 72k, generative LoRA 67k, full D2 72k. The actual
example populations remain native 10k and instruction-list 2k. Code commit
400a538; 58 tests, 297 real-tokenizer cache-parity checks and 3,300 order-view
checks passed on CPU. No GPU robustness result exists at queue launch.
Consult runs/formal_answer_order_20261104_v1/status.json and per-model
progress/report files. Do not count N/A native interfaces as invariant wins.

Latest Score research scope (2026-11-04): user keeps D2 as the current full
and requests a separate supplementary Score study, without retuning D2 to win
the existing five-level test. See docs/score_specialist_plan_20261104.md.
Current work is planning/raw intake; no new GPU training is launched by this
request. The proposed default budget is four fresh-base runs: same-D2-architecture
CE versus unchanged D2 ordinal loss on native ASAP scales, then matched soft-target
CE versus CDF objectives on STS-B/WMT continuous labels. The latter is explicitly
a separate continuous-label extension, not a change to the current D2 checkpoint
or main benchmark. Source test support remains complete; never round raw
fractional labels or sample evaluation candidates. New-source intake is scoped
to this Score-only study; the general three-task external panel stays deferred.
A3/order queues retain their existing ownership. The matrix is a proposal, not
an active queue, and does not revive cancelled baselines or seed repeats.

Latest order-audit correction (2026-11-04): user explicitly limits flip-rate
experiments to Choice and Noul; Score permutations/flip-rate reporting are out
of current scope. The earlier completed audit contained Score/Noul and no
Choice. Keep its raw files as provenance, but use its new classification_report.json
for current reports. Reuse Noul; do not report historical Score flip rates as
method advantages. Native Choice completion now runs under
runs/formal_choice_order_20261104_v1 in tmux deciform_choice_order_20261104:
GPU 6 ordinary then generative LoRA, GPU 7 D2 then Kev. All 5,000 frozen Choice
rows retain complete native K (including 21/77), with original/reverse/three
fixed unique shuffles; 25k predictions per model. IDs stay attached to candidate
meaning. Source commit 483f5fb, 33 tests and 45 real-tokenizer cache checks plus
225 order views passed. At launch verification the first two workers were loading
models, with no GPU metrics yet. See docs/experiment_log_20261104_classification_order.md.
Score-only dataset/loss planning remains a separate task and D2 stays frozen.

Latest Score execution authorization (2026-11-04): the user explicitly requests
advancing docs/score_specialist_plan_20261104.md on GPUs 4--5, with further arms
on released GPUs. This supersedes the earlier planning-only status. Active root:
runs/formal_score_specialist_20261104_v1, tmux deciform_score_specialist_20261104.
S3 soft CE starts on 4--5; S4 soft CDF uses the next successful released pair:
0--1 after A3, 6--7 after the Choice audit, or 4--5 after S3. Both use two GPUs,
fresh original base, identical continuous-label data and full 51/101-point grids,
11,457 train / 3,500 dev / 3,379 test, one epoch / 180 updates / batch 64.
Both retain original real-valued labels and the same adjacent-grid interpolation;
the headline is median native MAE per source and equal-source NMAE across ranges.
No score permutations are added. D2 full and main benchmark remain frozen.
ASAP S1/S2 are pending actual essays/rubrics, not queued with missing data.
See docs/experiment_log_20261104_score_specialist.md. Keep all live-run code and
configuration hashes immutable; the scheduler preserves A3 and Choice ownership.

ASAP continuation authorized and launched (2026-11-04 21:57): user requests S1/S2
on free GPUs 4--7. GitHub fold-0 IDs and HF mirror raw essays/descriptions are now
downloaded and audited. Root runs/formal_score_asap_20261104_v1, tmux
deciform_asap_score_20261104: S1 CE on 4--5, S2 original D2 ordinal loss on 6--7.
Fresh base only, identical architecture and data, 7,784 train / 2,593 dev / 2,598
test, one epoch / 122 updates / batch 64. Preserve all native 4--61 score levels
and full essay/prompt/source/rubric input. Both share a 4096 microbatch packing
budget: longer single requests remain complete, never truncated. Median native
MAE and equal-prompt NMAE are headline metrics. Original author decoding defects
are recorded explicitly in the data manifest. Prior S3/S4 are complete and D2
full remains frozen. See docs/experiment_log_20261104_asap_score.md; do not edit
live frozen code/config/data or restart completed/active arms silently.

Latest independent Score follow-up (2026-11-05): after all new-topology A1--A3
finished training, full dev/test and classification reversal audit (authoritative
`evaluation_recovery_20261105_v2` under each arm), user authorized filling the
three missing Score arms. Root `runs/formal_independent_score_completion_20261105_v1`:
ASAP CE on 2--3; continuous soft-CDF on 4--5; continuous soft-CE on 6--7.
The existing independent ASAP D2 continues on 0--1 under
`runs/formal_independent_asap_score_20261105_v5/independent_d2_asap`.
Each new arm uses an independent tmux supervisor with zero-update preflight,
fresh-base one-epoch training and automatic native dev/test. Keep both paired
arms' data, chunk-8 execution, initialization, full support and schedule matched;
only the Score objective differs. Read the score completion log and live statuses.
Do not change these frozen source/config hashes or confuse the original ablation
evaluation failures with the successful recovered evaluation. New A1--A3 results
are in `docs/experiment_record_final_d2_20261105.md`; they do not establish that
all three modules improve the new topology. Main full remains frozen.

Latest authorized execution (2026-11-05): user approved candidate-independent
D2 full on GPUs 4–7. Active root `runs/formal_independent_20261105_v1`, tmux
`deciform_independent_20261105`. Shared differentiable state/question prefix,
isolated query and candidate branches retain all 46,304,262 trainable parameters,
all-target LoRA/task routing/QF/head geometry/ordinal readout and frozen D2 losses.
Fresh original base, exact frozen 200k/one epoch/3125 updates/negative IDs; no
trained checkpoint inheritance. Actual 9B zero-update preflight passed, initial
hashes match D2 fresh initialization, and real training updates were verified.
Automatic complete-native dev/test plus Choice/Noul reversal audit follow the
endpoint; Score permutations and external evaluation remain excluded. Read
`docs/experiment_log_20261105_independent_candidates.md` and live progress.
Existing GPUs 0–3 D2 size curves are protected; do not modify either run’s frozen
source/configs. Candidate independence is an exact-arithmetic architecture
property; measure numerical deviations and do not claim quality gains yet.

Latest Score baseline scope (2026-11-05): user explicitly cancelled ordinary
LoRA for the supplementary Score panels; only Kev and JSON-generative LoRA
are authorized. Active root `runs/formal_score_baselines_20261105_v1`, tmux
`deciform_score_baselines_20261105`. ASAP Kev on 0–1 and JSON LoRA on 2–3
passed fresh-base preflight and reached real training updates; two continuous
Score arms automatically take the next released GPU pairs. Same frozen
ASAP/continuous data as S1–S4, one epoch, batch 64, full allowed score ranges,
no truncation/rounding/sampling or trained checkpoint inheritance. Kev keeps
original pointer plus CE (common interpolation for fractional labels); JSON
uses token CE and strictly parsed original numeric targets, with invalid outputs
counted and no fabricated probabilities. WMT en-zh is the reporting focus;
retain complete German artifacts and unchanged mixed-source training. Read
`docs/experiment_log_20261105_score_baselines.md`. All source/config hashes
are frozen. Main independent-D2 on 4–7 continues separately unchanged.

Latest zero-shot reference authorization (2026-11-05, explicit instruct-only choice): user requests an
unadapted LLM JSON reference for the main 200k study's held-out test, the two
Score panels, and classification position audits. User explicitly selected ONLY
official Qwen3.5-9B instruct. Active queue:
`runs/formal_zero_shot_instruct_20261105_v3`, model `instruct`, original official
weights at `../models/Qwen3.5-9B`, no task-data adaptation/LoRA/custom head or
optimizer updates. Use official chat template, enable_thinking=False, greedy
64-token JSON outputs. This reference has upstream post-training; do not call
it pretraining-only or an exact same-Base-weight ablation. The previous Base queue
`runs/formal_zero_shot_json_20261105_v1` was cancelled before any inference worker
or prediction existed. Preserve cancellation evidence and do not resume it.
Persistent tmux `deciform_zero_shot_instruct_20261105` waits for successful completion
and actual release of the current four Score GPU pairs. Four fixed one-GPU
shards evaluate 15k main + 2598 ASAP + 3379 continuous + 50k Choice-order +
10k Noul-prompt-order predictions (80,977/model). Choice reports separate matched
seven-order and legacy five-order subsets; Noul explicitly appends the frozen
boolean list. No Score permutations or external three-task panel. Keep invalid
JSON in denominators, report validity and both-valid flip rates, and do not
fabricate probabilities. Read the zero-shot log, frozen manifest and status.

Latest research-scope override (2026-11-05 20:11 Asia/Shanghai): the user freezes
the independent-candidate D2 main model and its completed A1--A3 evidence.
Future high-level-count Score optimization is a separate research question,
RQ-Score, documented at the top of docs/score_specialist_plan_20261104.md.
Do not launch a new A4, retrain the three-task full, or repeat its ablation suite
because of a Score-only proposal or improvement. Reuse protocol-matched existing
Score baselines and CE/CDF specialist controls; report specialist findings apart
from the frozen main model. Future targeted comparisons belong to this Score
study and do not automatically authorize a new full model or experiment matrix.
Keep native complete evaluation supports and existing data/decoder protocols.
This scope decision does not cancel or modify already-running jobs. The present
update changes planning documents only; proposed numeric-distribution modules
remain unimplemented. See docs/experiment_record_final_d2_20261105.md and
../papers/JEV_QF/notes/research_plan.md for the synchronized scope boundary.

Latest authorized allocation (2026-11-05 22:20 Asia/Shanghai): user requests four
GPUs for frozen final independent-D2 size curves and four for D2-based Score
research, and cancels future English--German work. New curves root:
`runs/formal_independent_curves_20261105_v1`. GPUs 0--1 run 25k then 50k;
2--3 run 100k after the already-trained ASAP CE finishes its current complete
evaluation. Exact Kev nested indices/negative IDs, fresh base, one epoch, batch
64; reuse the existing independent-D2 200k endpoint. Do not retrain full/ablations.
The inefficient old continuous_cdf/continuous_ce runs under
`runs/formal_independent_score_completion_20261105_v1` were explicitly cancelled;
preserve artifacts and do not resume them. New Score root:
`runs/formal_score_numeric_20261105_v1`, data
`data/score_specialist_stsb_enzh_20261105_v1` (8129 train/2500 dev/2379 test),
retaining exact STS-B/en-zh rows and excluding en-de from all new splits. GPUs
4--5 run a batched independent D2 reference, 6--7 a single-text-encoding numeric
CDF readout. Both use the same soft-CDF loss, full grids, original real labels,
median decoding, fresh base, one epoch/batch64/128 updates. Numeric Score is now
implemented as separate RQ-Score, not adopted into the frozen main model. Old
mixed-German training baselines are historical references, not matched controls
for the filtered training population. Code/configs are frozen after CUDA
throughput/gradient checks; supervisors automatically preflight/train/evaluate.
See `docs/experiment_log_20261105_curves_score_numeric.md`. No new German
training/evaluation jobs, no truncation, no sampled-support evaluation.

Continuation (2026-11-05 22:45): numeric_cdf completed the full filtered
STS-B/en-zh pair arm, 128 steps, 209.91 s train + 39.87 s eval; the matched
D2 reference is still running. GPUs 6--7 continue the unchanged numeric
module on native ASAP, root `runs/formal_score_numeric_asap_20261105_v1`,
7784/2593/2598 train/dev/test, fresh base, 122 updates, full native integer
levels, identical numeric module/loss. Half checkpoint, padded-token budget
8192 and at most four rows after complete-long-essay CUDA checks. Initial
four-longest-essay OOM probe preserved; final bounded batching passed. This
remains separate RQ-Score, not main-D2 retraining or an A4. Use the ASAP
metrics adapter and do not report continuous [51,101] metadata for this run.

Latest user-authorized Score study (2026-11-06): run two native ASAP loss
experiments in parallel on the unchanged numeric CDF architecture. Root
`runs/formal_score_objectives_20261106_v1`: R1 `r1_cdf_mean` on GPUs 4--5
adds squared normalized predictive-mean error (weight 1) to original CDF-log;
R2 `r2_crps` on 6--7 uses pure normalized CRPS scaled by 2*3.0383850790577127.
Same 7784/2593/2598 train/dev/test, full native supports, 1 epoch/122 updates,
batch64/seed20261101, fresh base and exact numeric-head initialization, original
8192 token budget/max4 rows/half checkpointing/optimizer/median decoder. No
checkpoint inheritance, hard truncation, German jobs or main-D2/A1--A3 reruns.
Both passed distributed preflight and supervisors have entered training.
Use isolated `worker_numeric_objectives.py` and `supervise_numeric_objectives.py`;
do not edit active frozen sources/configs. Auto-evaluate complete dev/test and
include JSON-generative LoRA (ASAP NMAE 0.087341) in final comparisons, along
with numeric CDF/D2/Kev; do not claim a win solely against D2. This is a loss
study, not a new model architecture claim. Mathematical/verification record:
`docs/experiment_log_20261106_score_objectives.md`. GPUs 0--3 continue the
existing final-D2 size curves unchanged.

Latest comparison preference (2026-11-06 01:12 Asia/Shanghai): user explicitly
accepts the historical English--German training-mixture difference in the
continuous Score references. Compare the existing STS-B/en-zh per-source
results directly in routine reports; do not repeatedly foreground this
difference, block comparison, or retrain baselines solely to remove it.
Preserve truthful source counts/manifests and avoid claiming identical
training populations or an isolated causal architecture effect. This does not
restore new German training/evaluation jobs or change active ASAP runs.

Latest user-authorized replay (2026-11-06 02:41 Asia/Shanghai): user doubts the
generative Score baseline and requests a fresh run plus R1/R2 results on all
three Score datasets. The old local cache/prompt/target-independence, split
overlap and raw-JSON rescoring audit passed, root
`runs/score_generative_integrity_20261106_v1`. New execution root
`runs/formal_score_recheck_20261106_v1`: 0--1 fresh exact ASAP JSON-LoRA replay;
4--5 continuous R1 CDF+mean then continuous JSON-LoRA replay; 6--7 continuous
R2 CRPS. All fresh Base, one epoch, automatic complete dev/test. Continuous
population is the current 8129/2500/2379 STS-B/en-zh split; historical 11457-row
generative training is retained and not called an identical-population replay.
ASAP remains the exact original 7784/2593/2598, seed/config/initialization.
Generation wrapper removes labels and answer tokens before predict. Numeric
objectives and initialization exactly match the completed ASAP objectives;
full support, fractional labels, median decoder and no truncation persist.
R1/R2 ASAP are complete: MAE .911855/.931486, macro NMAE .100677/.102575;
other datasets' R1/R2 results were previously missing, not completed.
Preserve original results pending verification, do not alter frozen run code,
and keep the main D2/ablations unchanged. 2--3 continues 100k curve. Read
`docs/experiment_log_20261106_score_recheck.md` and live status/launch proofs.

02:48 completion: both continuous objective runs finished successfully; R1/R2
STS-B MAE .5243705584/.5245025381 and WMT en-zh 10.3249999676/11.2306666241;
equal-source NMAE .1040620557/.1086035869. All three datasets now have R1/R2
results; see new root `objective_results_summary.json`. Continuous JSON replay
has automatically followed R1 on 4--5. ASAP JSON remains on 0--1. Both fresh
generative outputs still pending; preserve old evidence and frozen main D2.

Latest execution (2026-11-06 03:20 Asia/Shanghai): user requires the unified
record `docs/experiment_record_final_d2_20261105.md` to show only current results,
overwriting superseded cells instead of accumulating historical snapshots.
`scripts/formal/update_final_record.py --watch` maintains current curve/Score
tables in a separate tmux. Old reports/checkpoints remain in runs for provenance.
Ordinary LoRA/Kev main Score columns now use the already frozen median protocol
(.5196/.4256); the previous document incorrectly retained argmax values.
All final independent-D2 curves completed; 100k test Choice/Noul/Score-MAE are
.7240/.8770/.4400. Continuous generative replay completed with STS-B .4504865845,
WMT en-zh 15.555333334, all 2379 outputs valid. ASAP generative replay continues
on 0--1; updater will replace its old result only after complete evaluation.

User authorized bold ASAP research on spare GPUs, including newly released
2--3. New root `runs/formal_asap_exploration_20261106_v1`: E1 evidence_cdf on
4--5 uses full-prefix query-conditioned evidence pooling with original CDF loss;
E2 median_laplace on 6--7 uses a latent median/scale Laplace ordinal readout with
CDF-log + normalized median absolute error (weight1); E3 combines both on 2--3.
All use the same ASAP-only 7784/2593/2598 rows, fresh Base, no parent checkpoint,
one epoch/122 steps/batch64, seed20261101, original complete text/native levels,
median decoder, half checkpointing and 8192-token packing. No STS-B/WMT mixture,
new data, extra epochs or main-D2/A1--A3 changes. 27 CPU tests and each exact
model's two-GPU longest/dense/full-batch preflight passed; actual step1 verified.
Frozen source/configs must remain unchanged. Each tmux supervisor auto-evaluates
complete dev/test, records failures/stalls, and can resume only its own state.
See `docs/experiment_log_20261106_asap_exploration.md` and root launch proof.

Completion (2026-11-06 10:05 Asia/Shanghai): E1/E2/E3 ASAP arms and the fresh
ASAP JSON-LoRA replay completed with exit code 0 and full native dev/test. E1
MAE/NMAE .980370/.102086, E2 .910316/.099332, E3 .890685/.097932; fresh JSON
replay .803695/.086374 with 2598/2598 valid outputs. E3 is better than the
numeric-CDF/R1/R2 controls but remains below JSON generation; do not claim
superiority over all baselines from this single seed. The D2 100k curve also
completed: .7240/.8770/.4400 Choice/Noul/Score-MAE. GPUs are released; no active
training remains. The unified record was overwritten from completed reports by
`scripts/formal/update_final_record.py`.

Latest authorized baseline (2026-11-06): user requests Laya on the same 200k
training records and full-native quality/position evaluation on GPUs 4--7.
Active root `runs/formal_laya_20261106_v1`: sequential on 4--5, upstream parallel
layout on 6--7. Both use pinned Laya 0.4.26, fresh original ModernBERT-large
encoder (revision 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13), random decision
head, no released Laya/D2/Kev checkpoint inheritance. Same frozen training
views/negative IDs, one epoch/global batch64/3125 updates; native upstream RLCD,
encoder/head learning rates and cosine schedule. This is a cross-backbone
same-supervision comparison, not a same-backbone module ablation. Preserve
complete text and supports: 4/1/2 train/dev/test examples exceed the original
8192-token pretraining length (maximum 9662), are processed fully with original
RoPE frequencies, and passed actual longest-input GPU forward/backward checks.
Both initial encoder/head hashes match. CPU protocol tests and both zero-update
GPU preflights passed; real step120 and step100 resumable checkpoints verified.
Tmux supervisors auto-run dev6000/test15000 and the final-D2 seven-order semantic
audit on Choice/Noul10000; Score is excluded from flip rates and uses median MAE
with all four decoder reports. Source/config manifests are frozen; do not edit
active worker/vendor code. `scripts/formal/report_laya.py --watch` updates the
separate LAYA_CURRENT block of the unified record. See
`docs/experiment_log_20261106_laya.md` for exact paths/commands/recovery and
prospective D2-to-ModernBERT / Laya-to-Qwen transfer reasoning. Those transfer
models have not been implemented or launched; frozen main D2 and RQ-Score stay
separate.

Latest Score RQ execution (2026-11-06 12:44 Asia/Shanghai): first numeric-trie
N1/N2 study completed all three native Score tests under
runs/formal_numeric_trie_20261106_v1. N1 MAE ASAP/STSB/WMT is
.8233256351/.4343966642/11.3653333187; N2 is
.8298691301/.4337440174/11.1299999905. Fresh generative replay is
.8036951501/.4504865845/15.5553333340. Both trie methods improve the continuous
sources but fail ASAP native MAE, despite better ASAP macro NMAE. Do not call
this a three-source win or cross-seed stability.
Current follow-up is runs/formal_numeric_trie_consistency_20261106_v4:
ASAP T1 (CE + .5 exact-decimal token NLL) on 0--1; ASAP T2
(.5 CE + .5 CDF-log + .5 token NLL) on 2--3. Both passed actual two-GPU
zero-update preflight including longest backward, dense/first global batches,
full-native inference and gold/counterfactual suffix independence (observed
probability difference 0). Peak 54.404 GiB. CPU tests: 23 passed. Distribution
and token supervision share ONE causal backbone forward; the distribution
reads before the first answer digit, and evaluation reads prompt only.
Same seed, one epoch, batch64, full text/support, fresh Base and no inherited
head/adapter. No main D2 or A1--A3 changes. Each arm auto-evaluates dev/test.
Continuous T1/T2 are QUEUED on 4--5/6--7 AFTER Laya's complete training AND
quality/position evaluation, with explicit status dependencies. Do not stop
Laya or confuse a between-phase idle GPU with completed work. All four queues
are in tmux deciform_numeric_trie_consistency_v4_g{01,23,45,67}_20261106.
Failed follow-ups v1/v2/v3 had zero optimizer updates (constructor, double-graph
OOM, and audit-only routing-state errors respectively); preserve their logs,
use v4 as the current execution, and never resume failed versions. Worker code
and all source/config manifests are frozen. Scripts update_final_record.py
--watch maintains CURRENT_RESULTS block in the master document; Laya reporter
maintains LAYA_CURRENT separately. Full notes and commands are in
 docs/experiment_log_20261106_numeric_trie.md. Continue all-three native-MAE
comparison when complete; no test-driven per-source mixture of winners.

Latest user-authorized transfer (2026-11-06 13:54 Asia/Shanghai): implement the
A1–A3 core mechanisms on Laya and train matched same-layout comparisons.
Current root runs/formal_laya_mechanisms_20261106_v2, sequential on 4--5 and
parallel on 6--7. Baselines are completed original Laya same-layout arms in
runs/formal_laya_20261106_v1. The transfer retains full ModernBERT fine-tuning,
original 200k frozen views, original encoder/head initial weights, one epoch,
batch64/3125 steps, original RLCD loss, optimizer/schedule and native evaluation.
Do not inherit trained Laya/D2/static-full or switch baseline to frozen LoRA.
Added: rank16/alpha32 task-routed B(I+R_t)A residuals on 112 encoder projections;
centered bounded head geometry and shared bilinear residual; Choice-only QF.
Original scorer retained. All added branch outputs start at zero and all three
tasks exactly reproduce baseline initial outputs. Added initialization preserves
RNG; pretrained encoder and random head hashes match the original baselines.
Total added parameters 8,085,568; original trainables remain trainable. Added
parameters use existing head LR1e-4, original encoder remains2.5e-5. No new loss.
This includes extra shared A/B and bilinear capacity, so a gain cannot isolate
routing/geometry/QF individually. Candidate attention/layout remains original
Laya; do not claim D2 candidate independence. All native K and text preserved.
Fifteen CPU tests and actual two-GPU zero-update preflights passed, including
initial prediction identity, all-component gradients, checkpoint replay and
reloading, variable K and semantic permutation. Both real trainings reached
step10+. Automatic complete dev6000/test15000 and seven-order Choice/Noul10000
position audit follow. Tmux deciform_laya_mechanisms_v2_{sequential,parallel}_20261106
and report watcher deciform_laya_mechanisms_v2_report_20261106. The reporter
updates only LAYA_TRANSFER block, sharing the current-result document lock.
Source/config hashes are frozen; do not edit active files. v1 failed in audit
metadata handling with zero optimizer updates; v2 contains the corrected tensor
filter and starts fresh. Preserve v1 failure provenance. Exact notes, commands,
metrics and limitations: docs/experiment_log_20261106_laya_mechanisms.md.
The previous numeric-trie auxiliary Score study v4 has completed all four arms;
its results are separate and retained. No new Score follow-up is running here.
