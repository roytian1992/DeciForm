# Score/Noul answer-order robustness (2026-11-04)

**Latest user correction, 2026-11-04:** the active flip-rate study covers
Choice and Noul only. The completed experiment below evaluated Score/Noul,
so its Score flip-rate tables are not the current classification report, and
it did not evaluate Choice. Preserve these old files as historical provenance.
Current output is `runs/formal_answer_order_20261104_v1/classification_report.json`;
Noul is reused and Choice is completed separately under
`runs/formal_choice_order_20261104_v1`. See
`docs/experiment_log_20261104_classification_order.md` for the correction.
Do not run additional Score permutations or report Score flip-rate advantages.

## Question, models, and scope

The user requested order experiments with ordinary LoRA, Kev, JSON-generative
LoRA, and the current D2 full using two available GPUs. This is an inference
audit of the four completed 200k/one-epoch checkpoints, not another training
run. The current A1/A2 ablations occupy all eight GPUs. The audit is queued
after A2 completes, on GPUs 6 and 7; A3 continues on GPUs 0--3.

The hypothesis is that semantic predictions should remain stable when the
same allowed answers are presented in a different order. End-to-end stability
is not guaranteed by the proposed head: its causal backbone still sees the
candidate sequence. Results must include quality and robustness together;
constant incorrect predictions are not an advantage. If D2 is not better,
report that outcome rather than selecting favorable permutations.

References:
- Ordinary LoRA: `runs/formal_main_20261101_v2_one_epoch/lora/epoch_1.pt`.
- Kev: `runs/formal_main_20261101_v2_one_epoch/kev/epoch_1.pt`.
- Generative LoRA: `runs/formal_main_20261101_v2_one_epoch/json_sft/epoch_1.pt`.
- Full: `runs/formal_score_distribution_20261104_v1/d2_ordinal_main/epoch_1.pt`.

Every checkpoint/config/source is fingerprinted before inference. State loading
requires exact trainable parameter names and tensor equality. The model input
and decoding definitions used by the original completed runs are reused.

## Two distinct interventions

1. **Native interface**: all 5,000 score and 5,000 noul rows of the frozen
   in-domain test. Every candidate and its original meaning remain present.
   The five score levels receive ten distinct dihedral permutations (rotations
   and reversed rotations); each value occurs at each position twice. Noul
   receives both possible orders. Ordinary LoRA's scalar heads have no native
   candidate-list input, and JSON-generative noul asks for a boolean without
   listing the two answers. Those cells are explicitly N/A, with only their
   original input evaluated; they are not reported as zero-flip wins.

2. **Common instruction list**: a fixed 2,000-row panel, with 1,000 per task.
   Select 20% within each test source/task by a frozen hash of the row ID,
   without labels, predictions or error-based selection. Append the same
   `Allowed answers: [...]` field to the instruction for every model. Compare
   the permuted list to the same augmented original-order control. Native
   output interfaces remain canonical. This measures sensitivity to an explicit
   redundant instruction list; it is distinct from moving native pointer
   candidate spans. Do not compare its augmented baseline to the unaugmented
   input and call that difference an order effect.

For both pointer models, native permutation physically rearranges option token
spans while preserving state and instruction. After causal encoding, readout
positions are gathered by semantic value. This alignment is identical for Kev
and D2 and is necessary for the numeric score CDF to refer to actual ordered
values. Input tokens are never sorted back before the backbone. Boolean true
and false retain their meanings; ascending score values retain their numeric
meaning. Generative score permutes only the existing allowed-score list and
retains the original JSON format, parser and greedy decoding.

No input or candidate is truncated. Every evaluation example retains the
complete native inventory. The external-source study remains deferred.

## Measures and denominators

Use accuracy for noul and native MAE for score. Probability models use the
same native predictive median score decision; generation uses its parsed
numeric answer. Argmax, mean and nearest-native mean are retained in probability
prediction journals. JSON failures remain incorrect; invalid score output has
the maximum native score-span error. Never invent class probabilities from
a single generated answer.

Report original and each permuted order's quality/validity, reversal flip rate,
any-tested-order flip rate, mean error across orders, and per-example worst
error among the tested orders (not over all 120 score permutations). Also
report aligned probability TV/JS when probabilities exist and absolute score
changes on valid pairs with their explicit denominator. Pairing uses stable
row IDs and semantic outputs. Two thousand source/group-ID cluster bootstrap
resamples give 95% intervals for reversal error change and flip rate; these
intervals do not measure training-seed variation. Include source breakdowns.
Repeat the first unchanged batch per panel/task to measure deterministic
repeat behavior. Fixed paired batching minimizes changes due to batch shapes.

## Execution and recovery

Run root: `runs/formal_answer_order_20261104_v1`.
Environment: `qwen35-train` at
`/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
Working directory: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.

Prepare with `scripts/formal/prepare_order_audit.py`; the root manifest stores
the exact supervisor command, source snapshots, model hashes, panel counts,
data hashes and configuration. The supervisor waits for A2 success and memory
release on GPUs 6/7. GPU 6 evaluates ordinary then generative LoRA; GPU 7
evaluates D2 full then Kev. This uses at most two audit GPUs concurrently.
Per-model journals, progress, checkpoint-load proof, repeat controls and
reports are saved separately. Failed models produce explicit failure/alert
files; no samples are silently omitted or repaired. Root `comparison.json`
is produced only after all four models complete. Resume the supervisor with
`--resume` after examining a failure; complete models are retained and partial
journals resume by their exact panel/row/view keys.

## Validation and trajectory

2026-11-04: audited native interfaces before defining the comparison. Added
isolated order/evaluation code without changing the live training sources.
58 order/distribution tests passed. A real-tokenizer CPU audit checked 99 rows
covering all nine task/source strata and their longest states: all 297
original encodings matched the frozen caches exactly across three encoder
families, and 3,300 order views preserved semantic support and expected input
changes. No GPU inference result is claimed from these CPU checks. No backup
requested.

### 12:49 Asia/Shanghai — automatic evaluation queue verified

Source commit `400a538`; the frozen run contains 22,000 ordinary-LoRA,
72,000 Kev, 67,000 generative-LoRA and 72,000 D2 prediction cases (233,000
total, excluding unchanged-repeat controls). These are view counts over the
10,000 native-panel rows and 2,000 instruction-panel rows, not extra test
examples. Tmux `deciform_answer_order_20261104` is live. Root `status.json`
confirms `waiting_for_gpus`, with A2 still training and GPUs 6/7 still occupied
by that run. No GPU inference or robustness result has occurred yet. All
source hashes were rechecked after queue launch; CPU test/input proof is saved
in root `launch_verification.json`. The queue will start automatically after
A2 exits successfully and releases those GPUs.
