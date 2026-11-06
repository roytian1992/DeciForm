# Frozen LLM likelihood decisions — 2026-11-06

## Scope and protocol
User-authorized training-free inference on GPUs 0–3; Laya transfer stays on 4–7. Use the existing official instruction Qwen3.5-9B, with no task adapter, fitting, examples, optimizer or test-dependent decoder selection. The experiment-hygiene skill applies.

Primary: actual frozen language-model likelihood of each complete canonical JSON answer, normalized over every legal answer. Include every answer token and closing delimiter. Shared JSON prefix probabilities cancel in normalization. Reuse the current zero-shot task prompt and official chat template. Prefix caching must pass a real GPU comparison against ordinary teacher forcing.

$$
s_i=\sum_t\log P_\theta(a_{i,t}\mid x,C,a_{i,<t}),\qquad p_i=\frac{\exp s_i}{\sum_j\exp s_j}.
$$

Choice/Noul: argmax. Score: predeclared native posterior median for MAE; mean/argmax are diagnostics. Probabilities are conditional on canonical legal outputs, not guaranteed calibrated confidence. Full STSB/WMT supports are 51/101 points; free numeric generation has a broader continuous output space. Preserve original fractional targets. No truncation or sampled-K evaluation.

Secondary rule, from the same logits: renormalize over valid next-token branches at each node (constrained trie decoding). This differs from globally normalized sequence likelihood and is reported separately, never chosen per test source. No trained trie/GRU head is involved. Contextual calibration and domain-conditional PMI are literature-supported follow-ups, not implemented or claimed effective here.

## Inputs and outputs
- Checkpoint: ../models/Qwen3.5-9B, reverify hashes from runs/formal_zero_shot_instruct_20261105_v3/experiment_manifest.json.
- Main: original full test 15,000. ASAP: 2,598. STSB: 1,379; WMT en-zh: 1,000. Explicitly exclude en-de from the legacy continuous file and record the exclusion, preserving every selected row.
- Position: full Choice 5,000 across existing 10-order union; Noul 5,000 original/reversed. No Score flip-rate experiment.
- New output root: runs/formal_zero_shot_logits_20261106_v2. Immutable manifests, raw scores/probabilities, durable journals, progress and failures.
- Environment: qwen35-train; exact executable/packages/commands in manifest.
- Status: running, four active GPU workers verified; completed aggregate metrics pending.

## Literature
- Zhao et al., ICML 2021, Calibrate Before Use: Improving Few-Shot Performance of Language Models. https://proceedings.mlr.press/v139/zhao21c.html
- Holtzman et al., EMNLP 2021, Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. https://arxiv.org/abs/2104.08315
- Liu et al., EMNLP 2023, G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. https://aclanthology.org/2023.emnlp-main.153/ . Its probability-weighted numeric score motivates the expectation diagnostic here; this is not a reproduction of its CoT prompt or GPT-4 setup.
- EleutherAI LM Evaluation Harness official documentation: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md . Existing loglikelihood/multiple-choice evaluation and separate normalization metrics.
- These establish likelihood/prior-bias baselines; they do not guarantee improvement or order invariance here.

## Trajectory
- 2026-11-06: GPUs 0–3 verified free. Frozen JSON prompt/data protocol inspected. Implement a separate inference-only study; no existing result replaced. No backup requested.

## Implementation / checks / commands
- Core: scripts/formal/zero_shot_logits.py; worker: worker_zero_shot_logits.py; preparation/supervision: prepare_zero_shot_logits.py / schedule_zero_shot_logits.py.
- Original LM head, full-vocabulary log-softmax, breadth-first exhaustive answer-prefix traversal; no learned modules and no beam pruning. Include closing JSON delimiter; exclude EOS. Common answer tokens are forced once, with their identical probability factor cancelling globally.
- Local constrained-trie rule renormalizes only over legal next tokens at each node; deterministic edges contribute probability 1. Report both rules globally, without per-source winner selection.
- 23 CPU tests passed: exact dense-vs-cached probability enumeration on a deterministic autoregressive test model, 101 leaves, duplicated cache parents, mixed padding, raw fractional labels, semantic ID/permutation mapping and prompt leakage guards; existing JSON protocol tests included. Synthetic model is only a unit-test fixture, never an evaluation substitute.
- Real Qwen GPU preflight passed on all task types, K up to 101 and long ASAP input6074 tokens. Selected teacher-forced versus cached log probability differences <=0.124103 under BF16; left-padding/batching probability difference <=0.009261. This is numerical agreement within declared tolerance, not bit-exact equivalence. Suffix enumeration reversal observed delta0. Separate full native longest-input/batch preflights run before each panel/view. The preflight accidentally used one en-de record for the K101 arithmetic check; it is not included in any reported test panel.
- Position audit uses IDENTICAL row groups across all permutations, built from each record's maximum view length. Canonical output IDs/values remain stable. This prevents changing batch membership from confounding input order. Floating-point inference is still subject to ordinary BF16 numerical effects.
- v1 was stopped after partial main inference to enforce that position batching policy before any position evaluation. Its logs and predictions are preserved under runs/formal_zero_shot_logits_20261106_v1; none are final evidence. v2 restarts inference with the frozen corrected worker. The core scorer/preflight are unchanged.
- Checkpoint files rehashed against the original zero-shot manifest. Code/input files snapshotted/hashed in v2; complete tests/logs/preflight hashes recorded. No parent task checkpoint.
- Total 79,977 request/view evaluations; each produces two rule records, 159,954 journal lines. This includes repeated rows across permutation views; unique populations are the panel counts above.
- Launch: qwen35-train Python scripts/formal/schedule_zero_shot_logits.py --root runs/formal_zero_shot_logits_20261106_v2, in tmux deciform_zero_shot_logits_20261106.
- Resume the same command only after inspecting any shard failure; no automatic silent retry. Status/progress, worker logs, failure.json and alerts.json expose crashes/stalls. Completed journals are validated by expected IDs/counts/hash before aggregation. Results update a separate ZERO_SHOT_LOGITS block of the master document under runs/final_record_update.lock.

### 2026-11-06T14:35:26.869612+08:00 — launch verified
- Git commit 9c20776. All four workers reached real main-test inference on GPUs 0–3; 11,704/15,000 distinct main rows completed at this check (each has two rule records). No shard failure. Exact evidence: runs/formal_zero_shot_logits_20261106_v2/launch_verified.json.
- Automatic panel aggregation and master-record update active in tmux. Active Laya transfer on 4–7 preserved. No additional training is launched. Estimated total inference about 30–50 minutes from this early throughput; full high-K position views dominate, so this is provisional.
