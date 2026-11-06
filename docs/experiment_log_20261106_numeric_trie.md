# Numeric-trie Score research — 2026-11-06

## Scope and current state

Project: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm.
The first four numeric-trie runs are complete. They improve STS-B and WMT en-zh
against the fresh JSON-LoRA replay, but do not beat ASAP native MAE. The target
of one mechanism winning all three datasets has therefore not been met.
The main 200k D2 model and A1–A3 are unchanged.

Follow-up root: runs/formal_numeric_trie_consistency_20261106_v4.
Two ASAP arms on 0–1 and 2–3 have passed the mandatory GPU preflight and entered the train worker; two
continuous arms are queued behind the completed training, native evaluation
and position audit of Laya on 4–5 and 6–7. Live status, not this timestamped
sentence, is authoritative. All four queues run in persistent tmux sessions.

## Inputs and comparison contract

- Base: /vepfs-mlp2/c20250513/241404044/users/roytian/RSI_Research/external/baseline_models/qwen35_9b_base.
- Python: /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python.
- ASAP: existing native train/dev/test 7,784/2,593/2,598 rows; audit data/score_numeric_trie_20261106_v1/asap/audit.json.
- STS-B + WMT en-zh: train/dev/test 8,129/2,500/2,379; audit data/score_numeric_trie_20261106_v1/continuous/audit.json. Test contains STS-B 1,379 and WMT 1,000.
- Each method trains separately on the two panels. It is not one checkpoint jointly trained on all three datasets.
- Seed 20261101, one epoch, global batch 64, fresh Base with zero-B LoRA and no inherited checkpoint/head. ASAP 122 updates, continuous 128.
- Entire original input, complete native score support, fractional labels preserved, median decision fixed before test evaluation. Native MAE is primary; equal-prompt/source NMAE also retained.
- Generative references: runs/formal_score_recheck_20261106_v1/asap_json_replay and continuous_json_replay.
- Frozen anchor artifact: data/score_numeric_trie_20261106_v1/numeric_anchors.pt; SHA256 141ab8d8a655c5fd28f36b3c8861c31ed32df3c706e2788cd7cd022c859a0b6e.

## Mechanism and objectives

The shared JSON-baseline prompt receives the constant prefix {"score":. The
representation before the first answer digit supplies the scoring query.
The root next-digit prior uses frozen original Base LM-head rows for the
alphabet 0123456789.}. A compact GRU handles subsequent prefixes. Every legal
native decimal value is enumerated and leaf probabilities sum to one.
Only root logits come directly from the pretrained LM; subsequent GRU logits
are learned. This is not exact autoregressive backbone sequence likelihood.

For a legal value y with decimal-prefix edges E(y),

$$
p_\theta(y\mid x)=\prod_{(v,c)\in E(y)}p_\theta(c\mid v,x).
$$

N1 uses soft target-interpolation CE. N2 uses equal CE/CDF-log weights. Both
are predeclared common mechanisms evaluated on all three datasets.

Follow-up T1/T2 preserve the same distribution and add teacher-forced numeric
suffix supervision through the backbone's original frozen 12-character rows:

$$
\mathcal L_{\mathrm{T1}}=\mathcal L_{\mathrm{CE}}+0.5\mathcal L_{\mathrm{token}},
\qquad
\mathcal L_{\mathrm{T2}}=0.5\mathcal L_{\mathrm{CE}}+0.5\mathcal L_{\mathrm{CDF}}+0.5\mathcal L_{\mathrm{token}}.
$$

The auxiliary loss averages exact JSON suffix-token NLL within each example,
then the training engine applies the existing sample-weighted distributed
aggregation. It uses the fixed numeric alphabet, not the entire vocabulary.
Gold decimals use baseline JSON serialization without rounding. v3 evaluates
the prefix distribution and token losses in **one shared causal forward**.
The scoring query precedes every target digit; inference uses the prompt only.
This is auxiliary supervision, not a KL constraint that equates two distributions.

Development evidence motivating T1/T2: N1 is better than the replay on several
small-range ASAP prompts but worse on wide-range prompts 7 and 8. Their dev
MAE is 2.533546/3.458333 versus replay 2.258786/3.194444. Remaining multi-digit
modeling is a hypothesis to test; improvement is not guaranteed.

## Completed results

Source for each row: evaluations/epoch_1.test_in_domain.json with complete.json
and successful status.json under its run directory. These are native MAE.

| Method | ASAP (2,598) | STS-B (1,379) | WMT en-zh (1,000) |
|---|---:|---:|---:|
| Fresh JSON-LoRA replay | 0.803695 | 0.450487 | 15.555333 |
| N1 trie CE | 0.823326 | 0.434397 | 11.365333 |
| N2 trie CE/CDF | 0.829869 | 0.433744 | 11.130000 |

N1/N2 ASAP equal-prompt NMAE is 0.084060/0.086287, versus replay 0.086374.
Thus both improve the equal-prompt normalized metric, while both remain worse
on native MAE. The earlier log incorrectly said both metrics were worse; this
is corrected here. Neither is a three-dataset native-MAE win.

First-study root: runs/formal_numeric_trie_20261106_v1, arms
asap_n1_trie_ce, asap_n2_trie_ce_cdf, continuous_n1_trie_ce,
continuous_n2_trie_ce_cdf. ASAP training took 3,314/3,304 seconds plus
644/641 seconds of evaluation; continuous training took 262/261 seconds plus
46/46 seconds of evaluation. These are measured completed runs, not guaranteed
T1/T2 runtimes. All 46,161,164 trainable parameters start from the same protocol.

## Validation and recovery

- First study: 14 CPU tests and all four actual two-GPU preflights passed.
- Correct first-study peak preflight memory: ASAP 54.371 GiB; continuous 37.399 GiB. The earlier text incorrectly assigned the continuous figure to all arms.
- Follow-up: 23 total CPU tests passed, including single-forward gradients, exact decimal target serialization, teacher-forcing alignment, full support, and counterfactual-label independence.
- Actual-backbone preflight adds prompt-only versus true/counterfactual-suffix checks on a representative and longest input, followed by full longest backward, dense batches and native evaluation. Proofs are in preflight.causal_prefix.json and preflight_passed.json.
- v1 follow-up failed before training because token_weight was forwarded to the distribution-only objective constructor. v2 fixed this but failed the longest-input backward with CUDA OOM: two long-text computation graphs were retained concurrently. Neither produced optimizer updates or valid metrics. Outputs are retained under their versioned roots.
- v3 exposed a missing routing-context clear between causal audit forwards and failed before any optimizer update; v4 corrects the audit and adds a stateful routing guard to unit tests.
- v3/v4 fix the root cause by sharing the causal forward. No input truncation, record dropping, reduced support, changed batch size or inherited weights was introduced.
- Every manifest pins data/config/source SHA256 and snapshots source files. Supervisors verify initialization, zero-B LoRA, parameter inventory and zero-update preflight before training. Resume is only from the run's own checkpoint.
- Entry points: scripts/formal/prepare_numeric_trie_consistency.py; schedule_research_queue.py --root runs/formal_numeric_trie_consistency_20261106_v4 --gpus 0,1 (or 2,3 / 4,5 / 6,7). Per-arm launch/resume commands are in experiment_manifest.json.
- Queue completion dependencies explicitly cover Laya's entire worker, including post-training evaluations. An idle gap before evaluation is not treated as completion.
- Automatic dev/test and overwrite-style master-table refresh are configured. Failed/pending runs remain placeholders and are not represented as completed measurements.

## Trajectory

1. Implemented and completed N1/N2 on both panels; the joint target is unmet.
2. Diagnosed multi-digit ASAP weaknesses on dev and froze two auxiliary-loss hypotheses.
3. Corrected the constructor issue and identified double-forward OOM during mandatory preflight, before optimizer updates.
4. Implemented shared-forward T1/T2, passed 23 CPU tests and exact target serialization audit, launched actual ASAP preflights on 0–3, and queued continuous arms behind Laya on 4–7.
5. Next: inspect successful preflight and real step progress; compare all three native-MAE outputs when complete. Single-seed evidence will not be called cross-seed stability, and a dataset-specific winner will not be presented as one universal mechanism.

### 2026-11-06 12:44 Asia/Shanghai — shared-forward preflight passed

Both ASAP arms passed all actual GPU checks, with peak memory 54.403511 GiB
and zero optimizer updates in preflight. Prompt-only versus true and changed
suffix probabilities were exactly equal (maximum observed difference 0),
including the longest training input. Supervisors now launched train workers;
real optimizer-step progress is checked separately before reporting training
as established. Both continuous queues still wait on Laya completion.
The master record was overwritten with N1/N2 complete results and T1/T2
placeholders. No remote backup was requested or made.
