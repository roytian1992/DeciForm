# Numeric-trie Score research — 2026-11-06

## Scope

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`
- Purpose: test a score-specific mechanism that can transfer the pretrained model's numeric prior into a non-autoregressive full-support distribution, with the same three native Score panels used by the current comparisons.
- Status: four fresh-base runs active; no result has been selected.

## Inputs

- Base: `/vepfs-mlp2/c20250513/241404044/users/roytian/RSI_Research/external/baseline_models/qwen35_9b_base`
- ASAP: existing complete native train/dev/test split, 7,784/2,593/2,598 rows, data audit from `data/score_baseline_encodings_20261105_v1/asap/audit.json`.
- Continuous panel: existing STS-B + WMT en-zh split, 8,129/2,500/2,379 rows, audit from `data/score_baseline_encodings_stsb_enzh_20261106_v1/audit.json`.
- Evaluation: complete native grid, median decision, native MAE and equal-source normalized MAE. No candidate or score-grid sampling at evaluation.
- Reference generation replay: `runs/formal_score_recheck_20261106_v1/{asap_json_replay,continuous_json_replay}`.

## Method

The model reads the same prompt prefix as the JSON Score baseline through the Qwen backbone. A fixed `{"score":` prefix is appended only as a representation anchor; no answer value is appended. The root numeric logits use the frozen original Base rows for `0–9`, `.`, and `}`. A compact GRU trie then assigns normalized probability to every legal decimal representation, so multi-digit values, fractional labels, and score ranges of different widths share one exact normalized distribution.

Two objectives are frozen before test evaluation:

- `trie_ce`: soft categorical target interpolation on the native support.
- `trie_ce_cdf`: equal-weight soft CE and the existing ordered CDF-log loss.

Both arms use the same fresh Base, broad LoRA, D2 routing/geometry, seed `20261101`, one epoch, batch 64, and no parent checkpoint. The only comparison is the explicitly registered numeric-trie readout/objective.

## Command and environment

- Environment: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`
- Preparation: `scripts/formal/prepare_numeric_trie.py`
- Queues: `scripts/formal/schedule_research_queue.py`
- Supervisors: `scripts/formal/supervise_numeric_trie.py`
- Root: `runs/formal_numeric_trie_20261106_v1`
- GPU pairs: ASAP CE `0,1`; ASAP CE/CDF `2,3`; continuous CE `4,5`; continuous CE/CDF `6,7`.

## Provenance and checks

- Numeric anchors are saved in `data/score_numeric_trie_20261106_v1/numeric_anchors.pt`; the manifest records the exact Base shard hashes and anchor hash.
- The cache audit re-tokenized every prompt without truncation and checked that every legal score suffix follows the fixed prefix exactly.
- `tests/test_numeric_trie.py`: 14 tests passed.
- All four two-GPU preflights passed longest-input backward, dense batch backward, first global batch backward, native support inference, and zero optimizer updates. Peak preflight memory was 37.40 GiB per GPU.

## Counts and metrics

No test metrics are available yet. The current reference metrics remain in the existing recheck reports; this run will add only completed artifacts after both dev and test evaluation finish. A single seed cannot establish cross-seed stability.

## Trajectory

### 2026-11-06 10:xx Asia/Shanghai — launch

- Goal: find one principled score mechanism that can beat the fresh generative LoRA on ASAP, STS-B and WMT en-zh under the frozen full-grid protocol.
- Actions: rejected a single-token numeric-anchor idea after checking the tokenizer; implemented the exact multi-token numeric trie and two predeclared objectives; built independent audited caches; launched four two-GPU supervisors.
- Current state: all preflights passed; continuous runs are training and ASAP runs have started their first training step.
- Next step: wait for dev/test reports, compare each mechanism on all three panels, and retain a mechanism only if the common criterion is met. No test-based arm selection.
