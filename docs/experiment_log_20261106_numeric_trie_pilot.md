# Small numeric-prefix CE/CDF and sequence-supervision replay

## Scope
User authorizes this follow-up after the current small Score proper pilot.
This repeats existing N2/T2 mechanisms on the fixed SMALL populations; it is
not a newly invented method or a new full-data training cycle. The historical
full-panel results in experiment_log_20261106_numeric_trie.md remain valid:
T2 was not uniformly better than N2. No changes to R26 main experiments.

## Matched experiment
- Input selection: data/pilot_score_proper_20261106_v1, unchanged IDs/order/text,
  numeric labels and full native supports in every split.
- ASAP: train512/dev64/test128, eight prompts. Only five prompt8 test records;
  small-sample uncertainty precludes strong wide-range conclusions.
- Continuous: train2048/dev256/test512, equal STS-B/WMT en-zh quotas.
- Reuse existing numeric-trie encodings selected by exact parent indices;
  verify source rows equal the frozen small records. No resampling/truncation.
- New encodings/data provenance: data/pilot_numeric_trie_20261106_v1.
- Fresh Qwen3.5-9B-Base, no inherited trained checkpoint/head, seed20261101,
  one epoch, batch16, 32/128 steps. Identical paired model and initialization.
- Both arms retain decimal-prefix CE/CDF with existing ordinal scale3.038385079.
  Token-NLL weight is0 for control and0.5 for treatment.
- Both arms execute the same teacher-forced causal forward, including suffix
  length/packing. The control uses zero auxiliary gradient. The score query is
  before every target digit; GPU preflight compares prompt-only, true suffix,
  and counterfactual suffix on representative/longest records.
- Auxiliary NLL uses frozen original LM-head rows over 12 numeric characters,
  not the whole vocabulary. It is distinct from the compact GRU trie path CE.
  A single causal forward avoids the previous double-forward memory issue.
- Primary decoding: native predictive median; report MAE and equal-source or
  equal-prompt NMAE. Other decoders are diagnostic; no test-based selection.
- The trie prompt/readout differs from original D2; compare control/treatment
  for auxiliary supervision. D2 comparisons describe whole-system changes.

## Implementation, environment and outputs
- Model: scripts/formal/models_numeric_trie_pilot.py, existing shared-forward
  models_numeric_trie_consistency.py and numeric_trie.py remain unchanged.
- Worker: scripts/formal/worker_numeric_trie_pilot.py.
- Builder: scripts/formal/prepare_numeric_trie_pilot.py.
- Reporter: scripts/formal/report_numeric_trie_pilot.py.
- Python: /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python.
- CWD: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm.
- Root: runs/pilot_numeric_trie_20261106_v1, per-arm config/manifest/source hashes.
- GPUs0-1: asap_token then continuous_token; GPUs6-7: asap_control then
  continuous_control. Queue waits for corresponding current ASAP supervisor
  completion AND released GPU memory; it does not preempt train/evaluation.
- Resume only own checkpoint via each manifest's resume_command. Stop on
  preflight/hash/dependency failures. Automatic full-native small Dev/Test.
- Results: per-arm evaluations/epoch_1.{dev,test_in_domain}.json and rank journals;
  results_summary.json and report.md. Master record NUMERIC_TRIE_SMALL block is
  updated under shared runs/final_record_update.lock. Pending results remain placeholders.
- CPU checks: 29 passed, including probability mass, exact decimals, causal
  alignment, zero-weight objective/gradient equivalence, and distinct auxiliary
  backbone gradients. Actual-backbone preflight required before training.
- No remote backup requested.

## Trajectory
### 2026-11-06 Asia/Shanghai — prepare authorized small follow-up
Found completed N2/T2 implementations and preserved their mixed results.
Created isolated small-study adapter/worker/builder/reporter; all158 historical
source hashes match. Do not edit current running workers/configs. Queue launch
and actual step status must be verified from current output artifacts.
