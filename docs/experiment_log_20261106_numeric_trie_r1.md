# Small Decimal-prefix CE/CDF + R1 mean regression

## Scope
User asks to add R1 to the current decimal-prefix model. This authorizes a
small matched Score pilot, not a full-data sweep or a change to R26 main.
No numeric-sequence supervision, new head, candidate sampling, label rounding,
or input truncation. Existing completed controls and all negative findings stay.

## Objective and interpretation
The same prefix distribution assigns mass p_i to every native score s_i.
Define normalized score t_i=(s_i-s_min)/(s_max-s_min), normalized label z,
and predicted mean mu=sum_i p_i*t_i. The data objective is:

$$
L = 0.5 L_{CE} + 0.5 L_{CDF,scaled} + (mu-z)^2.
$$
CDF scale stays3.0383850790577127. Mean weight1 is fixed before launch,
matching the earlier R1 weight. Structural regularization stays unchanged.
This adds mean MSE to current CE/CDF; historical R1 used CDF alone plus MSE.
It is the squared error OF the expectation, not expected squared error; it
adds no variance penalty by itself. Labels are the original values including
fractional labels. Median native MAE remains the primary evaluation rule;
mean regression need not improve median MAE. Do not switch decoder after test.

## Data and matched controls
- Data: data/pilot_numeric_trie_20261106_v1/{asap,continuous}/audit.json.
- ASAP512 train/64dev/128test across8prompts. Test prompt8 has only5records;
  this is small-sample screening, not strong evidence for wide-range scoring.
- Continuous2048train/256dev/512test, equal STS-B/WMT en-zh quotas.
- Exact cached inputs, split IDs, labels and full supports reused; no resampling.
- Completed controls: runs/pilot_numeric_trie_20261106_v1/{asap_control,continuous_control}.
- Fresh Qwen3.5-9B-Base, no inherited trained checkpoint; seed20261101,
  epoch1/global batch16/world2,32 ASAP updates and128continuous updates.
- Same model parameters, initial LoRA/pointer/head hashes, optimizer groups,
  OneCycle schedule, microbatch rows4/token budget8192, checkpoint policies,
  original scoring query and fixed median decoder as the completed controls.
- For an exact computational match, retain the control's causal forward
  containing teacher-forced suffix positions with zero sequence-loss weight.
  The scoring state is before the suffix; only the prefix-distribution R1
  contributes a new gradient. This is not numeric-sequence supervision.
  Real-backbone prompt-only/true/counterfactual checks must pass on representative
  and longest records. Inference remains prompt-only. Future removal of unused
  suffix work is a separate equivalence check, not part of this loss comparison.

## Implementation and execution
- Model/loss: scripts/formal/models_numeric_trie_r1.py.
- Worker: scripts/formal/worker_numeric_trie_r1.py; initial hashes and optimizer
  groups must exactly equal the corresponding completed control.
- Builder: scripts/formal/prepare_numeric_trie_r1.py.
- Reporter: scripts/formal/report_numeric_trie_r1.py.
- Root: runs/pilot_numeric_trie_r1_20261106_v1; asap_r1 uses0-1,
  continuous_r1 uses6-7. R26 KEV2-5 is untouched.
- CWD: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm.
- Python: /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python.
- Persistent tmux queues use scripts/formal/schedule_research_queue.py --root
  runs/pilot_numeric_trie_r1_20261106_v1 --gpus 0,1 (or6,7).
- Per-arm experiment_manifest.json has exact command/env/data/config/source SHA,
  copied source snapshot, launch_command and own-checkpoint resume_command.
- Preflight: full longest backward, native full-support inference, fresh paired
  initialization, finite distributed gradients, no target leakage checks;
  only then train. Checkpoint16, per-step heartbeat, stale alert15minutes.
- Outputs: evaluations/epoch_1.{dev,test_in_domain}.json and rank journals;
  complete.json, status.json, results_summary.json, report.md. Reporter updates
  only NUMERIC_TRIE_R1_SMALL in the master record under its shared lock.
- No results exist at preparation. Finished populations require complete.json,
  exact row counts and successful status. Failed cases remain visible.

## Verification and trajectory
### 2026-11-06 — implement authorized R1 follow-up
40CPU tests passed across prefix model, joint causal path, pilot and R1.
Tests cover zero-weight control value/gradient equivalence, one shared forward,
no extra parameters, affine score-range invariance, fractional targets,
finite gradients/extreme logits, no implicit variance penalty, component logs.
All original control files remain unchanged. Actual GPU training state is
verified from artifacts after launch. No remote backup requested.
