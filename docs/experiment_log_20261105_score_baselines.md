# Score baselines, 2026-11-05

## Authorized scope and comparison

The user explicitly cancelled ordinary LoRA in this Score study and authorized
Kev and JSON-generative LoRA on GPUs 0–3. Four fresh-base runs cover the two
frozen panels; the existing S1/S2/S3/S4 Score specialist checkpoints/results are
reused. The candidate-independent main model continues on GPUs 4–7 unchanged.
No additional ordinary LoRA model or per-K threshold bank is adopted or queued.

| Panel | Training | Dev | Test | Runs | Complete allowed scores |
|---|---:|---:|---:|---|---|
| ASAP author fold 0 | 7,784 | 2,593 | 2,598 | Kev, JSON LoRA | Native 4/5/6/11/31/61 integer levels |
| STS-B + WMT en-de/en-zh | 11,457 | 3,500 | 3,379 | Kev, JSON LoRA | 51/101-point grid for pointer models; original continuous range for JSON |

Both baselines retain the established formal all-target LoRA configuration:
rank 16, alpha 32, dropout .05. Kev keeps the original pointer head and has
45,376,000 trainable parameters. JSON-generative LoRA keeps its original frozen
vocabulary readout and has 43,278,336 trainable parameters. Neither receives
our routing, geometry, QF, ordinal readout, structural penalty or CDF loss.

- Kev uses categorical CE on native integer scores; continuous labels use the
  exact same adjacent-grid soft-label interpolation as S3/S4, without rounding.
- JSON LoRA uses teacher-forced token CE for an explicitly requested JSON object
  containing the original numeric label. ASAP requires an allowed integer;
  continuous tasks allow finite fractional numbers within the original range.
- Full original state, essay, task, rubric and score range are retained. Only
  baseline-specific encoding/output formatting differs; no input truncation,
  score candidate sampling, new record selection or label modification.
- Each run initializes from the original Qwen3.5-9B base, no trained checkpoint.
  Initial LoRA (and Kev pointer) hashes must match the respective formal
  baseline's fresh initialization. Seed 20261101, batch 64, two GPUs, one epoch,
  OneCycle with 30% warmup, fixed endpoint: 122 ASAP or 180 continuous updates.
- Half the decoder layers checkpoint activations. ASAP packing budget 4096;
  continuous train/eval packing budgets 8192/16384. A longer single request is
  processed completely. These are the existing specialist settings.

## Evaluation contract

Every dev/test row is retained. Probability models use the frozen native-grid
median, with the same four decision-rule diagnostics as existing specialists.
Generated JSON uses its actual numeric answer; no invented distribution, median,
NLL or calibration values are assigned to a generated scalar. Thus continuous
JSON is not restricted to the pointer's representation grid, which is disclosed
as a baseline interface difference and not a matched loss-only comparison.

JSON parsing rejects duplicate/extra keys, malformed JSON, nonfinite numbers,
wrong types and out-of-range values. No regex extraction, repair, clipping or
rounding. Only actual trailing EOS/pad tokens are removed, including mixed
terminal sequences. Invalid outputs remain in the denominator with full-range
error; validity and valid-only MAE are also reported. This penalty is distinct
from ordinary numeric MAE whenever invalid outputs occur.

Primary reports: per-prompt/source native MAE and equal-prompt/source NMAE.
Original labels and per-example predictions remain available. WMT en-zh is the
user-requested main WMT view; German results remain saved and the training pool
stays identical to S3/S4. Do not reinterpret an en-zh-only result as the original
three-source aggregate. This is not official WMT z_mean/Pearson leaderboard
replication. No source selection or decoder choice based on these new test runs.

## Reproducibility, command and recovery

Project/cwd: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
Python: `../anaconda3/envs/qwen35-train/bin/python` (qwen35-train).
Base: `../RSI_Research/external/baseline_models/qwen35_9b_base`.
Data: `data/score_specialist_asap_fold0_20261104_v1` and
`data/score_specialist_continuous_20261104_v1`.
Separate baseline cache: `data/score_baseline_encodings_20261105_v1`;
original datasets, original Kev caches and live-run source remain untouched.
The cache audits freeze all exact row/cache paths, counts and SHA256 checksums.
Root: `runs/formal_score_baselines_20261105_v1`.

Preparation: `python scripts/formal/prepare_score_baselines.py`.
Queue: `python -u scripts/formal/schedule_score_baselines.py --root
runs/formal_score_baselines_20261105_v1`, tmux `deciform_score_baselines_20261105`.
The queue owns GPU pairs 0–1 and 2–3; each must have a successful completed
D2 size-curve dependency and less than 1000 MiB/card before reuse. GPU pairs
are reserved while a supervisor is active, preventing two queued jobs racing
for the same idle pair. Priority: ASAP Kev, ASAP JSON, continuous Kev, continuous
JSON. The next pending arm takes the next available pair.

Each supervisor verifies frozen config/source hashes, then runs zero-update
GPU preflight, fresh training and automatic dev/test evaluation. It records
status/progress, preflight/train logs, failures and 15-minute stale-progress
alerts. Own optimizer/RNG checkpoints save every 25 updates. Resume command is
in each runtime manifest, using `supervise_score_baselines.py --manifest
<runtime_manifest.json> --resume`; do not restart an existing root scheduler.

## Verification and trajectory

Initial CPU preparation included unused ordinary scalar encodings. The user
then cancelled that model; its draft threshold bank was removed before any
GPU execution. No ordinary baseline checkpoint, config or training was created.

52 focused tests passed: strict native/continuous JSON targets and parsing,
malformed-output denominators, actual terminal-token stripping, unchanged
probability metric replay, native CE value/gradient equivalence at K=4–101,
continuous-label interpolation, and existing ASAP/specialist contracts.
Real-tokenizer cache parity and complete-label JSON roundtrips are recorded in
`data/score_baseline_encodings_20261105_v1/baseline_preflight_cpu.json`.

Status at source freeze: implementation and CPU tests complete; GPU preflight
and first updates will be recorded below. Local scoped source commit plus
per-run source snapshot; no remote backup/push requested.

### 2026-11-05T01:54:27.202390+08:00 — two real training launches verified

- Source commit `3c8d9d4`; all frozen source/config hashes verified. 52 tests, 78 real-tokenizer cache comparisons, and 31,311 exact original-label JSON roundtrips passed.
- ASAP Kev on GPUs 0–1 and ASAP JSON-generative LoRA on GPUs 2–3 both passed zero-update GPU preflight and reached update 1/122 with finite loss/gradients. Preflight peaks: 47.9201 / 49.3597 GiB. Initial LoRA and Kev pointer hashes match the original formal baselines, with no inherited trained weights.
- Both existing D2 size-curve dependencies completed successfully before allocation. Continuous Kev and continuous JSON remain queued for the next released pairs. Automatic full-native/full-range dev/test evaluation is part of each worker.
- No failures or alerts at verification. Main independent D2 on 4–7 continues. Current launch evidence is `runs/formal_score_baselines_20261105_v1/launch_verified.json`; use live status/progress for later updates. No remote backup/push.

### 2026-11-05T08:09:58.078664+08:00 — all four baseline arms complete

- Scheduler status `runs/formal_score_baselines_20261105_v1/status.json` is complete; all four arms exited 0, with no failure or alert files. GPU pairs 0–3 are released.
- ASAP test (2,598 rows): Kev native median MAE/NMAE `1.05427/0.110759`, JSON-generative MAE/NMAE `0.80062/0.087341`; both validity 1.0.
- Continuous test (3,379 rows): Kev native median MAE/NMAE `7.20263/0.124375`, JSON-generative numeric MAE/NMAE `7.64234/0.113773`; both validity 1.0.
- Existing matched specialist references remain S1/S2 ASAP NMAE `0.109069/0.105900` and S3/S4 continuous NMAE `0.108133/0.101199`; generated JSON has a distinct scalar interface and no probability metrics, so comparisons are reported with that caveat.
- 4–7 independent candidate D2 is also complete: 3,125/3,125 updates, 200,000 visits, test Choice accuracy `0.7302`, Noul accuracy `0.8880`, Score median MAE `0.4168`; source `runs/formal_independent_20261105_v1/complete.json` and test evaluation.
- D2 100k size curve completed full-native evaluation: test Choice `0.7126`, Noul `0.8802`, Score MAE `0.4450`; source `runs/formal_d2_curves_20261104_v1/d2_100k/evaluations/epoch_1.test_in_domain.json`.
- These are endpoint results from one seed/one epoch, not cross-seed significance. External evaluation remains deferred. No remote backup/push.
