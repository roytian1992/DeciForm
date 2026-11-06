# Minimal D2 inference factor experiments, 2026-11-06

User allocated free GPUs 6–7 for minimal speed optimizations after the exact-output
route cache. This study runs three reversible inference implementations; no
training or checkpoint editing. Existing GPU0–5 work remains untouched.

## Design

All paths load the frozen final independent D2 checkpoint. Reuse exactly the
150-row native-candidate sample from `formal_inference_timing_20261106_v1`:
50 Choice, 50 Noul, 50 five-level main Score. Input encoding through CPU answer
is timed at batch1, BF16 autocast; no truncation or candidate sampling.
Every variant retains the QF, task geometry and ordinal readout unchanged.

| GPU | Trial | Purpose | Numerical expectation |
|---|---|---|---|
| 6 | Route cache + folded A | Precompute $A_t=A+R_tA$ | Algebraically equivalent; BF16 rounding may differ |
| 6 | Route cache + folded B | Precompute $B_t=B+BR_t$ | Algebraically equivalent; BF16 rounding may differ |
| 7 | Route cache + precast factors | Keep frozen A/B in BF16 between calls | Same operand values as BF16 autocast; require observed exact parity |

The task-dependent low-rank update is

$$
s B(I+R_t)Ax=sBA_t x=sB_tAx.
$$

The fold trials compute fixed effective factors in FP32 from the BF16 operands
used at runtime, with TF32 disabled, then store the factors in BF16. They remove
the separate runtime rank-space product/add but may change finite-precision
results. No claim of lossless inference is made solely from the algebra.

Each GPU first measures the original implementation on all150 rows for a fresh
same-device prediction reference. It then runs its route-cache control and trial
variants for three repetitions, rotating variant order and sharing the same
request order within each repeat. All routes warm up on the source/K sample
representatives. CPU affinities are disjoint between the two workers. Compare
each trial with its own GPU's cache control, not raw times across GPUs/runs.
Original-reference timing is one pass; repeated cache/trial timing is primary.

Record every probability vector, prediction, maximum probability discrepancy,
L1 discrepancy, changed IDs, timings by task/source/K, repeat, and peak memory.
The complete original LoRA/router/head weight digest must match after contexts
exit. Setup/loading is excluded and reported separately. Constant route caches
contain no inputs/activations/answers. Both task folding and type conversion are
opt-in and fully restored; no training/evaluation source files are edited.

The small sample can detect numerical changes and latency; it cannot establish
full-test quality preservation. Any changed output prevents a strict lossless
claim even if all sampled argmax/median predictions remain unchanged.

## Reproducibility and execution

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
- Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
- Code: `scripts/formal/inference_factor_trials.py`,
  `scripts/formal/run_inference_factor_trials.py`.
- Code commit: `99e5b07`. Tests: six passed across
  `tests/test_inference_factor_trials.py` and `tests/test_inference_route_cache.py`.
- Output: `runs/formal_inference_factor_trials_20261106_v1`.
- Child directories `fold`, `precast` contain manifests (sample/checkpoint/code
  hashes, environment and CPU/GPU identity), status, requests, block memory
  records, and final summary. GPU logs live at root.
- Tmux: `deciform_inference_fold_g6_20261106`,
  `deciform_inference_precast_g7_20261106`. Failures record traceback in status.

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/run_inference_factor_trials.py --arm fold
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/run_inference_factor_trials.py --arm precast
```

## Trajectory

- 2026-11-06: confirmed GPUs6–7 free, scoped three minimal inference attempts,
  implemented reversible contexts and parity/timing worker, passed six tests and
  launched both persistent tmux workers. Previous route-cache proof and six-model
  benchmark remain preserved. Status: running; no result claim before summaries.
  Backups not requested.

- 2026-11-06 17:38 +08: both latency workers completed and verified exact
  restoration of original weights. GPU6 folding arm1500 requests; GPU7 precast
  arm1050 requests. Each includes150 original-reference requests and450 per
  subsequent variant. Raw data retained in the per-arm requests and summaries.
- User then clarified zero decision changes are a hard acceptance requirement.
  Folded A is rejected: two unique Choice rows changed (six repeated requests).
  Folded B is also withheld because probabilities changed despite zero observed
  decisions changing on this small sample. No folded implementation was enabled
  in production or written to a checkpoint. Future exactness checks fail on the
  first probability difference or semantic decision flip.
- The retained cache+precast path has450/450 exactly equal prediction objects,
  maximum probability difference0 and decision changes0. Added a separate gate
  on all100 Choice/Noul rows in the sample, each original/reversed/shuffled with
  full candidates, comparing original and optimized paths. Score is not reordered.
  Script: `scripts/formal/verify_inference_exact_order.py`; output `exact_order`
  under the same run root, ranks0/1 onGPUs6/7. Status: verification in progress.

## Completed latency screen

Same-GPU controls must be used for ratios; these are implementation timings,
not a new accuracy study. Folding trials also use precast BF16 factors, so the
fold-versus-cache difference is a combined implementation change.

| GPU | Implementation | Mean ms/request | Probability agreement | Decision changes | Disposition |
|---|---|---:|---|---|---|
| 6 | Route cache control | 190.34 | Exact | 0 | Control |
| 6 | Fold into A + BF16 factors | 149.47 | Maximum difference0.01903 | 2 unique rows / 150 | Rejected |
| 6 | Fold into B + BF16 factors | 149.86 | Maximum difference0.01467 | 0 / 150 | Not accepted as lossless |
| 7 | Route cache control | 194.43 | Exact | 0 | Control |
| 7 | Route cache + precast factors | 175.60 | Exact,450/450 requests | 0 / 150 | Retain; order gate required |

Values above must be read from each arm's summary; precise means and all repeat
and task breakdowns remain in JSON. The original unoptimized GPU7 reference
averaged229.84ms (one pass), so the retained path reduces measured latency by
about23.6% relative to that reference, and9.7% versus its repeated cache control.
  The two rejected folding attempts are provenance only, not proposed method speed.

## Final hard-gate verification

Both order shards completed successfully. The original and cache+precast model
each processed100 classification rows (50Choice/50Noul) in three complete
candidate orders: original, reverse and fixed random permutation. All600
requests completed, maximum optimization probability difference0, prediction
differences0 and semantic permutation flips0. No Score row was reordered.
Evidence: `exact_order/summary.json` and both rank request files. The root
`summary.json` records acceptance/rejection and all timing/order results;
`report.md` and the master document's separate INFERENCE_FACTOR_TRIALS block
report only the exact-output path as retained. GPU6–7 workers exited normally.

Final disposition: cache+precast retained as an opt-in tested optimization;
foldedA rejected; foldedB withheld from lossless inference. All three latency
trials restored the original model weights exactly. No checkpoint was replaced.
The persistent user zero-flip requirement is recorded in AGENTS.md. The strict
gate aborts on the first numerical output difference or semantic order flip.
