# Frozen D2 route-cache verification, 2026-11-06

User asked whether D2 inference could be accelerated without quality loss. This
bounded implementation tests one conservative optimization: reuse task-dependent
constant route matrices after loading a frozen model, without reassociating
matrix products or changing any model/input/decoding operation.

## Protocol

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
- Code: `scripts/formal/inference_route_cache.py` (opt-in context) and
  `scripts/formal/verify_inference_route_cache.py` (paired verification).
- Code commit: `ea7d91f`; CPU tests: `tests/test_inference_route_cache.py`, 3 passed.
- Frozen checkpoint: `runs/formal_independent_20261105_v1/epoch_1.pt`.
- Same 150 full-native rows from the original six-model latency benchmark, 50 per
  task, three repetitions per variant, GPU 0 A100, batch1, BF16 autocast, identical
  CPU affinity/threads and generation/decision semantics.
- One loaded checkpoint; alternate blocks original/cached/cached/original/
  original/cached, with the same seeded request permutation within each pair.
  Both paths warm up on the fixed source/K representatives before each block.
  Setup/load and logging excluded; input encoding through returned CPU answers
  included. No activation, prompt, prefix-state, or answer cache across requests.
- Route constants are computed using the original per-layer operation shapes.
  Selected/expanded tensors are cached by task IDs, branch counts and dtype.
  This avoids 64 preparations per request while preserving the per-projection
  batched matrix product and the complete QF/geometry/ordinal head.
- Cache guards reject training, stale router parameters and overlapping request
  contexts. The prototype changes a process-local routing function within an
  opt-in context and restores it on exit; it is not a concurrent serving engine.
  Existing training/evaluation code and the official six-model table stay intact.

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/verify_inference_route_cache.py
```

## Results

Source: `runs/formal_inference_route_cache_20261106_v1/summary.json` and raw
`requests.jsonl`. Manifest records input/config/checkpoint references and code
hashes. `code_commit.json` records the environment command and test result.
All 900 timed requests completed; 450 per variant = 150 unique rows × 3.
Every returned prediction object equals the original-path reference exactly,
including the full probability vector; maximum probability difference = 0.
Seven task/branch-count cache contexts, 485 hits and seven misses including
warmup. No failed or skipped request, no optimizer updates or weight changes.

| Task | Original ms/request | Cached ms/request |
|---|---:|---:|
| Choice | 229.56 | 195.40 |
| Noul | 215.08 | 190.18 |
| Score | 238.94 | 193.75 |
| Equal-task aggregate | 227.86 | 193.11 |

Latency falls 15.25%; serial throughput increases 18.00% (1.180×). Use the paired
227.86 ms original-path measurement for this optimization estimate, rather than
mixing it with the earlier 222.56 ms six-model run. This is exact agreement on
the tested population/hardware, not proof of bitwise invariance for every future
input/device. No full-test metric replacement or retraining is warranted by this
implementation-only check. The main many-level Score specialist supports are
not included in this five-level main-task subset.

## Further optimization boundary

The existing rank-space transformation could be folded into a task-specific
LoRA factor: $A_t=(I+R_t)A$. This is algebraically equivalent but can change BF16
rounding, so it is not included in this strict exact-output result. CUDA graph
capture, fused operators and a dedicated prefix/branch serving path also need
their own numerical checks. Nothing here establishes latency below Kev.

## Trajectory

- 2026-11-06: implemented the separate frozen-only cache and paired verifier;
  tested mixed task IDs/variable branch counts, stale-state rejection and context
  cleanup. Ran all 900 timed requests to completion on otherwise free GPU 0.
  Preserved existing results, checkpoint, active GPU4–5 work and main interfaces.
  Cached path remains opt-in; no blanket production switch. Backups not requested.
