# Small inference-latency benchmark, 2026-11-06

User requested average inference time for core baselines on a small dataset.
This adds no training and changes no existing evaluation population or weights.

## Frozen protocol

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
- Input: `data/formal_200k_20261101_v1/cache/test_in_domain.views.jsonl` (15,000).
- Fixed sample: 150 distinct rows, 50 each Choice/Noul/Score. Proportional source
  quotas within task; one seeded draw per character-length stratum within source.
  Selection uses neither labels nor model predictions. Full original candidate
  sets and text retained, including Banking77 K=77 and CFPB K=21.
- Six trained models: original ordinary LoRA, Kev, JSON-generative LoRA (all
  `runs/formal_main_20261101_v2_one_epoch`); final independent D2
  (`runs/formal_independent_20261105_v1`); original Laya sequential and parallel
  (`runs/formal_laya_20261106_v1`). Each used the same 200k main training records.
- Same physical A100 GPU 0, one model resident at a time, fresh process per arm.
  Batch size 1; 4 CPU threads with the same CPU affinity; BF16 autocast, eval and
  inference mode. Native inference implementations and unmerged adapters.
- Warm up the longest sampled row for each source and each represented K.
  Three timed passes use the same seeded request orders for all arms.
- Synchronized wall time includes tokenization, collation/transfer, model forward
  or full greedy JSON generation, and CPU answer decoding. Model/checkpoint load,
  warmup, disk/network I/O, result logging and metric calculation are excluded.
  No request or answer reuse, cross-request prefix cache, truncation, or sampled K.
- Report mean/median/P95, per task/source/K, serial requests per second, memory.
  Serial throughput is reciprocal mean latency, not saturated serving throughput.
  Laya has a smaller ModernBERT encoder; this is an implementation/system
  comparison, not proof of architectural speedup at equal backbone capacity.
  No quality conclusions from this small subset; full main quality remains fixed.

## Execution and evidence

- Script: `scripts/formal/benchmark_inference.py`.
- Output: `runs/formal_inference_timing_20261106_v1`.
- Exact row IDs/contents: `sample.jsonl`; protocol, hashes: `config.json`.
- Per arm: `environment.json`, `status.json`, `requests.jsonl`, `summary.json`.
- Aggregate: `status.json`, `summary.json`, `report.md`; master document receives
  only the separate `INFERENCE_TIMING` block under the shared results lock.
- Environment: `qwen35-train`, Python executable
  `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python`.
- tmux: `deciform_inference_timing_20261106`.

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/benchmark_inference.py prepare
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/benchmark_inference.py run
```

The run fails visibly rather than skipping failed inputs. Each worker flushes
raw requests individually, updates progress every 25 requests, records failures
and tracebacks. A failed worker stops the queue and retains all evidence.

## Trajectory

- 2026-11-06 16:39 +08: implemented the fixed sample, native checkpoint loaders,
  synchronized timer, per-request records, progress and aggregate reporting.
  Status: preparation/verification; no inference results claimed yet.
  Existing GPU 4–5 training remains undisturbed. Backups not requested.
- 2026-11-06 16:52 +08: all six arms completed on GPU 0. Actual queue interval
  16:42:04–16:52:14 (10m10s, including model loads). Code commit `3470106`;
  execution is additionally recorded in `launch_verified.json`. Three CPU tests
  passed. Final `verification.json` confirms exactly 150 matching IDs × 3 per
  arm, all 2,700 requests valid, complete probability supports, unchanged source
  and sample hashes, and no code changes during execution. All six outputs are
  complete; no failed/skipped requests. No backups requested. GPU 0 released.

## Completed measurements

Source: `runs/formal_inference_timing_20261106_v1/summary.json` and per-arm raw
`requests.jsonl`. All latency columns below are milliseconds per request.

| Model | Mean | Choice | Noul | Score | P95 | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary LoRA | 109.68 | 197.88 | 58.03 | 73.12 | 338.43 | 15.95 |
| Kev | 93.46 | 89.82 | 88.12 | 102.44 | 122.78 | 15.21 |
| Generative LoRA | 458.10 | 545.68 | 407.25 | 421.37 | 590.95 | 17.21 |
| Independent D2 | 222.56 | 224.10 | 210.07 | 233.53 | 304.06 | 15.49 |
| Laya sequential | 18.90 | 18.83 | 18.32 | 19.54 | 21.85 | 1.66 |
| Laya parallel | 18.82 | 18.64 | 18.33 | 19.48 | 21.74 | 1.66 |

D2 takes 2.38× Kev's latency and 0.486× generative LoRA's latency on this
matched sample and implementation. Thus this measurement supports a speed
advantage over generation, but does not support a speed advantage over Kev.
Laya uses a 421M-parameter model, a substantially smaller backbone. Main Score
examples have five native levels; this benchmark does not characterize latency
for all supports from the separate many-level Score specialist study.

The sample includes 6 Banking77 K=77 and 12 CFPB K=21 examples. Ordinary LoRA
averages 256.97 ms on the former and 541.08 ms on the latter; different input
lengths mean K alone does not explain runtime. All source/K breakdowns, repeated
pass means and CPU encoding contributions are retained in the aggregate JSON.
