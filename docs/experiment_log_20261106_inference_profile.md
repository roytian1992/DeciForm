# D2 inference latency diagnosis, 2026-11-06

User asked why final independent D2 is slower. This diagnosis changes no
checkpoint, training run, model outputs, or official latency result. A separate
script wraps existing forward calls with CUDA events, then verifies predictions
against the uninstrumented functions on the same loaded checkpoint.

## Protocol and artifacts

- Script: `scripts/formal/profile_inference_stages.py`.
- Output: `runs/formal_inference_profile_20261106_v1`.
- Source protocol, checkpoint paths/hashes and CPU settings are inherited from
  `runs/formal_inference_timing_20261106_v1/config.json`.
- Six unchanged, full-native rows from the fixed timing sample: the median
  character-length row within AG News, CFPB, Banking77, BoolQ, HelpSteer3 and
  Prometheus. Three repetitions per row/model/mode, after warmup. This intentionally
  covers specific paths; it does not re-estimate the formal sample's mean latency.
- GPU 0, A100 80GB; single request, BF16 autocast, inference mode, four CPU threads
  and identical affinity. Both models use their original checkpoint and functions.
- Each worker loads a fresh model, warms up, measures 18 uninstrumented requests,
  installs event-only wrappers, warms up, measures 18 instrumented requests.
- Exact evidence: `manifest.json`, `sample.jsonl`, `{kev,d2}/results.json`, logs
  and status JSON. All probabilities match their uninstrumented references
  exactly on these requests; all tasks and native candidates retained.
- Environment: `qwen35-train`. No subagent, training, backup or external messaging.

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python scripts/formal/profile_inference_stages.py
```

## Findings

Official full timing sample: D2 222.56 ms, Kev 93.46 ms. Tokenization accounts for
2.47 and 2.33 ms respectively. Resident parameter counts are 7,982,988,806 versus
7,982,060,544: only 928,262 additional D2 parameters, approximately 0.012%.
Both already adapt the broad projection inventory; extra model size cannot
explain their 2.38× latency ratio by itself. The generative model additionally
retains the vocabulary output head, so parameter inventories differ by interface.

Diagnostic means below are CUDA event timeline intervals, including stream idle
and launch gaps, not isolated hardware kernel compute times. CPU enqueue timings
overlap GPU execution and must not be added to these intervals. Wrappers cause
overhead: uninstrumented/instrumented means are 95.40/97.00 ms for Kev and
238.38/246.44 ms for D2. These six-row means are not the official 150-row means.

| Stage | Calls/request | Mean CUDA timeline ms |
|---|---:|---:|
| Kev decoder stack | 32 | 91.30 |
| Kev head | 1 | 0.23 |
| D2 prefix decoder stages | 32 | 100.14 |
| D2 candidate decoder stages | 32 | 107.91 |
| D2 route preparation | 64 | 22.71 |
| D2 prefix state expansion | 32 | 1.76 |
| D2 routing clear | 64 | 1.31 |
| D2 entire head | 1 | 0.99 |

Code explains the shape of the result:

1. `independent_candidates._layer_step` invokes each of 32 decoder layers twice:
   once for the common text prefix, then once for the packed independent candidate
   and query branches. Kev's measured `forward_rows_batch` encodes the whole
   single-question row once. D2 does not run K complete backbones serially, nor
   recompute the entire prefix K times. The two-stage execution introduces more
   calls, small branch operations, weight access and cache work; token counts or
   FLOP savings alone do not guarantee lower single-request latency.
2. `routing` recomputes centered/bounded task maps for both stages of every layer:
   64 route preparations. This is convenient for training/checkpoint replay; the
   maps are fixed functions of trained weights and task at inference. Route
   preparation CPU enqueue time is 42.34 ms in this instrumented run and overlaps
   the decoder's GPU work; it cannot be treated as an additional 42 ms wall cost.
3. Every routed projection computes a separate rank-space batched matrix product
   and addition. Those costs are included inside the decoder-stage intervals,
   not the route preparation row. Their isolated causal contribution has not
   been measured by this diagnostic.
4. The entire head is about 1 ms, including geometry, QF for Choice and the
   ordinal distribution for Score. Removing these modules is not supported as
   the principal latency remedy. Training losses are not evaluated in inference.

## Optimization implications (not implemented here)

Cache the bounded per-task route matrices and head geometry for frozen inference.
For column-vector convention, the routed update is

$$
\Delta W_t x = s B(I+R_t)Ax = s B A_t x,
\qquad A_t=(I+R_t)A.
$$

Thus a per-task effective low-rank factor can remove the runtime rank-space
product while retaining the real-arithmetic function. Full per-task weight
merging is also possible but has a larger memory footprint. BF16 reassociation
can change rounding; validate full probabilities and decisions before claiming
equivalence or replacing the official implementation. Baseline LoRA merging
must receive the same opportunity in any optimized-deployment comparison.

The independent candidate execution also needs a dedicated serving path to
reduce launch/dispatch overhead and repeated state materialization. Cached maps
alone cannot remove the two decoder stages. CUDA graphs or compiled/bucketed
execution may help, subject to dynamic native K/length and numerical validation.
No claim of matching or beating Kev latency before measuring such a path.

## Trajectory

- 2026-11-06: inspected model and cached-prefix implementations and existing
  150-row timing evidence. Ran the six-row stage diagnosis to completion on
  otherwise free GPU 0. Both workers complete; 36 timed requests per model,
  with exact prediction identity across instrumentation. Source and checkpoint
  hashes recorded. Production/model files and official results remain unchanged.
