# Candidate-independent D2 full, 2026-11-05

## Authorization and scope

The user approved retaining the full projection coverage and implementing a
candidate-independent model, then training on GPUs 4–7. GPUs 0–3 remain assigned
to the existing D2 size curves. Earlier models/results are preserved.

This run retains D2's 46,304,262 trainable parameters: all-target rank-16/alpha-32
LoRA, anchored task routing in rank space, task-conditioned bilinear geometry,
candidate-local Choice QF, and ordinal Score CDF readout. Choice CE, Noul BCE,
Score ordered-CDF objective (scale 3.0383850790577127), structural penalty and
native-median Score decoding are unchanged. No trained weights are inherited.

## Method

Let P=(state, question). The original D2 causal row encoded every candidate
before its final query token. This permits candidate-order leakage even though
the head is permutation equivariant conditional on fixed embeddings.

The new forward uses

\[
q=F_\theta(P,\texttt{decide}),\qquad
h_i=F_\theta(P,\texttt{opt}\ c_i\ \texttt{/opt}).
\]

Each candidate branch and the query branch starts from the same prefix attention
KV, DeltaNet convolution window, recurrent state, and actual prefix position.
The prefix is computed once per decoder layer. Functional state fan-out preserves
all branch gradients into the prefix. Half-layer checkpointing replays the
prefix/branches jointly; there are no detached inference caches.

The unchanged task-conditioned low-rank projection is
\(W x+(\alpha/r)B(I+R_t)Ax\). It remains in Q/K/V/O, gate/up/down, and DeltaNet
input/output projections. Only the current decoder layer's routing matrices are
materialized in each pass, with the same anchored formula and parameters.

Choice/Noul scores depend on the query and that candidate only; permutations
therefore permute scores in exact arithmetic. Softmax still normalizes across
the complete candidate support. The existing Score ordinal transformation
operates in native numeric-support order. Native semantic IDs are preserved;
permutations change actual branch execution order, not just output labels.
Exact classification ties use stable semantic IDs instead of display position.
Floating-point deviations are measured, not assumed absent. Choice/Noul reverse
audit runs after full native evaluation; Score flip experiments remain excluded.

Functional prefix-cache utilities are reused from the attributed vendored Kev
`shared_prefix.py`; prefix sharing itself is not a novelty claim. The work here
adapts it to independent option/query branches, task routing and the frozen D2
head. Training shares prefix dropout across candidate branches, so it is not
bitwise equivalent to repeated-prefix independent dropout during training.

## Inputs, execution and outputs

- Project/cwd: `/vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm`.
- Reference: `runs/formal_score_distribution_20261104_v1/d2_ordinal_main`.
- Data: `data/formal_200k_20261101_v1`; cache audit SHA256
  `cb186c69bf247ed0c6abf599902be5c0d9db355d5d7f2fcb421846724ef9ea2e`.
- Frozen populations: 200,000 train, 6,000 dev, 15,000 in-domain test.
- Identical train row order, candidate IDs, one view (at most three Choice
  negatives), seed 20261101, global batch 64, one epoch/3,125 updates.
- Evaluation always uses complete native candidates. No text truncation,
  candidate dropping, new sampling, or external-source evaluation.
- Fresh original base: `../RSI_Research/external/baseline_models/qwen35_9b_base`.
- Python: `../anaconda3/envs/qwen35-train/bin/python` (qwen35-train).
- Output: `runs/formal_independent_20261105_v1`.
- Preparation: `python scripts/formal/prepare_independent.py`.
- Launch: `python scripts/formal/supervise_score_specialist.py --manifest
  runs/formal_independent_20261105_v1/experiment_manifest.json` in tmux
  `deciform_independent_20261105`, CUDA_VISIBLE_DEVICES=4,5,6,7, world size 4.
- Recovery: same command with `--resume`, using its own optimizer/RNG checkpoint.
- Manifest freezes config, source snapshots/SHA256, environment, exact inputs,
  commands, original initial-head/LoRA/pointer hashes, and reference provenance.
- Progress/status, preflight/train logs, failure files and 15-minute stale alerts
  reside in the output directory. Checkpoints every 100 updates; automatic full
  native dev/test and Choice/Noul reversal follow the sole epoch endpoint.

## Verification and trajectory

2026-11-05, implementation: 64 focused tests passed, covering frozen D2 objective,
readout, ablations, and new shared-prefix output/all-trainable-gradient parity
against independent full-prefix computation. Tiny actual Qwen3.5 includes both
DeltaNet and full attention, nonzero LoRA/routing weights, varied prefix/option
lengths, mixed tasks, and half-checkpoint replay. Candidate permutation and
other-candidate content isolation also pass.

The first CPU attempt encountered installed CUDA-only convolution dispatch.
Tests now unwrap the official Transformers Torch reference kernels only inside
their monkeypatch fixture. Production GPU kernels and library files are unchanged.
The reference test also disables checkpointing in the deliberately slow reference
forward; checkpointed shared-prefix gradients still compare against it.

Status at code preparation: implementation/CPU tests complete; actual 9B GPU
zero-update preflight pending. No quality or speed improvement is claimed.
Local scoped source commit and per-run source snapshot are prepared; no remote
backup/push requested. Subsequent entries below record actual launch evidence.
