# Laya mechanism transfer — 2026-11-06

User authorizes migrating A1–A3 mechanisms to Laya. Preserve all existing D2,
Laya and Score experiments. Apply experiment-hygiene tracking and verification.

## Frozen experiment contract

Sequential+mechanisms on 4–5 and parallel+mechanisms on 6–7, each compared with
its completed same-layout baseline in runs/formal_laya_20261106_v1. Use fresh
original ModernBERT-large and random Laya head, not the trained baseline.
Same frozen 200k training rows/negative IDs, seed20261101, epoch1, batch64,
3125 updates, original full-text/native dev6000/test15000, seven-order
Choice/Noul audit. Preserve original RLCD loss, full encoder fine-tuning,
encoder/head LR and cosine schedule. New parameters use the existing head LR.

## Transfer design

Task routing adds B(I+R_t)A to the existing trainable W at attention Wqkv/Wo
and MLP Wi/Wo, rank16/alpha32, shared A/B plus centered norm-bounded task maps.
This preserves full fine-tuning and adds shared low-rank capacity; it is not a
frozen-encoder LoRA experiment. B starts at zero. Input layout, attention masks
and the original Laya scorer are preserved.

The head adds query/candidate bilinear evidence with centered, bounded task
geometry and Choice-only query-relative quadratic residual. Query is the
existing head CLS representation, not a new candidate-independent encoding.
Candidate projection starts at zero, query projection random; the quadratic
output starts at zero. Therefore initial logits equal the original Laya, while
both new branches can receive training gradients. No new task loss or explicit
regularization loss is added; existing AdamW and bounded parameterization apply.

All added initialization is under a restored RNG context. Hash-check the
original encoder/head against baseline initialization. Module additions do not
change initial input-output behavior. Trainables/cost are recorded separately.
The baseline-to-full difference includes added shared bilinear and low-rank
capacity; it cannot isolate task conditioning without additional ablations.

## Checks and reporting

CPU checks cover exact initial outputs, preserved baseline RNG, all-task and
component gradients, routing constraints, variable K, checkpoint recomputation,
semantic permutation and checkpoint reload. Actual two-GPU preflight checks
longest full-text backward, first full batch, native maximum K and input lengths,
original initialization hashes and initially identical predictions.

Score uses native median MAE; Choice/Noul accuracy. Every result, including
negative results, is reported. Single seed and two layouts do not establish
cross-seed stability. Module migration is a hypothesis, not a guaranteed gain.
No formal-model changes or new training data. No text or candidate truncation.

Status: both v2 arms completed training, full evaluation and position tests.
Exact commands and source/config hashes are frozen in the manifests below.

## Mathematical parameterization

For projection family j and task t, the original trainable projection is retained:

$$
h'=W_jh+\frac{32}{16}B_j(I+R_{jt})A_jh.
$$

The Laya task order is choice/score/noul, and weights are 0.25/0.375/0.375.
For raw matrices U, set a weighted-centered map with a shared radial bound:

$$
\widetilde U_t=U_t-\sum_s\pi_sU_s,\qquad
R_t=\frac{0.5\widetilde U_t}{\max(1,\max_s\|\widetilde U_s\|_F)}.
$$

Thus the weighted mean is zero and Frobenius norm is at most 0.5. This is the
same centering/bounding construction as the source mechanism. Each projection
family has its own maps; different families do not compete in normalization.

Let q be a learned projection of the head CLS hidden state, k_i a learned
projection of candidate marker i, and b_i the original Laya scorer. Then

$$
s_i=b_i+\frac{q^\top(I+H_t)k_i}{\sqrt{256}}
+\mathbf1[t=\mathrm{choice}]\,4\tanh\left(\frac{a^\top(z_i\odot z_i)}{4\sqrt{64}}\right),
$$

$$
z_i=\mathrm{LN}\big(P\,\mathrm{LN}(h_i)-V\,\mathrm{LN}((I+H_t)^\top q)\big).
$$

H uses the same weighted centering and 0.5 radius. B, candidate projection k,
and QF output a are initially zero; A, q, P and V have ordinary random
initialization. The combined output initially equals b_i exactly. Once B and k
move away from zero, their corresponding task matrices can receive gradients.
Choice QF has no direct Score/Noul contribution, but all tasks share the encoder.

The CLS query still reads the original Laya shared context and may depend on
other candidates. No claim of candidate independence is introduced here.

## Exact run, code and environment

- Project: /vepfs-mlp2/c20250513/241404044/users/roytian/DeciForm.
- Current run: runs/formal_laya_mechanisms_20261106_v2.
- Python: /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/qwen35-train/bin/python.
- Original inputs/cache/encoder/upstream 0.4.26: runs/formal_laya_20261106_v1.
- Modules: scripts/formal/laya_mechanisms.py.
- Worker: scripts/formal/worker_laya_mechanisms.py; original worker preserved.
- Prepare: scripts/formal/prepare_laya_mechanisms.py.
- Supervisor: original scripts/formal/supervise_laya.py.
- Queue: scripts/formal/schedule_research_queue.py --root runs/formal_laya_mechanisms_20261106_v2 --gpus 4,5 or 6,7.
- Report: scripts/formal/report_laya_mechanisms.py --root runs/formal_laya_mechanisms_20261106_v2 --watch.
- Resume: exact per-arm experiment_manifest.json resume_command, only the arm's own resume.pt.
- Standalone inference: laya_mechanisms.load_transfer_checkpoint reads transfer_config.json and model.safetensors; the unmodified upstream loader cannot construct the added modules.

Total parameters are 429,379,395; trainable 429,115,457; added 8,085,568
(about 1.92% of the baseline total). Added shared A/B: 7,196,672; task routing:
86,016; shared bilinear+head geometry: 720,896; Choice QF: 81,984.
Original encoder parameters use LR2.5e-5, existing head and all added parameters
LR1e-4. Existing AdamW decay0.01, clip1 and cosine minimum1e-6 retained. All
default steps/batches/packing and RLCD noise settings match the original runs.

Every root/arm manifest pins original code/cache/config/initialization hashes
and contains a source snapshot. Exact tmux launch commands and wall time are
in launch_proof.json. Baseline initialized encoder and head hashes are checked
before installation and again afterward; both ranks must match all parameters.

## Trajectory

### 2026-11-06 13:50 Asia/Shanghai — implementation and preflight launch

Implemented all three mechanisms as additive, initially neutral changes to
Laya. Fifteen CPU tests passed, including variable 2/7/77 candidate support,
initial logits equality, RNG preservation, centered maps, live component
gradients, real encoder checkpointing, parallel semantic permutation and the
actual saved checkpoint loader. Both 2-GPU runs were launched in persistent tmux.

The first preflight (v1) failed before any optimizer update because its audit
attempted .cuda() on non-tensor batch metadata. The audit now filters tensors,
as the existing training path does. Failure records remain in v1; v2 re-freezes
the fix and restarts from original Base, with no resume/inherited parameters.

Current GPU ownership: sequential transfer 4–5, parallel transfer 6–7. Their
preceding continuous Score jobs completed before allocation. ASAP Score is
independent and was allowed to finish on 0–3. Each new run automatically performs
Dev6000/Test15000 and complete seven-order Choice/Noul10000 audit after training.
Current master document section 13 contains placeholders until original reports
exist. No remote backup requested or created.

### 2026-11-06 13:54 Asia/Shanghai — real training verified

Both v2 arms passed actual two-GPU preflight. All three task logits and act
outputs exactly match original Laya at initialization (max difference 0).
Longest training backward, native largest-K evaluation, complete longest test
input, first global batch and permutation checks passed. Preflight peak memory
was 5.053/5.062 GiB. Original encoder/head hashes match each completed baseline.
Both entered formal training and passed step10/3125 with finite loss/gradients.
First ten steps averaged 2.111/2.082 seconds; approximately 1.8 hours training
remains, excluding checkpoint I/O and automatic evaluation. This is an early
estimate, slower than the original Laya ~1.5 seconds/step. Frozen source hashes
were rechecked. training_started_verified.json stores observed progress.
Master record section13 and both tmux supervisors/report watcher are active.

### 2026-11-06 — completed results and migration audit

Both arms exited successfully. Parallel transfer Choice/Noul improved from
0.6918/0.8054 to 0.6960/0.8106, while Score median MAE worsened from 0.4992
to 0.5046. Sequential Choice/Noul changed from 0.6848/0.8066 to 0.6758/0.8126,
and Score MAE from 0.5024 to 0.5062. Neither transfer wins all three tasks.
Training took 6101.915 seconds parallel and 6197.215 seconds sequential.
Current original result paths remain in master section13 and results_summary.json.

User subsequently authorized a migration correction after read-only Dev
diagnostics. The new arm uses native scorer geometry and task-only encoder
contrasts with dedicated module LR and structural regularization. It is a fresh
parallel run, not a resumed version of these checkpoints. Evidence, math and
limitations: [native-path transfer](experiment_log_20261106_laya_native_transfer.md).
Preserve all previous negative results and immutable experiment artifacts.
