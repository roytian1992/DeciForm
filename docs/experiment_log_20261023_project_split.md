# Project split — 2026-10-23

## Scope and status

Separate standalone decision-model development (DeciForm) from recursive
self-improvement experiments (JEV_RSI). Status: code migration and validation
complete. No corpus preparation, production training or benchmark rerun was
started by this change.

## Inputs and environment

- Starting DeciForm commit: 0892e9541e5e324b1dbe8c4120da7fddd785988c, tag v0.1.0.
- Published initialization: checkpoints/DeciForm-9B-v0.1; tensors unchanged.
- Existing JEV_RSI source, staged index and working changes were snapshotted.
- Environment: qwen35-train, Python 3.10; existing dependency versions retained.
- Editable install: python -m pip install --no-deps --no-build-isolation
  -e ./DeciForm -e ./JEV_RSI from the shared workspace.

## Changes

DeciForm owns model modules, standard and structured training primitives,
episode adaptation, inference, data preparation, probability metrics and native
baseline evaluation. RSI orchestration imports these implementations.
Compatibility modules only re-export; model-only historical launchers/configs
are archived locally under DeciForm/legacy/jev_rsi. Existing runs, dataset paths,
logs and checkpoint releases remain intact.

JEV_RSI keeps controller/self-edit learning, task/verifier code, reward/budget
accounting, candidate promotion and rollback, and experiment-specific control
arms. Both README files and AGENTS.md files define this ownership.

The discovery selector uses DeciFormModel. The existing discovery_v1 config
retains its original checkpoint. A separate discovery_deciform_v0.1 config pins
the released model manifest and checksums, so model selection is explicit.

## Verification

- DeciForm tests: 64 passed (63-test full pass plus the added runtime-path
  regression, with the affected 16-test subset rerun).
- JEV_RSI tests: 64 passed.
- 34 model-tool --help invocations completed; all source compiled.
- Six AST comparisons confirm unchanged model functions/classes and loss.
- Real GPU selector parity: identical selection and probabilities (maximum
  absolute difference 0.0); prefill count 193 and matmul precision preserved.
- Model-free RSI protocol and task-reference checks completed separately.
- Published tensor/config checksums and original RSI staged index unchanged.

Reference checks are execution tests, not additional research results. No
complete RSI v2 learning loop is claimed by this refactor.

## Outputs and recovery

The local migration snapshot is DeciForm/runs/project_split_20261023_152900.
It contains before/ source copies, original staged/working patches, migration
manifest, GPU parity and CLI/AST verification records. The public provenance
file is provenance/project_split_20261023.json. v0.1.0 preserves the initial
release; this code boundary is package version 0.1.1.

RSI smoke artifacts: JEV_RSI/runs/project_split_protocol_20261023 and
JEV_RSI/runs/project_split_reference_20261023. These are isolated new outputs.
Existing experiments must retain their original code manifests for exact
resume; the refactor does not bypass code-hash checks.

The first RSI collection test exposed a moved audit_validation_plan.py import.
That audit belongs to RSI data-role enforcement and was restored there. A
baseline evaluator provenance path was corrected to deciform/evaluation.py.

## Trajectory

2026-10-23: snapshot and classify → migrate model code → connect explicit package
and release contracts → verify both projects and GPU parity → record migration.
DeciForm changes are committed separately. JEV_RSI had an unborn repository
with pre-existing staged changes; its index is left unchanged and the split
is additionally preserved as a dedicated local patch/snapshot.
