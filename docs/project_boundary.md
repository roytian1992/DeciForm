# DeciForm and RSI integration

DeciForm owns the standalone decision model: architecture, LoRA/head training,
optimizer and throughput changes, inference, calibration, data preparation and
model benchmark comparisons. Implement and validate model changes here.

JEV_RSI imports deciform and calls its training primitives. It owns self-edits,
learned controllers, reward/budget accounting, candidate search, lineage,
rollback and transfer protocols. RSI may update model parameters through these
primitives without maintaining a second trainer.

## Entry points

| Task | Entry point |
|---|---|
| New supervised continuation | scripts/prepare_data.py, scripts/train.py |
| Inference | scripts/predict.py, deciform.model.DeciFormModel |
| Fixed test evaluation | scripts/prepare_evaluation.py, scripts/evaluate.py |
| Native baseline evaluation | scripts/evaluate_unified_baseline.py |
| Existing model comparison report | scripts/report_full_model_comparison.py |
| Bounded episode training for an RSI caller | scripts/train_episode.py |
| Episode scoring | scripts/evaluate_episode.py |
| Registered structured-adapter training | scripts/train_structured.py |
| Adaptation feedback evaluation | scripts/evaluate_adaptation.py |

train.py remains the recommended standalone supervised trainer. The episode and
structured entrypoints preserve earlier research protocols, including explicit
budgets and registered data plans. The RSI caller chooses the candidate and
output directory; DeciForm executes the model operation.

The deciform adapter_program, adapter_transfer and latent_program modules own
adapter compilation, parameterization and checkpoint conversion. Deciding which
program to propose or promote belongs to the RSI controller.

## Data and historical runs

The existing raw datasets, baseline checkouts and completed experiment outputs
keep their original paths. Source-specific historical data tools accept their
existing manifests; generic prepare_data.py and prepare_evaluation.py provide
the public portable path. Local historical configs and launchers are archived
in legacy/jev_rsi and excluded from Git because they contain machine paths.
The local configs/local/current_comparison.json reads completed predictions and
writes a new report under DeciForm/runs.

Source-specific importers for Arrow/Parquet or saved dataset objects require
the optional data dependencies: python -m pip install -e '.[data]'.

Future standalone training outputs belong to DeciForm/runs; model releases go
to checkpoints; reported scores go to results. RSI candidate runs and controller
checkpoints belong to JEV_RSI/runs. Historical outputs are not relabeled as new
runs or resumed implicitly after code movement.

## Release contract

Consumers pin a release path, release manifest checksum, adapter/head hashes,
backbone identity, and code revision. Weights are immutable after publication.
A new training result gets a new release version. The v0.1.0 tag preserves the
original standalone export and its verification record.
