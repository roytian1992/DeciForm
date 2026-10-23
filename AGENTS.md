# DeciForm ownership

DeciForm owns the decision model architecture, supervised training, optimizer
and LoRA implementations, inference, calibration, data preparation and model
benchmark comparisons. Implement and test model changes here.

JEV_RSI consumes this package and explicit checkpoint manifests. Keep recursive
controllers, self-edit selection, rewards, episode orchestration and reset-transfer
experiments in JEV_RSI. Do not import jev_rsi from the active deciform package.

Preserve released weights and historical experiment artifacts. New training
runs use new output directories and record their parent checkpoint and hashes.
Historical source files under legacy/ are provenance, not supported launchers.
