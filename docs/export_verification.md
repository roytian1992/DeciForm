# Standalone release verification

Date: 2026-10-23. Release: DeciForm-9B-v0.1.

## Export scope

The supervised trainer, candidate scorer, data indexing, runtime settings,
probability metrics and locked-test evaluator were extracted into the deciform
package. All runtime imports resolve inside this repository or its declared
Python dependencies. The source-to-export mapping and checksums are in
[source_export.json](../provenance/source_export.json).

The published adapter and decision-head files are byte-identical to the
latest evaluated source checkpoint. Their combined size is 8,936,349 bytes.
Backbone weights remain separately downloadable. Exported metadata records
portable model identifiers and relative training-stage identifiers.

## Checks performed

| Check | Result |
|---|---|
| Python compilation | Passed for src, scripts and tests |
| Unit tests | 19 passed |
| Python wheel build | deciform-0.1.0-py3-none-any.whl built successfully |
| Release checksums | Adapter, head, configuration and progress verified |
| FP32 GPU inference | Same predictions as source evaluation on four representative records |
| Option-order diagnostic | Maximum probability difference 0.000000656 |
| BF16 distributed training | Two GPU workers completed a nine-record synthetic run |
| Pause/resume | Saved at step 1, resumed to step 3 |
| Tail-batch coverage | All nine records visited once, including final single record |
| Weight update | Both adapter and head changed in the isolated training check |
| Release preservation | Published adapter/head hashes unchanged after checks |

The inference records cover Banking77 (77 candidates), F2 priority (3), SST-5
(5) and TREC (6). The largest probability difference from the original saved
evaluation is 0.000266791. This compares export behavior on four examples;
the complete benchmark results remain those of the previously evaluated
checkpoint. See [inference_verification.json](../provenance/inference_verification.json).

The training check used global batch 4, physical batch 1 per worker, BF16,
gradient checkpointing and the published initialization. It paused after four
sample visits and resumed to nine visits across three optimizer steps. Final
loss was 0.00185116. Peak allocated memory was 17.96 GiB on the recorded worker.
See [training_verification.json](../provenance/training_verification.json).

Both GPU checks used the standard Transformers reference kernels without
the original isolated FLA runtime, and did not import sibling-project code.
The training check tests execution and coverage, not learning quality or
large-corpus throughput. Synthetic smoke checkpoints remain local; the
published model is the original fully trained checkpoint.

## Commands

From the repository root, with the tested environment:

~~~bash
python -m compileall -q src scripts tests
python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/deciform-wheel .
~~~

The training smoke run used the same prepare_data.py and train.py commands
documented in the README, a nine-row synthetic choice dataset, global batch 4,
physical rows per GPU 1, and two workers. The first invocation set
--stop-after-steps 1. The second used --resume with step_00000001 and completed
step_00000003. Production training data and release weights were not modified.

## Environment

Python 3.10, PyTorch 2.11.0+cu128, Transformers 5.20.2, PEFT 0.21.2,
NumPy 2.2.6, safetensors 0.8.0 and accelerate 1.15.0. Exact package versions
are listed in [requirements-tested.txt](../requirements-tested.txt).
