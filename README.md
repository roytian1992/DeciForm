# DeciForm

**Open-weight models for structured decision making.**

DeciForm converts a state, a question and explicit candidate definitions into a
choice with probabilities. This repository contains the standalone supervised
LoRA trainer, inference interface, evaluation tools and the latest
**DeciForm-9B-v0.1** adapter and decision-head weights.

## Released model

| Component | DeciForm-9B-v0.1 |
|---|---|
| Backbone | Qwen/Qwen3.5-9B, post-trained/instruct variant |
| Adapter | Standard PEFT LoRA, rank 16, alpha 32, q_proj/v_proj, dropout 0 |
| Decision head | LayerNorm followed by a scalar linear projection |
| Trainable parameters | 2,232,321 |
| Training | BF16 backbone, FP32 head, AdamW, learning rate 2e-5 |
| Published evaluation | FP32 backbone; saved temperature 1.113263487815857 |
| Weight package | Approximately 8.94 MB: adapter plus choice head |

The weights are in [checkpoints/DeciForm-9B-v0.1](checkpoints/DeciForm-9B-v0.1).
Download the frozen backbone separately. Adapter and head tensor files are
byte-identical to the final evaluated checkpoint. A portable model ID replaces
the original machine path in adapter_config.json.

Current inference supports **choice** questions, including binary decisions
represented as two options. Each candidate is scored independently using the
same prompt and head; probabilities are normalized across candidates.
The prompt contains candidate definitions, without option-position letters.
Reordering candidates preserves each candidate's mathematical score;
floating-point batching differences and exact-tie selection still need care.

## Install

The tested environment is in [requirements-tested.txt](requirements-tested.txt).
Use a CUDA-compatible PyTorch installation. From the repository root:

~~~bash
conda env create -f environment.yml
conda activate deciform
~~~

Or install into an existing compatible environment:

~~~bash
python -m pip install -e '.[test]'
pytest -q
~~~

The original training environment used Python 3.10, PyTorch 2.11.0+cu128,
Transformers 5.20.2 and PEFT 0.21.2. It also loaded fla-core 0.5.2 through an
isolated runtime path. That optional runtime is not vendored here. Record the
kernel configuration when comparing speed or numerical output.

Download **Qwen/Qwen3.5-9B** from ModelScope into a local directory; use its
post-trained/instruct weights, not Qwen3.5-9B-Base. With the ModelScope SDK:

~~~python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B', local_dir='models/Qwen3.5-9B')
~~~

Expected config and tokenizer checksums are in
[base_model.json](checkpoints/DeciForm-9B-v0.1/base_model.json).

## Inference

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py \
  --base-model /path/to/Qwen3.5-9B \
  --checkpoint checkpoints/DeciForm-9B-v0.1 \
  --input examples/request.jsonl \
  --output /tmp/deciform-predictions.jsonl
~~~

Default inference uses FP32, matching reported evaluation. Optional --dtype bf16
changes numerical precision. Both preserve full text; GPU memory constrains
usable prompt length.

~~~python
import json
from deciform.model import DeciFormModel

model = DeciFormModel.from_pretrained(
    '/path/to/Qwen3.5-9B', 'checkpoints/DeciForm-9B-v0.1')
with open('examples/request.jsonl') as f:
    result = model.predict(json.loads(next(f)))
print(result['prediction'], result['probabilities'])
~~~

## LoRA training

Training uses JSONL with labels; inference does not need labels. Training scores
one positive and up to two deterministic sampled negatives per row. Evaluation
scores **all** offered candidates.

Prepare a data stage and supply reserved/history records to exclude:

~~~bash
python scripts/prepare_data.py \
  --source support=/path/to/new_support_train.jsonl \
  --exclude-jsonl /path/to/reserved_test.jsonl \
  --exclude-jsonl /path/to/previously_trained.jsonl \
  --base-model /path/to/Qwen3.5-9B \
  --parent checkpoints/DeciForm-9B-v0.1 \
  --output data/prepared

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config data/prepared/training_config.json \
  --output-dir runs/my_continuation \
  --start-from-completed checkpoints/DeciForm-9B-v0.1
~~~

Defaults: global batch 128, up to four physical rows per GPU, length-aware
microbatches and gradient checkpointing. A global permutation visits every
retained row, including the final incomplete batch.

Continuation inherits adapter/head weights and starts a new optimizer and data
cursor. A run created by this trainer can resume its optimizer, RNG and cursor:

~~~bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config data/prepared/training_config.json \
  --output-dir runs/my_continuation \
  --resume runs/my_continuation/checkpoints/step_00000064
~~~

See [training.md](docs/training.md) for data format and export details.

## Evaluation results

The current Static model and two released baselines evaluated the same 68,156
local held-out records from 12 sources:

| Model | Source-equal mean accuracy | Record-weighted accuracy |
|---|---:|---:|
| DeciForm-9B-v0.1 | 80.56% | 98.27% |
| Kev-9B | 78.92% | 98.03% |
| Open-Jev-9B | 71.94% | 93.48% |

See [per-source results](results/README.md) and machine-readable metrics.
DBpedia14 contributes 66,012 records; source-equal and pooled summaries answer
different questions. Models have different training histories, and external
training/test overlap has not been fully audited. This describes a local
engineering comparison, not an equal-budget method comparison.

Evaluation data is maintained separately. The evaluator accepts a locked
inventory and passing overlap audit; see [evaluation.md](docs/evaluation.md).

## Project boundary

DeciForm owns the model, supervised training, calibration metadata and evaluation
interfaces. Recursive self-improvement research can consume checkpoints through
these interfaces in a separate project. This release is a supervised Static model.

## Provenance and license

- [Model card](checkpoints/DeciForm-9B-v0.1/MODEL_CARD.md)
- [Training stages](provenance/training_stages.json)
- [Export verification](docs/export_verification.md)
- [Source migration map](provenance/source_export.json)

Code and project-trained adapter/head: Apache-2.0. The separately obtained Qwen
backbone has its own Apache-2.0 license. Dataset licenses remain with their
respective providers; raw training/test data is not bundled.
