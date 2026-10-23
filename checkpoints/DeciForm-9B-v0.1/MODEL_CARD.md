---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
  - structured-decision-making
  - lora
  - text-classification
---

# DeciForm-9B-v0.1

DeciForm-9B-v0.1 scores explicit candidate definitions conditioned on a state
and question. It returns the selected candidate and a probability distribution.
The release contains the trained LoRA adapter and scalar decision head, used
with the separately downloaded **Qwen/Qwen3.5-9B post-trained/instruct** backbone.

## Model interface

Each candidate uses the prompt:

~~~text
State: {state.text}
Question: {question.instructions}
Candidate: {candidate_definition}
~~~

The final non-padding token representation passes through parameter-free
LayerNorm and a trained scalar linear projection. Candidate scores are divided
by the saved temperature, 1.113263487815857, and normalized with softmax.
Candidate IDs are bookkeeping keys; their positions are not included in the
model prompt. Candidate permutations therefore preserve the mathematical
scoring function. Floating-point batching and exact ties can affect numerical
comparisons and tie-breaking.

This release supports choice questions, including binary choices. Training uses
one labeled positive and up to two sampled negatives; evaluation scores every
candidate. The implementation preserves the full candidate prompt.

## Weights and configuration

| Component | Value |
|---|---|
| Backbone | Qwen/Qwen3.5-9B, frozen |
| LoRA | rank 16, alpha 32, dropout 0, q_proj and v_proj |
| Head | LayerNorm + Linear(4096, 1) |
| Trainable parameters | 2,232,321 |
| Adapter file | 8,917,464 bytes |
| Head file | 18,885 bytes |
| Training precision | BF16 backbone, FP32 head |
| Reported evaluation precision | FP32 |

The adapter and head are byte-identical to the evaluated source checkpoint
joint_static_four_source_4gpu_v1_seed20261015/checkpoints/step_00000169.
Only the adapter configuration's local backbone path was replaced with the
portable model ID. See [release.json](release.json), [hashes.json](hashes.json)
and [base_model.json](base_model.json).

The release supports inference and further supervised training. To continue,
initialize a new data stage from these weights. Optimizer and RNG state from
the original training run are not included; checkpoints produced by the
provided trainer include them for resumable local runs.

## Training history

| Stage | Sample visits | Optimizer steps |
|---|---:|---:|
| Static full-data stage | 4,866,714 | 38,022 |
| Previously unused-data continuation | 915,569 | 7,153 |
| Banking77, SST-5, TREC and F2 priority continuation | 21,575 | 169 |

These stages total **5,803,858 sample visits**. An earlier 736-example pool was
used for head and adapter initialization. Sample visits are not a count of
globally unique texts. Per-source counts, configuration and report hashes are
recorded in [training_stages.json](../../provenance/training_stages.json).

## Evaluation

On the current local held-out set of 68,156 records from 12 sources, the model
achieves 80.56% source-equal mean accuracy and 98.27% record-weighted accuracy.
DBpedia14 accounts for 66,012 records. The
[per-source table](../../results/README.md) includes Kev-9B and Open-Jev-9B
evaluated on the same records, with their separate training histories.

The release publishes aggregate results and provenance; raw dataset records
remain in the separately managed data collection. The evaluation tools accept
user-provided choice JSONL and record training/evaluation overlap checks.

## Usage

Use DeciFormModel.from_pretrained(base_model_directory, checkpoint_directory)
from deciform.model, or the inference, training and evaluation commands in
the [repository README](../../README.md). The custom decision head must be
loaded together with the PEFT adapter.

## License

Project code and trained adapter/head weights are released under Apache-2.0.
The separately downloaded Qwen backbone is governed by its own Apache-2.0
license. Dataset terms remain with the respective providers.
