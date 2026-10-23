# Fixed-set evaluation

scripts/evaluate.py uses the original FP32 backbone path, all candidates,
saved temperature, full text and exact-length batching. Labels do not enter
model prompts. Workers check option reversal on the first example of each source.

Prepare a locked inventory and train/test isolation audit. Supply every training
JSONL the audit should check; unlisted historical sources are not audited.

~~~bash
python scripts/prepare_evaluation.py \
  --source support=/path/to/test.jsonl \
  --training-jsonl /path/to/train.jsonl \
  --output data/evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc_per_node=4 scripts/evaluate.py \
  --model-dir /path/to/Qwen3.5-9B \
  --checkpoint checkpoints/DeciForm-9B-v0.1 \
  --manifest data/evaluation/manifest.json \
  --audit data/evaluation/audit.json \
  --output-dir runs/evaluation --workers 4

python scripts/evaluate.py \
  --model-dir /path/to/Qwen3.5-9B \
  --checkpoint checkpoints/DeciForm-9B-v0.1 \
  --manifest data/evaluation/manifest.json \
  --audit data/evaluation/audit.json \
  --output-dir runs/evaluation --workers 4 --aggregate-only
~~~

Use a fresh prediction output directory. Aggregation checks complete coverage,
unique IDs, checkpoint identity, source and labels. Reports contain accuracy,
macro-F1, Brier score, 10-bin ECE, NLL and fixed-threshold selective risk.
Test labels score predictions; they do not fit temperature.

Published results use the original frozen local inventory hash in
results/comparison_20261023.json. A new inventory has a new benchmark identity
and does not reproduce the original split IDs.
