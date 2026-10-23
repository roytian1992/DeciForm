# Training and data protocol

## A labeled choice record

~~~json
{
  "id": "billing-001",
  "state": {"text": "I was charged twice for one order."},
  "question": {
    "type": "choice",
    "instructions": "Choose the support category.",
    "options": ["billing", "technical"],
    "option_ids": ["billing", "technical"],
    "option_definitions": ["Payments and invoices", "Technical faults"]
  },
  "label": "billing"
}
~~~

IDs must be unique. The label identifies a stable option ID; definitions carry
the semantics. Yes/no tasks use two explicitly defined options. This release
does not provide a separately trained score/regression head.

## Exact computation

For each candidate, the prompt is:

~~~text
State: {state.text}
Question: {question.instructions}
Candidate: {option_definition}
~~~

The final non-padding hidden vector is converted to FP32, layer-normalized and
passed through a scalar linear head. Cross-entropy compares candidate scores
within an example. Training includes gold plus up to two negatives selected by
a deterministic ID/seed hash. Softmax over all scores divided by the saved
temperature gives inference probabilities.

## Data preparation

prepare_data.py streams source JSONL, writes a byte-offset index, and records
input/output hashes. IDs and normalized text matching explicitly supplied
--exclude-jsonl files are excluded, followed by cross-source deduplication.
Audits distinguish raw, excluded, duplicate and retained counts. Historical
examples cannot be inferred from a checkpoint: supply reserved/history JSONL.

Normalization uses Unicode casefold and whitespace collapse of state.text.
This is exact normalized-text matching, not semantic near-duplicate detection.
No 1,024-token cap or silent truncation is introduced. Very long prompts may
still exhaust memory; a failure preserves the last saved checkpoint.

## Trainer export

scripts/train.py is a scoped extraction of the production trainer. Loss,
candidate sampling, shuffling, tail-row weighting, microbatch packing, DDP
accumulation, AdamW updates, clipping and checkpoint format are preserved.
Imports and paths now resolve inside this repository. Adapter evolution,
structural branches and recursive controllers remain outside this package.

New supervised data stages may choose their documented configuration; the
RSI-specific requirement that all parent settings stay identical was removed.
Exact resume remains strict about code, configuration, data plan and workers.
Explicit runtime migrations are recorded separately.

The release contains weights, hashes, completed-stage progress and metadata.
The original optimizer/RNG file is not needed to train a new data stage and is
not included. Standalone training checkpoints save optimizer/RNG for resumption.

## Historical coverage

The model inherits a 736-example initialization, then:

1. Static full: 4,866,714 sample visits.
2. Unused-example delta: 915,569 sample visits.
3. Four-source continuation: 21,575 visits (Banking77 7,957; SST-5 8,054;
   TREC 4,964; F2 priority 600).

The three later stages total 5,803,858 sample visits, not a claim of globally
distinct records across initialization history. Per-source counts and historical
configurations are in provenance/training_stages.json. Reconstructing the
original corpus additionally requires separately managed datasets and split
manifests.
