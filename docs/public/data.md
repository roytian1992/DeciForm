# Data preparation

## Normalized JSONL

Common fields: id (unique within a split), task, state (text or {text: ...}), question.instructions, and label. question.type may repeat task but must agree. Optional source, source_split, group_id, and provenance remain audit metadata.

- Choice: question.options contains complete descriptions. Optional question.option_ids contains stable identities; label refers to an identity, or to the option string when IDs are absent. Optional option_definitions overrides descriptive text. IDs must be unique. Do not renumber IDs after changing order.
- Noul: label is a JSON Boolean; candidate_values defaults to [false, true]. Both values are required.
- Score: provide score_support as a strictly increasing list, or score_range={min,max,step}. The grid must include its endpoint. label is the raw finite numeric target within the declared range. label_kind is native_integer or continuous; set continuous explicitly for fractional supervision even on an integer decoding grid.
- Multi-positive Choice: target_distribution aligns with the supplied candidates and sums to one. Ordinary Choice continues to use one label. Ranking and single-label panels are evaluated separately.

Evaluation files contain complete native candidates. Training files produced by prepare include explicit training_view metadata and cannot be evaluated as a native panel. No automatic prompt truncation or long-row dropping is performed. A configured max_input_tokens is an admission check that raises an error; it never cuts input text.

## Local raw conversion

~~~bash
python -m deciform.core convert \
  --recipe examples/public/source_recipe.json \
  --input examples/public/raw.jsonl --output data/raw_example.jsonl
~~~

Recipes explicitly name task, label_field, text_template, instructions_template, source, and split. Dotted field paths are supported for labels and IDs. Templates use named Python format fields. Choice may supply fixed options or options_field and a fixed/native option identity list. label_is_index explicitly converts source indices into identities. Noul string labels require a label_mapping: no truthiness inference is performed. Score declares its full support/range.

Each converted file has a companion manifest with the recipe, source checksum, output checksum, counts, and zero truncation/drop counters. JSONL and gzip JSONL, CSV/TSV, and Parquet are supported. Invalid records stop conversion with their source row; partial outputs are not published as complete.

## Pinned Hugging Face intake

A recipe can add huggingface={path,name,split,revision}. revision must be a full 40-character commit hash. Install the data extra, then run:

~~~bash
python -m deciform.core fetch --recipe /path/to/source_recipe.json --output data/source.train.jsonl
~~~

The adapter records the dataset revision and loader fingerprint. Obtain each source under its original access and redistribution terms. Download helpers do not make source licenses interchangeable.

## Split and candidate freezing

prepare preserves explicit train/dev/test inputs. It rejects duplicate IDs within each split, exact input overlap across splits (including reordered candidates), and overlaps of supplied group_id within a source family. For near-duplicate grouping, supply the source's group identifiers before preparation; this command does not perform approximate text deduplication.

Sampling is deterministic from seed, row identity, and epoch=0. Each original Choice retains all positives and up to negative_count*views distinct negatives; it is written once. Noul and Score are unchanged. Training consumes the frozen file without hidden resampling. All models in a comparison should use that same file.

## MIND

Use the official labeled MIND-small train/dev news.tsv and behaviors.tsv files:

~~~bash
python -m deciform.core prepare-mind \
  --news /path/to/MINDsmall_train/news.tsv \
  --behaviors /path/to/MINDsmall_train/behaviors.tsv \
  --split train --output data/mind_train --negative-count 3
~~~

Repeat with --split dev and the official Dev files. Every reading-history article and native impression candidate is retained in the full file. All clicked candidates receive target mass; the training view samples only non-clicks. User IDs are not model input. The public ModernBERT path uses stable news IDs as candidate names; older MIND workers used display aliases. The public conversion is not presented as byte-identical historical MIND preprocessing. Official Dev is labeled as Dev, not relabeled as the unavailable official Test. Missing news or invalid impressions cause an explicit failure.

## Historical populations

provenance/data/mixed_100k.json and mixed_200k.json publish counts and checksums for the existing frozen populations. Their training task proportions are Choice/Noul/Score = 35%/30%/35%; both use the same 6,000 Dev and 15,000 in-domain Test rows. The main and specialist source lists are in configs/public/data_populations.json.

These manifests identify the historical files. They do not claim that the generic conversion commands reproduce those bytes: complete source-specific conversion, original membership selection, and historical near-duplicate grouping still need release packaging. The research checkout retains those original files. Raw research data is not copied into the public export.
