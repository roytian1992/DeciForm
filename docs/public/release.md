# Public release preparation — 2026-11-17

## Implemented

- A standalone deciform.core package with Qwen and full-update ModernBERT backends; no research-run or sibling-project imports.
- Candidate isolation, shared-A adapters, output-scale, public-task-type private adaptation, log+spherical classification and decimal-prefix CE/CDF scoring.
- Explicit raw-file conversion, pinned Hugging Face intake, JSONL validation, split checks, deterministic frozen Choice/Union sampling and MIND-small TSV conversion.
- Fresh distributed training, atomic pause/resume checkpoints, explicit continuation, portable safetensors exports, prediction, full-candidate metrics and permutation auditing.
- Public configurations, synthetic examples, source provenance, CPU tests, and an allowlisted source-export script.

README contains usage and architecture only, with no experimental results.

## Verification

The local test environment uses Python 3.10, PyTorch 2.11.0+cu128 and Transformers 5.20.2. The final isolated source run passed 17 tests; one redundant Qwen distributed case is intentionally skipped because the same aggregation routine is exercised with ModernBERT. CPU coverage includes all three task types, both backbone families, candidate-content isolation, exact permutation alignment, correct loss gradients, source conversion, split checks, MIND multi-positive targets, export/reload, deterministic resume, and two-process gradient aggregation with uneven task allocation.

Full-size local Qwen3.5-9B-Base and ModernBERT-large checks each executed one gradient update over all three tasks and exactly reproduced their predictive distributions after export/reload. These are integration checks, not benchmark or throughput measurements. Qwen smoke verification used unwrapped PyTorch fallback kernels. Historical encoder comparisons matched token IDs and layouts on nine mixed-task records per backbone.

A portable verification receipt is in provenance/public_verification.json. Historical result records remain in the research workspace; they are not inserted into this README.

## Remaining reproduction packaging

The public runtime is operational. These items remain before claiming a complete reproduction of all paper experiments:

1. Pin and package every raw-source download recipe and original population membership/grouping step. Current historical 100k/200k manifests identify frozen files but are not an end-to-end downloader for those exact files.
2. Package baseline launchers/configurations for the pinned Kev and Laya versions, plus optional generative comparisons.
3. Add explicit converters for historical research checkpoints, whose wrappers and key names differ from the new public export format.
4. Release the selected trained weights and finalized paper-specific experiment recipes separately.

The current GPU smoke fixtures do not constitute a new research training run. No frozen result is relabeled as a public-recipe result.

## Export a clean source tree

From the research checkout, choose a new destination outside that checkout:

~~~bash
python scripts/public/export_source.py --output /path/to/new/DeciForm-public
~~~

The allowlist copies only public code, examples, configs, documentation, tests, licenses and provenance. It excludes raw data, run artifacts, model checkpoints, operational AGENTS records and old research workers. SOURCE_MANIFEST.json hashes every included file. Review that tree before publishing; exporting does not commit, push or change repository visibility.


## Active-tree cleanup

The superseded modules, research launchers, tests, configs and v0.1 weights have been removed from the active source tree. Historical experiment documentation and results were archived locally with file hashes rather than discarded. Data and current run/checkpoint directories retain their existing locations. The public installation discovers only the current package; default test discovery runs tests/public.

This is a working-tree cleanup. It does not erase previous commits or remove already published files from remote Git history.
