# DeciForm repository guidance

## Current public implementation

- Public runtime: src/deciform/core. Public configurations, examples and tests live in their public subdirectories.
- CLI: python -m deciform.core. Run python -m pytest -q for the public test suite.
- README documents installation, interfaces and usage; do not add experimental results unless requested.
- Historical implementations and uncommitted research changes are archived under legacy/research_20261117_before_cleanup. Consult cleanup_manifest.json for original paths and hashes. Do not restore old interfaces to the package automatically.
- Data and experiment runs remain at data/ and runs/; they are excluded from the public source export. The existing Git history has not been rewritten.

## Model and evaluation contracts

- Support explicit Choice, Noul and Score task types. Route only by those public types, never sample, user or source IDs.
- Qwen freezes its backbone; the current ModernBERT recipe jointly trains backbone and adapters.
- Keep the pretrained backbone, adaptation configuration, initialization and actual parent checkpoint explicit.
- Candidate sampling controls training cost only. Dev and Test always use the complete native candidate or numerical support.
- Do not silently truncate text, remove candidates or drop failed examples. Align permutation outputs by stable semantic identity.
- Preserve original numerical labels; Score reports MAE. Preserve all clicks for multi-positive MIND ranking.
- Keep architecture comparisons' data and sampling matched. Record code/config/data hashes and restore complete optimizer/scheduler/RNG state for exact resume.
- Preserve historical experiment evidence; a revised public recipe is not automatically a reproduction of a historical result.

## Local operations and releases

- scripts/research_reset/gpu_activity_guard.py is a retained local service dependency. Do not move it or interrupt active GPU guards/BGE when cleaning public code.
- Local GPU state and archive recovery details are in legacy/research_20261117_before_cleanup/OPERATIONS_CURRENT.md.
- Build public exports with scripts/public/export_source.py. Check that raw data, weights, runs, archived code and secrets are excluded.
- Do not commit unrelated work, push or change remote visibility without user authorization.
