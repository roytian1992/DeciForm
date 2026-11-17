# Third-party attribution

## Kev compatibility definitions

The public package includes adapted definitions from the receptron/Kev project:

- core/integrations/kev_prefix.py: Prefix and _masks from upstream shared_prefix.py.
- core/integrations/kev_encoding.py: user_tokens and PointerHead from upstream model.py.

These retain their Apache-2.0 licensing. The upstream license is included at src/deciform/core/integrations/LICENSE.kev and in built distributions. Source snapshot and destination hashes are recorded in provenance/public_components.json. Modifications isolate these definitions from the historical runtime, replace imports, and provide DeciForm-owned public orchestration.

## Backbones, data and research comparisons

Qwen and ModernBERT are downloaded separately. Their weights, tokenizers and configuration files remain governed by their upstream terms. Laya and Kev baseline research code is not bundled by the public export script. The ModernBERT public encoder/collator uses DeciForm's own directed masks, encoding and readout integration.

Dataset source code, content, and access conditions are separate from this repository's code license. The public export contains synthetic format examples and provenance manifests, not third-party raw corpora or historical model checkpoints.
