"""Matched CE/CDF prefix pilot with optional numeric-sequence supervision.

Both arms share the same causal forward and packing, including suffix length.
The control multiplies auxiliary token loss by zero. Its scoring representation
precedes all target digits; actual-backbone counterfactual checks are mandatory.
This is a small-sample replay of existing N2/T2 mechanisms, not a new method.
"""
from dataclasses import asdict, dataclass

from models_numeric_trie_consistency import FormalModel as JointModel
from numeric_trie import TrieObjective


@dataclass(frozen=True)
class PilotObjective:
    kind: str = 'pilot_trie_ce_cdf'
    ordinal_scale: float = 3.0383850790577127
    cdf_weight: float = .5
    token_weight: float = 0.

    def __post_init__(self):
        if self.kind != 'pilot_trie_ce_cdf' or self.token_weight not in (0., .5):
            raise ValueError('Pilot allows only matched token weights 0 or 0.5')
        self.distribution()

    def distribution(self):
        return TrieObjective('trie_ce_cdf', self.ordinal_scale, self.cdf_weight)

    def inventory(self):
        return asdict(self)


class FormalModel(JointModel):
    def __init__(self, *args, distribution_objective=None, **kwargs):
        obj = PilotObjective(**distribution_objective)
        super().__init__(*args, distribution_objective={
            'kind': 'trie_token_ce_cdf', 'ordinal_scale': obj.ordinal_scale,
            'cdf_weight': obj.cdf_weight, 'token_weight': .5}, **kwargs)
        self.distribution_objective = obj
        self.ablation_inventory.update(
            score_research_architecture='small_numeric_trie_paired',
            token_weight=obj.token_weight,
            matched_teacher_forced_forward=True)
