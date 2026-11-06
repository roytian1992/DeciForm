"""Full-support numeric trie plus exact-decimal token NLL in one causal pass.

The distribution reads only the state BEFORE the first answer token. Later
positions supply teacher-forced numeric-token NLL. Inference reads no labels.
The auxiliary softmax covers the fixed 12-character alphabet, not full vocabulary.
"""
from dataclasses import asdict, dataclass
import json
import math

import torch
from torch.nn import functional as F

from common import TASKS
from kev_ours_v5 import anchored_maps
from numeric_trie import ALPHABET, TrieObjective, components, logit_diagnostics
from models_score_distribution import A4_FLAGS
from models_numeric_trie import FormalModel as TrieModel
from score_specialist import support


@dataclass(frozen=True)
class ConsistencyObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127
    cdf_weight: float = 0.0
    token_weight: float = 0.5

    def __post_init__(self):
        if self.kind not in ('trie_token_ce', 'trie_token_ce_cdf'):
            raise ValueError('Expected trie_token objective')
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale <= 0:
            raise ValueError('Positive ordinal scale required')
        expected = 0. if self.kind == 'trie_token_ce' else .5
        if self.cdf_weight != expected or not math.isfinite(self.token_weight) or self.token_weight <= 0:
            raise ValueError('Frozen objective weights do not match the arm')

    def inventory(self):
        return asdict(self)

    def distribution(self):
        return TrieObjective('trie_ce_cdf' if self.cdf_weight else 'trie_ce',
                             self.ordinal_scale, self.cdf_weight)


def suffix_tokens(row, tokenizer):
    # Identical JSON numeric serialization to the baseline, with no rounding.
    text = json.dumps({'score': row['label']}, separators=(',', ':'), allow_nan=False)[len('{"score":'):]
    if any(c not in ALPHABET for c in text):
        raise ValueError(f'Unsupported decimal character: {text!r}')
    ids = tokenizer.encode(text, add_special_tokens=False)
    expected = [tokenizer.encode(c, add_special_tokens=False)[0] for c in text]
    if ids != expected:
        raise ValueError(f'Numeric suffix does not use the frozen character tokens: {text!r}')
    return ids, [ALPHABET.index(c) for c in text]


class FormalModel(TrieModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None, distribution_objective=None,
                 numeric_anchor_path=None, numeric_anchor_sha256=None):
        objective = ConsistencyObjective(**distribution_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation,
                         distribution_objective=objective.distribution().inventory(),
                         numeric_anchor_path=numeric_anchor_path, numeric_anchor_sha256=numeric_anchor_sha256)
        self.distribution_objective = objective
        self.ablation_inventory.update(score_research_architecture='numeric_trie_joint_token_nll',
                                       shared_causal_forward=True, auxiliary_alphabet=ALPHABET)

    def joint_scores_and_token_loss(self, rows, encodings):
        if not rows or len(rows) != len(encodings) or any(r['task'] != 'score' for r in rows):
            raise ValueError('Paired Score-only inputs required')
        if any(e['labels'] != [None] or e['tasks'] != ['score'] for e in encodings):
            raise ValueError('Cached prompt must not contain answer labels')
        suffixes = [suffix_tokens(row, self.tok) for row in rows]
        # Final closing brace is a prediction target; no state after it is used.
        sequences = [e['ids'] + s[0][:-1] for e, s in zip(encodings, suffixes)]
        ids, positions, attention = self.net._pad_rows([(s, list(range(len(s)))) for s in sequences])
        tasks = torch.full((len(rows),), TASKS.index('score'), dtype=torch.long, device=self.device)
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.net.task_adapter.matrix_raw.dtype
        self.net.task_adapter.prepare(tasks, dtype)
        hs = self.lm(input_ids=ids, position_ids=positions, attention_mask=attention, use_cache=False).last_hidden_state
        starts = [len(e['ids'])-1 for e in encodings]
        hidden = hs[torch.arange(len(rows), device=self.device), torch.tensor(starts, device=self.device)]
        q = self.net.head.q(hidden.float())
        geometry = anchored_maps(self.net.head.geometry_raw, self.net.head.task_weights, .5)[tasks].to(q.dtype)
        q = q + torch.bmm(geometry.transpose(-2,-1), q.unsqueeze(-1)).squeeze(-1)
        scores = self.net.head.numeric_score(hidden, q, [support(r) for r in rows])
        losses = []
        with torch.autocast(device_type=hs.device.type, enabled=False):
            for i, (start, suffix) in enumerate(zip(starts, suffixes)):
                states = hs[i, start:start+len(suffix[1])].float()
                logits = F.linear(states, self.net.head.numeric_score.output_rows)
                targets = torch.tensor(suffix[1], device=logits.device)
                losses.append(F.cross_entropy(logits, targets))
        return scores, torch.stack(losses)

    def losses(self, rows, encodings):
        zs, token = self.joint_scores_and_token_loss(rows, encodings)
        objective = self.distribution_objective.distribution()
        losses = []
        for i, (row, z) in enumerate(zip(rows, zs)):
            total, ce, cdf = components(z, support(row), row['label'], objective)
            losses.append(total + self.distribution_objective.token_weight * token[i])
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((1,7), device=z.device, dtype=torch.float64)
                    self.objective_component_sums = torch.zeros(4, device=z.device, dtype=torch.float64)
                self._loss_diagnostics[0] += logit_diagnostics(z, support(row), row['label'], objective)
                self.objective_component_sums += torch.stack((z.new_ones(()), ce.detach(), cdf.detach(), token[i].detach())).double()
        return torch.stack(losses) + self.net.regularization()

    @torch.no_grad()
    def verify_causal_prefix(self, rows, encodings):
        """Real-backbone audit: no suffix, true suffix, and changed suffix."""
        if self.training:
            raise ValueError('Causality audit requires eval mode (dropout disabled)')
        reference = self.scores(rows, encodings)
        self.clear()
        actual, _ = self.joint_scores_and_token_loss(rows, encodings)
        self.clear()
        changed = [{**r, 'label': support(r)[-1] if r['label'] != support(r)[-1] else support(r)[0]} for r in rows]
        altered, _ = self.joint_scores_and_token_loss(changed, encodings)
        delta = max(float((a.softmax(0)-b.softmax(0)).abs().max())
                    for a,b in zip(reference+reference, actual+altered))
        # FLA chunk boundaries/BF16 may vary with padded suffix length.
        if delta > .005:
            raise AssertionError(f'Suffix changed prefix distribution beyond tolerance: {delta}')
        self.clear()
        return {'rows':len(rows),'max_probability_difference':delta,'tolerance':.005,
                'checks':['prompt-only vs joint prefix','gold vs counterfactual suffix'],
                'suffix_used_for_distribution':False}
