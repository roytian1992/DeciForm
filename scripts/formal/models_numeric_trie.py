"""Score-only pretrained numeric-root prior + compact decimal decoder."""
import hashlib
from pathlib import Path

import torch

from common import TASKS, sha
from kev_ours_v5 import anchored_maps
from models_score_distribution import A4_FLAGS
from models_score_research import NumericD2
from score_specialist import support
from numeric_trie import ALPHABET, NumericTrieReadout, TrieObjective, components, logit_diagnostics


class FormalModel(NumericD2):
    def __init__(self, arm, base, checkpointing=True, ablation=None, distribution_objective=None,
                 numeric_anchor_path=None, numeric_anchor_sha256=None):
        objective = TrieObjective(**distribution_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation,
                         distribution_objective={'kind':'soft_cdf', 'ordinal_scale':objective.ordinal_scale})
        digest = hashlib.sha256()
        for name, parameter in self.net.head.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        self.initial_numeric_head_sha256 = digest.hexdigest()
        if sha(numeric_anchor_path) != numeric_anchor_sha256:
            raise ValueError('Frozen numeric anchor artifact changed')
        anchors = torch.load(numeric_anchor_path, map_location='cpu', weights_only=True)
        assert anchors['alphabet'] == ALPHABET and anchors['base'] == str(Path(base).resolve())
        assert anchors['token_ids'] == [self.tok.encode(c, add_special_tokens=False)[0] for c in ALPHABET]
        with torch.random.fork_rng(devices=[]):
            self.net.head.numeric_score = NumericTrieReadout(
                anchors['output_rows'], anchors['input_rows'], self.net.head.q.out_features).to(self.device)
        self.distribution_objective = objective
        self.objective_component_sums = None
        self.ablation_inventory.update(score_research_architecture='pretrained_numeric_root_trie',
            numeric_cdf_enabled=False, numeric_anchor_sha256=numeric_anchor_sha256,
            trainable_parameters=sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def scores(self, rows, encodings):
        if not rows or len(rows) != len(encodings) or any(r['task'] != 'score' for r in rows):
            raise ValueError('Paired Score-only rows required')
        if any(e['labels'] != [None] or e['tasks'] != ['score'] for e in encodings):
            raise ValueError('No gold answer may enter the encoding')
        sequences = [e['ids'] for e in encodings]
        ids, positions, attention = self.net._pad_rows([(s, list(range(len(s)))) for s in sequences])
        tasks = torch.full((len(rows),), TASKS.index('score'), dtype=torch.long, device=self.device)
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.net.task_adapter.matrix_raw.dtype
        self.net.task_adapter.prepare(tasks, dtype)
        hs = self.lm(input_ids=ids, position_ids=positions, attention_mask=attention, use_cache=False).last_hidden_state
        lengths = torch.tensor([len(s)-1 for s in sequences], device=self.device)
        hidden = hs[torch.arange(len(rows), device=self.device), lengths]
        q = self.net.head.q(hidden.float())
        geometry = anchored_maps(self.net.head.geometry_raw, self.net.head.task_weights, .5)[tasks].to(q.dtype)
        q = q + torch.bmm(geometry.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)
        return self.net.head.numeric_score(hidden, q, [support(r) for r in rows])

    def losses(self, rows, encodings):
        losses = []
        for row, z in zip(rows, self.scores(rows, encodings)):
            total, ce, cdf = components(z, support(row), row['label'], self.distribution_objective)
            losses.append(total)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((1,7), device=z.device, dtype=torch.float64)
                    self.objective_component_sums = torch.zeros(4, device=z.device, dtype=torch.float64)
                self._loss_diagnostics[0] += logit_diagnostics(z, support(row), row['label'], self.distribution_objective)
                self.objective_component_sums += torch.stack((z.new_ones(()), ce.detach(), cdf.detach(), total.detach())).double()
        return torch.stack(losses) + self.net.regularization()

    def take_objective_components(self):
        value = self.objective_component_sums
        self.objective_component_sums = None
        return value if value is not None else torch.zeros(4, device=self.device, dtype=torch.float64)
