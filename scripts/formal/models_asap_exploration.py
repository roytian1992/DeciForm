"""Two ASAP-only hypotheses and their combination, with a fixed backbone/data."""
import hashlib
import torch

from common import TASKS
from independent_candidates import split_candidates
from kev_ours_v5 import anchored_maps
from models_score_distribution import A4_FLAGS
from models_score_research import NumericD2
from score_specialist import SpecialistObjective, data_loss, logit_diagnostics, support
from asap_research_modules import EssayObjective, QueryEvidencePool, MedianLaplaceReadout


def digest(items):
    value = hashlib.sha256()
    for name, parameter in items:
        value.update(name.encode())
        value.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class FormalModel(NumericD2):
    def __init__(self, arm, base, checkpointing=True, ablation=None, distribution_objective=None):
        objective = EssayObjective(**distribution_objective)
        super().__init__(arm, base, checkpointing=checkpointing, ablation=ablation,
                         distribution_objective={'kind':'soft_cdf','ordinal_scale':objective.ordinal_scale})
        self.initial_numeric_head_sha256 = digest(self.net.head.named_parameters())
        self.distribution_objective = objective
        if objective.kind in ('evidence_cdf', 'evidence_median_laplace'):
            with torch.random.fork_rng(devices=[]):
                self.net.head.evidence_pool = QueryEvidencePool(
                    self.net.head.q.in_features, self.net.head.q.out_features).to(self.device)
        if objective.kind in ('median_laplace', 'evidence_median_laplace'):
            with torch.random.fork_rng(devices=[]):
                readout = MedianLaplaceReadout(self.net.head.q.out_features).to(self.device)
                # Preserve the shared projection's initial values; new outputs start fresh.
                readout.project.load_state_dict(self.net.head.numeric_score.project.state_dict())
                self.net.head.numeric_score = readout
        self.ablation_inventory.update(
            score_research_architecture=objective.kind,
            trainable_parameters=sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        self.objective_component_sums = None
        self._locations = self._scales = None

    def scores(self, rows, encodings):
        if not rows or len(rows) != len(encodings) or any(r['task'] != 'score' for r in rows):
            raise ValueError('Paired Score-only inputs required')
        values = [support(r) for r in rows]
        sequences = [s['prefix']+[s['query_token']] for s in map(split_candidates, encodings)]
        tasks = torch.full((len(rows),), TASKS.index('score'), dtype=torch.long, device=self.device)
        ids, positions, attention = self.net._pad_rows([(s, list(range(len(s)))) for s in sequences])
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.net.task_adapter.matrix_raw.dtype
        self.net.task_adapter.prepare(tasks, dtype)
        hidden = self.lm(input_ids=ids, position_ids=positions, attention_mask=attention,
                         use_cache=False).last_hidden_state
        lengths = torch.tensor([len(s)-1 for s in sequences], device=self.device)
        q = self.net.head.q(hidden[torch.arange(len(rows), device=self.device), lengths].float())
        geometry = anchored_maps(self.net.head.geometry_raw, self.net.head.task_weights, .5)[tasks].to(q.dtype)
        q = q + torch.bmm(geometry.transpose(-2,-1), q.unsqueeze(-1)).squeeze(-1)
        if self.distribution_objective.kind in ('evidence_cdf', 'evidence_median_laplace'):
            valid = torch.arange(hidden.shape[1], device=self.device)[None] < lengths[:,None]
            q = self.net.head.evidence_pool(hidden, q, valid)
        if self.distribution_objective.kind == 'evidence_cdf':
            self._locations = self._scales = None
            return self.net.head.numeric_score(q, values)
        logits, self._locations, self._scales = self.net.head.numeric_score(q, values)
        return logits

    def losses(self, rows, encodings):
        logits = self.scores(rows, encodings)
        objective = SpecialistObjective('soft_cdf', self.distribution_objective.ordinal_scale)
        losses = []
        for i, (row, z) in enumerate(zip(rows, logits)):
            values = support(row)
            cdf = data_loss(z, values, row['label'], objective)
            regression = z.new_zeros(())
            scale = z.new_zeros(())
            if self._locations is not None:
                target = (row['label']-values[0])/(values[-1]-values[0])
                regression = (self._locations[i]-target).abs()
                scale = self._scales[i]
            losses.append(cdf + self.distribution_objective.median_weight*regression)
            if self.collect_loss_diagnostics:
                if self._loss_diagnostics is None:
                    self._loss_diagnostics = torch.zeros((1,7), device=z.device, dtype=torch.float64)
                    self.objective_component_sums = torch.zeros(4, device=z.device, dtype=torch.float64)
                self._loss_diagnostics[0] += logit_diagnostics(z, values, row['label'], objective)
                self.objective_component_sums += torch.stack((z.new_ones(()), cdf.detach(),
                    regression.detach(), scale.detach())).double()
        return torch.stack(losses) + self.net.regularization()

    def take_objective_components(self):
        value = self.objective_component_sums
        self.objective_component_sums = None
        return value if value is not None else torch.zeros(4, device=self.device, dtype=torch.float64)

    def clear(self):
        super().clear()
        self._locations = self._scales = None
