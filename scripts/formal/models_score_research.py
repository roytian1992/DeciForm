"""RQ-Score: batched final-D2 reference and a single-encoding numeric readout."""
import hashlib
import torch
from torch.nn import functional as F

from common import TASKS
from independent_candidates import shared_hidden, split_candidates
from kev_ours_v5 import anchored_maps
from models_score_specialist import FormalModel as ScoreModel
from numeric_score_distribution import NumericCDFReadout
from score_specialist import support


class BatchedD2(ScoreModel):
    def scores(self, rows, encodings):
        if not rows or len(rows) != len(encodings) or any(r['task'] != 'score' for r in rows):
            raise ValueError('Expected Score-only paired rows and full encodings')
        values = [support(r) for r in rows]
        if any(e['labels'] != [None] or len(e['opt_idx'][0]) != len(s)
               for e, s in zip(encodings, values)):
            raise ValueError('Continuous labels stay outside transport labels; retain every score level')
        tasks = torch.full((len(rows),), TASKS.index('score'), dtype=torch.long, device=self.device)
        q, options = shared_hidden(self.lm, [split_candidates(e) for e in encodings],
                                   self.tok.pad_token_id, tasks, self.net.task_adapter)
        lengths = [len(h) for h in options]
        maximum = max(lengths)
        h = torch.stack([F.pad(v, (0, 0, 0, maximum-len(v))) for v in options])
        valid = torch.arange(maximum, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        if self.net.head.numeric_supports is not None:
            raise RuntimeError('Overlapping numeric contexts')
        self.net.head.numeric_supports = values
        try:
            scores = self.net.head.padded(q, h, valid, tasks)
        finally:
            self.net.head.numeric_supports = None
        return [z[:n] for z, n in zip(scores, lengths)]


class NumericD2(ScoreModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        digest = hashlib.sha256()
        for name, parameter in self.net.head.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        self.initial_shared_head_sha256 = digest.hexdigest()
        # All base/LoRA/router/geometry tensors retain fresh D2 initialization.
        with torch.random.fork_rng(devices=[]):
            self.net.head.numeric_score = NumericCDFReadout(self.net.head.q.out_features).to(self.device)
        for module in (self.net.head.k, self.net.head.project, self.net.head.query_condition,
                       self.net.head.qf_out, self.net.head.ordinal):
            module.requires_grad_(False)
        self.ablation_inventory = {**self.ablation_inventory,
            'ordinal_readout_enabled': False, 'numeric_cdf_enabled': True,
            'frozen_component_parameters': ['head.' + name for name, p in self.net.head.named_parameters()
                                            if not p.requires_grad],
            'trainable_parameters': sum(p.numel() for p in self.net.parameters() if p.requires_grad),
            'scope': 'Score-specific readout replacement; no claim of unchanged full three-task head'}

    def scores(self, rows, encodings):
        if not rows or len(rows) != len(encodings) or any(r['task'] != 'score' for r in rows):
            raise ValueError('Expected Score-only paired rows')
        values = [support(r) for r in rows]
        splits = [split_candidates(e) for e in encodings]
        # Rubric, original range and state are entirely in the prefix. Numerical
        # levels are consumed by the output projection, not discarded from output.
        # Encode prefix+query together: the FLA one-token recurrent kernel used
        # by a query-only branch has no CUDA backward implementation.
        tasks = torch.full((len(rows),), TASKS.index('score'), dtype=torch.long, device=self.device)
        sequences = [s['prefix'] + [s['query_token']] for s in splits]
        ids, positions, attention = self.net._pad_rows([(s, list(range(len(s)))) for s in sequences])
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else self.net.task_adapter.matrix_raw.dtype
        self.net.task_adapter.prepare(tasks, dtype)
        hidden = self.lm(input_ids=ids, position_ids=positions, attention_mask=attention,
                         use_cache=False).last_hidden_state
        lengths = torch.tensor([len(s)-1 for s in sequences], device=self.device)
        q = hidden[torch.arange(len(rows), device=self.device), lengths].float()
        projected = self.net.head.q(q)
        geometry = anchored_maps(self.net.head.geometry_raw, self.net.head.task_weights, .5)[tasks].to(projected.dtype)
        conditioned = projected + torch.bmm(geometry.transpose(-2, -1), projected.unsqueeze(-1)).squeeze(-1)
        return self.net.head.numeric_score(conditioned, values)
