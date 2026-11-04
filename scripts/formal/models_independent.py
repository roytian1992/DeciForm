"""Frozen D2 parameterization/loss with independent shared-prefix candidates."""
import torch
from torch.nn import functional as F

from common import TASKS, support
from models_score_distribution import *
from models_score_distribution import FormalModel as D2Model
from independent_candidates import shared_hidden, stable_argmax


class FormalModel(D2Model):
    def scores(self, rs, es):
        if any(e['tasks'] != [r['task']] for r, e in zip(rs, es)):
            raise ValueError('Task identity and candidate encoding disagree')
        task_ids = torch.tensor([TASKS.index(r['task']) for r in rs], device=self.device)
        query, options = shared_hidden(self.lm, es, self.tok.pad_token_id,
                                       task_ids, self.net.task_adapter)
        lengths = [len(h) for h in options]
        maximum = max(lengths)
        hidden = torch.stack([F.pad(h, (0, 0, 0, maximum - len(h))) for h in options])
        valid = torch.arange(maximum, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        if self.net.head.numeric_supports is not None:
            raise RuntimeError('Overlapping head contexts')
        self.net.head.numeric_supports = [support(r) if r['task'] == 'score' else None for r in rs]
        try:
            logits = self.net.head.padded(query, hidden, valid, task_ids)
        finally:
            self.net.head.numeric_supports = None
        return [z[:n] for z, n in zip(logits, lengths)]

    @torch.no_grad()
    def predict(self, rs, es, max_new_tokens):
        predictions = super().predict(rs, es, max_new_tokens)
        for row, prediction in zip(rs, predictions):
            if row['task'] in ('choice', 'noul'):
                prediction['prediction_index'] = stable_argmax(prediction['probabilities'], support(row))
        return predictions
