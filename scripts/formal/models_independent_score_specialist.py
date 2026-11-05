"""Frozen independent D2 architecture on integer or continuous Score supports.

Branch execution matches the already-running native ASAP D2 arm, including
chunk size and retaining the query from the first chunk. Only the objective
differs between the paired CE/CDF arms.
"""
import torch
from torch.nn import functional as F

from common import TASKS as MODEL_TASKS
from independent_candidates import shared_hidden, split_candidates
from models_score_specialist import FormalModel as ScoreModel
from score_specialist import support

BRANCH_CHUNK_SIZE = 8


def chunked_shared_hidden(lm, encs, pad_id, tasks, router=None):
    queries, options = [], []
    for enc, task in zip(encs, tasks):
        split = split_candidates(enc)
        query, candidate_hidden = None, []
        for start in range(0, len(split['branches']), BRANCH_CHUNK_SIZE):
            end = start + BRANCH_CHUNK_SIZE
            chunk = {**split, 'branches': split['branches'][start:end],
                     'semantics': split['semantics'][start:end]}
            q, h = shared_hidden(lm, [chunk], pad_id, task[None], router)
            if query is None:
                query = q[0]
            candidate_hidden.extend(h[0].unbind(0))
        queries.append(query)
        options.append(torch.stack(candidate_hidden))
    return torch.stack(queries), options


class FormalModel(ScoreModel):
    def scores(self, rs, es):
        if len(rs) != len(es) or not rs:
            raise ValueError('Expected nonempty paired rows and encodings')
        if any(r['task'] != 'score' or e['tasks'] != ['score'] or
               e['labels'] != [None] for r, e in zip(rs, es)):
            raise ValueError('Raw Score labels stay in rows, never transport labels')
        values = [support(r) for r in rs]
        if any(len(e['opt_idx'][0]) != len(v) for e, v in zip(es, values)):
            raise ValueError('Every score level must have its own candidate branch')
        # The model retains all three task routes: Score is route 2, not route 0.
        task_ids = torch.tensor([MODEL_TASKS.index('score')] * len(rs), device=self.device)
        query, options = chunked_shared_hidden(self.lm, es, self.tok.pad_token_id,
                                               task_ids, self.net.task_adapter)
        lengths = [len(h) for h in options]
        maximum = max(lengths)
        hidden = torch.stack([F.pad(h, (0, 0, 0, maximum - len(h))) for h in options])
        valid = torch.arange(maximum, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        if self.net.head.numeric_supports is not None:
            raise RuntimeError('Overlapping head contexts')
        self.net.head.numeric_supports = values
        try:
            logits = self.net.head.padded(query, hidden, valid, task_ids)
        finally:
            self.net.head.numeric_supports = None
        return [z[:n] for z, n in zip(logits, lengths)]
