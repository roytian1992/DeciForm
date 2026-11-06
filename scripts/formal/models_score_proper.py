"""Frozen specialist D2 architectures with only Score data loss replaced."""
import torch
import models_independent
import independent_candidates
from models_score_research import BatchedD2
from score_proper import ScoreProperLossMixin, ScoreProperObjective

_shared_hidden = independent_candidates.shared_hidden


def asap_shared_hidden(lm, encs, pad_id, tasks, router=None):
    # Exact chunk-eight execution used by the completed ASAP D2 reference.
    queries, options = [], []
    for enc, task in zip(encs, tasks):
        split = independent_candidates.split_candidates(enc)
        query, hidden = None, []
        for start in range(0, len(split['branches']), 8):
            chunk = {**split, 'branches':split['branches'][start:start+8],
                     'semantics':split['semantics'][start:start+8]}
            q, h = _shared_hidden(lm, [chunk], pad_id, task[None], router)
            if query is None:
                query = q[0]
            hidden.extend(h[0].unbind(0))
        queries.append(query)
        options.append(torch.stack(hidden))
    return torch.stack(queries), options


class ASAPProper(ScoreProperLossMixin, models_independent.FormalModel):
    def __init__(self, *args, distribution_objective=None, **kwargs):
        objective = ScoreProperObjective(**distribution_objective)
        super().__init__(*args, distribution_objective={'kind':'ordinal_main',
                         'moment_weight':.5,'ordinal_scale':3.0383850790577127}, **kwargs)
        self.distribution_objective = objective


class ContinuousProper(ScoreProperLossMixin, BatchedD2):
    def __init__(self, *args, distribution_objective=None, **kwargs):
        objective = ScoreProperObjective(**distribution_objective)
        super().__init__(*args, distribution_objective={'kind':'soft_cdf',
                         'ordinal_scale':3.0383850790577127}, **kwargs)
        self.distribution_objective = objective


def model_for_panel(panel):
    if panel == 'asap':
        models_independent.shared_hidden = asap_shared_hidden
        return ASAPProper
    if panel == 'continuous':
        return ContinuousProper
    raise ValueError(panel)
