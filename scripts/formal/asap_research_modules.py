"""Full-text evidence pooling and a median-parameterized ordinal distribution."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EssayObjective:
    kind: str
    ordinal_scale: float = 3.0383850790577127
    median_weight: float = 0.0

    def __post_init__(self):
        if self.kind not in ('evidence_cdf', 'median_laplace', 'evidence_median_laplace'):
            raise ValueError('Unknown ASAP research arm')
        if not math.isfinite(self.ordinal_scale) or self.ordinal_scale <= 0:
            raise ValueError('Positive CDF scale required')
        if self.median_weight != (0.0 if self.kind == 'evidence_cdf' else 1.0):
            raise ValueError('Use the preregistered objective weight')

    def inventory(self):
        return asdict(self)


class QueryEvidencePool(nn.Module):
    """Four query-conditioned summaries over all valid prefix token states."""
    def __init__(self, token_size, query_size, rank=64, heads=4):
        super().__init__()
        self.rank, self.heads = rank, heads
        self.key_value = nn.Linear(token_size, rank, bias=False)
        self.query = nn.Linear(query_size, heads*rank, bias=False)
        self.slots = nn.Parameter(torch.randn(heads, rank)*.02)
        self.output = nn.Linear(heads*rank, query_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self, tokens, query, valid):
        if valid.shape != tokens.shape[:2] or not bool(valid.any(-1).all()):
            raise ValueError('Every row needs a nonempty complete prefix')
        values = self.key_value(F.layer_norm(tokens.float(), (tokens.shape[-1],)))
        queries = self.query(F.layer_norm(query.float(), (query.shape[-1],)))
        queries = queries.reshape(len(query), self.heads, self.rank) + self.slots
        logits = torch.einsum('bhr,btr->bht', queries.float(), values.float())/math.sqrt(self.rank)
        attention = logits.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        contexts = torch.einsum('bht,btr->bhr', attention, values.float()).flatten(1)
        delta = self.output(contexts).float()
        return query.float() + self.gate_logit.sigmoid()*delta


def laplace_log_probabilities(location, scale, support):
    """Censor a latent Laplace distribution into the nearest complete native level.

    Midpoints define cells; the first and last cells absorb both infinite tails.
    The native median is exactly nearest-support(location), including boundaries.
    """
    if location.numel() != 1 or scale.numel() != 1:
        raise ValueError('Scalar location and scale required')
    values = torch.as_tensor(support, device=location.device, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not bool(torch.isfinite(values).all()) or not bool((values.diff()>0).all()):
        raise ValueError('Complete finite increasing score support required')
    if not bool(torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError('Positive finite scale required')
    t = (values-values[0])/(values[-1]-values[0])
    edges = (t[1:]+t[:-1])/2
    z = (edges-location.double())/scale.double()
    # Clamp only the unused branch: finite tails and the correct derivative at z=0.
    left = .5*torch.exp(z.clamp_max(0))
    right = 1-.5*torch.exp(-z.clamp_min(0))
    cdf = torch.where(z <= 0, left, right)
    masses = torch.cat((cdf[:1], cdf.diff(), 1-cdf[-1:])).clamp_min(1e-12)
    return (masses.log()-masses.sum().log()).to(location.dtype)


class MedianLaplaceReadout(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.project = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[1] = math.log(.14/.35)

    def forward(self, features, supports):
        h = F.silu(self.project(F.layer_norm(features.float(), (features.shape[-1],))))
        parameters = self.output(h).float()
        locations = parameters[:, 0].sigmoid()
        scales = .01 + .49*parameters[:, 1].sigmoid()
        logits = [laplace_log_probabilities(m, b, s)
                  for m, b, s in zip(locations, scales, supports)]
        return logits, locations, scales
