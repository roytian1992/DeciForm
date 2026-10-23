"""Deterministic development panels and descriptive probability metrics."""
from collections import Counter, defaultdict
import hashlib
import math


def sample_key(row, seed):
    return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()


def select_panels(rows, *, seed, target=1000):
    """Natural sample first; balanced sample from disjoint remaining records.

    Natural means the eligible source population, not a deployment prior.
    Scarce sources get a smaller natural panel and equal per-class quotas.
    """
    ordered = sorted(rows, key=lambda row: sample_key(row, seed))
    natural_count = min(target, len(ordered) // 2)
    natural, remaining = ordered[:natural_count], ordered[natural_count:]
    labels = sorted({label for row in rows for label in row['question']['options']})
    groups = defaultdict(list)
    for row in remaining:
        groups[row['label']].append(row)
    quota = min([target // len(labels)] + [len(groups[label]) for label in labels]) if labels else 0
    balanced = [row for label in labels for row in groups[label][:quota]]
    balanced.sort(key=lambda row: sample_key(row, seed + 1))
    return {'natural': natural, 'balanced': balanced}


from deciform.metrics import summarize
