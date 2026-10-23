"""Shard complete inference batches without changing their padding or order."""
from __future__ import annotations


def assigned_prompt_indices(total: int, batch_size: int, rank: int, world: int) -> list[int]:
    if total < 0 or batch_size < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError("invalid inference shard")
    return [i for block, start in enumerate(range(0, total, batch_size))
            if block % world == rank for i in range(start, min(start + batch_size, total))]


def restore_prompt_order(parts, total: int):
    import torch
    indices = [i for positions, _ in parts for i in positions]
    if sorted(indices) != list(range(total)):
        raise ValueError("inference shards must cover each prompt exactly once")
    features = torch.cat([values for _, values in parts if values is not None], dim=0)
    if features.shape[0] != total:
        raise ValueError("inference feature count differs from prompt count")
    return features[torch.argsort(torch.tensor(indices))]
