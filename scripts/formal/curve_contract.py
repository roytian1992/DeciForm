"""Validate frozen record-level nested subsets without resampling candidates."""
from collections import Counter
import json
from pathlib import Path

from common import sha


def load_subset(path, expected_sha256, rows, expected_size):
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('Frozen subset hash mismatch')
    subset = json.loads(path.read_text())
    ids = subset['indices']
    if subset['status'] != 'passed' or subset['source_rows'] != len(rows):
        raise ValueError('Subset belongs to a different source population')
    if subset['size'] != expected_size or len(ids) != expected_size:
        raise ValueError('Subset size mismatch')
    if (not ids or any(type(i) is not int for i in ids)
            or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= len(rows)):
        raise ValueError('Subset indices must be sorted, unique and in range')
    for key, field in [('task_counts', 'task'), ('source_counts', 'source')]:
        if dict(Counter(rows[i][field] for i in ids)) != subset[key]:
            raise ValueError(f'Subset {key} do not match source rows')
    return ids


def require_nested(subsets):
    ordered = sorted(subsets.items())
    for (small_size, small), (large_size, large) in zip(ordered, ordered[1:]):
        if not set(small) < set(large):
            raise ValueError(f'Subsets {small_size}/{large_size} are not strictly nested')
