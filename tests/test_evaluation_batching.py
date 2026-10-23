import pytest
import torch
from deciform.evaluation_batching import assigned_prompt_indices, restore_prompt_order


@pytest.mark.parametrize("total,world", [(33, 2), (120, 4), (3, 4), (256, 2)])
def test_parallel_evaluation_preserves_complete_batches_and_order(total, world):
    reference = torch.arange(total * 3).reshape(total, 3)
    parts = []
    for rank in range(world):
        positions = assigned_prompt_indices(total, 16, rank, world)
        for start in range(0, len(positions), 16):
            batch = positions[start:start + 16]
            assert batch == list(range(batch[0], min(batch[0] + 16, total)))
            assert batch[0] % 16 == 0
        parts.append((positions, reference[positions] if positions else None))
    assert torch.equal(restore_prompt_order(parts, total), reference)


def test_inference_shards_reject_duplicates_and_missing_prompts():
    with pytest.raises(ValueError, match="exactly once"):
        restore_prompt_order([([0, 0], torch.ones(2, 3))], 2)
