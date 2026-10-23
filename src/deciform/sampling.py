"""Deterministic positive-plus-negative candidate sampling."""
import hashlib
from .choice import render_choice_prompts

def choose_candidates(row: dict, seed: int, negative_count: int) -> list[int]:
    option_ids, _ = render_choice_prompts(row)
    positive = option_ids.index(row["label"])
    negatives = sorted(
        (index for index in range(len(option_ids)) if index != positive),
        key=lambda index: hashlib.sha256(
            f"negative:{seed}:{row['id']}:{option_ids[index]}".encode()
        ).hexdigest(),
    )[:negative_count]
    if len(negatives) != negative_count:
        raise ValueError("candidate budget exceeds available negatives")
    return sorted(
        [positive, *negatives],
        key=lambda index: hashlib.sha256(
            f"order:{seed}:{row['id']}:{option_ids[index]}".encode()
        ).hexdigest(),
    )

def candidate_subset(row: dict, indices: list[int]) -> dict:
    question = row["question"].copy()
    for key in ("options", "option_ids", "option_definitions"):
        if key in question:
            question[key] = [question[key][index] for index in indices]
    return {**row, "question": question}
