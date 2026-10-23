"""Candidate-conditioned choice scoring with a frozen text backbone."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ChoiceHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(hidden.float(), (self.hidden_size,))
        return self.linear(normalized).squeeze(-1)


def render_choice_prompts(
    row: dict, order: Sequence[int] | None = None
) -> tuple[list[str], list[str]]:
    question = row["question"]
    if question["type"] != "choice":
        raise ValueError("choice scorer only accepts choice questions")
    options = question["options"]
    ids = question.get("option_ids", options)
    definitions = question.get("option_definitions", options)
    if len(options) < 2 or len(ids) != len(options) or len(definitions) != len(options):
        raise ValueError("choice options, IDs and definitions must align")
    if len(set(ids)) != len(ids):
        raise ValueError("option IDs must be unique")
    indices = list(range(len(options))) if order is None else list(order)
    if sorted(indices) != list(range(len(options))):
        raise ValueError("order must be an option permutation")
    prompts = [
        f"State: {row['state']['text']}\n"
        f"Question: {question['instructions']}\n"
        f"Candidate: {definitions[index]}"
        for index in indices
    ]
    return [ids[index] for index in indices], prompts


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden and attention_mask have incompatible shapes")
    if not bool(attention_mask.any(dim=1).all()):
        raise ValueError("every candidate needs at least one token")
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    last = positions.expand(hidden.shape[0], -1).masked_fill(
        ~attention_mask.to(device=hidden.device, dtype=torch.bool), -1
    ).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()


def encode_prompts(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_prompt_tokens: int | None,
    fixed_padding: bool = False,
    bucket_by_length: bool = False,
) -> torch.Tensor:
    if batch_size < 1 or (max_prompt_tokens is not None and max_prompt_tokens < 1):
        raise ValueError("batch_size must be positive and a prompt cap, when set, must be positive")
    if fixed_padding and bucket_by_length:
        raise ValueError("fixed padding and length buckets are mutually exclusive")
    if bucket_by_length:
        encoded = tokenizer(
            list(prompts), padding=False, truncation=False, add_special_tokens=True
        )
        groups: dict[int, list[int]] = defaultdict(list)
        for index, token_ids in enumerate(encoded["input_ids"]):
            length = len(token_ids)
            if length < 1:
                raise ValueError("prompt must contain at least one token")
            if max_prompt_tokens is not None and length > max_prompt_tokens:
                raise ValueError(f"prompt length {length} exceeds {max_prompt_tokens}")
            groups[length].append(index)
        if not groups:
            raise ValueError("no candidate prompts to encode")
        ordered: list[torch.Tensor | None] = [None] * len(prompts)
        model.eval()
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                input_ids = torch.tensor(
                    [encoded["input_ids"][index] for index in chunk],
                    dtype=torch.long, device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    hidden = model.model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True,
                    ).last_hidden_state
                    vectors = last_token_pool(hidden, attention_mask).cpu()
                for position, index in enumerate(chunk):
                    ordered[index] = vectors[position]
        if any(vector is None for vector in ordered):
            raise RuntimeError("candidate feature missing from length bucket")
        return torch.stack(ordered)
    features = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(
            list(prompts[start : start + batch_size]),
            padding="max_length" if fixed_padding else True,
            max_length=max_prompt_tokens if fixed_padding else None,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        longest = int(tokens["attention_mask"].sum(dim=1).max())
        if max_prompt_tokens is not None and longest > max_prompt_tokens:
            raise ValueError(f"prompt length {longest} exceeds {max_prompt_tokens}")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        with torch.inference_mode():
            hidden = model.model(**tokens, use_cache=False, return_dict=True).last_hidden_state
            features.append(last_token_pool(hidden, tokens["attention_mask"]).cpu())
    if not features:
        raise ValueError("no candidate prompts to encode")
    return torch.cat(features, dim=0)


def choice_response(
    option_ids: Sequence[str], logits: torch.Tensor, *, temperature: float = 1.0
) -> dict:
    if temperature <= 0 or len(option_ids) != logits.numel():
        raise ValueError("temperature and option IDs are invalid")
    probabilities = torch.softmax(logits.float().reshape(-1) / temperature, dim=0)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("non-finite choice probabilities")
    values = probabilities.detach().cpu().tolist()
    selected = int(probabilities.argmax())
    return {
        "type": "choice",
        "option_ids": list(option_ids),
        "probabilities": dict(zip(option_ids, values, strict=True)),
        "prediction": option_ids[selected],
        "confidence": values[selected],
        "abstain": False,
    }
