"""Complete-input encodings; semantic IDs are alignment keys, not learned routing."""

import json
import torch
from .schemas import normalize, support, canonical_order
from .integrations.kev_encoding import user_tokens

SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")


def score_prompt(row):
    values = support(row)
    text = f'State: {row["state"]["text"]}\nQuestion: {row["question"]["instructions"]}\n'
    if row["label_kind"] == "native_integer":
        text += (
            "Allowed scores: "
            + json.dumps(values)
            + '. Return only one JSON object with the field "score" containing one allowed integer.'
        )
    else:
        text += f'Allowed score range: [{values[0]:g}, {values[-1]:g}]. Fractional scores are valid. Return only one JSON object with the field "score" containing a finite JSON number.'
    return text + "\nAnswer: "


def qwen_encode(record, tok):
    row = normalize(record)
    order = canonical_order(row)
    if row["task"] == "score":
        ids = tok(
            score_prompt(row), add_special_tokens=True, truncation=False
        ).input_ids + tok.encode('{"score":', add_special_tokens=False)
        return {
            "ids": ids,
            "tasks": ["score"],
            "labels": [None],
            "original_tokens": len(ids),
            "canonical_to_row": order,
        }
    special = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    if any(t is None or t == tok.unk_token_id for t in special):
        raise ValueError("Tokenizer lacks required Qwen delimiter tokens")
    prefix = (
        [special[0]]
        + user_tokens(tok, row["state"]["text"])
        + [special[1]]
        + user_tokens(tok, row["question"]["instructions"])
    )
    values = support(row)
    texts = (
        row["question"].get("option_definitions", row["question"].get("options"))
        if row["task"] == "choice"
        else ["yes: The statement is true" if v else "no: The statement is false" for v in values]
    )
    branches = [[special[2]] + user_tokens(tok, texts[i]) + [special[3]] for i in order]
    return {
        "prefix": prefix,
        "branches": branches,
        "query_token": special[4],
        "semantics": list(range(len(branches))),
        "tasks": [row["task"]],
        "labels": [None],
        "original_tokens": len(prefix) + sum(map(len, branches)) + 1,
        "canonical_to_row": order,
    }


def modern_encode(record, tok):
    row = normalize(record)
    task = row["task"]
    q = row["question"]
    values = support(row)
    order = canonical_order(row)

    def tokens(text):
        return tok(
            text.replace(tok.mask_token, " "), add_special_tokens=False, truncation=False
        ).input_ids

    shared = (
        [tok.cls_token_id]
        + tokens(f"{task} question: {q['instructions']}")
        + [tok.sep_token_id, tok.sep_token_id]
        + tokens(row["state"]["text"])
        + [tok.sep_token_id]
    )
    if task == "score":
        shared += tok.encode(
            " Allowed scores: " + ", ".join(format(v, ".12g") for v in values),
            add_special_tokens=False,
        ) + [tok.sep_token_id]
    query = len(shared)
    shared.append(tok.mask_token_id)
    ids = list(shared)
    positions = list(range(len(ids)))
    branches = [0] * len(ids)
    markers = []
    if task == "score":
        markers = [query] * len(values)
    else:
        if task == "noul":
            texts = [
                "true: The statement is true" if v else "false: The statement is false"
                for v in values
            ]
        else:
            definitions = q.get("option_definitions", q["options"])
            keys = row.get("json_keys", [str(v) for v in values])
            if len(keys) != len(values):
                raise ValueError("Misaligned semantic candidate names")
            texts = [str(k) if v == "" else f"{k}: {v}" for k, v in zip(keys, definitions)]
        for slot, i in enumerate(order):
            branch = [tok.mask_token_id] + tokens(" " + texts[i])
            markers.append(len(ids))
            ids += branch
            positions += list(range(len(shared), len(shared) + len(branch)))
            branches += [slot + 1] * len(branch)
    return {
        "ids": ids,
        "markers": markers,
        "qtype": {"choice": 0, "score": 1, "noul": 2}[task],
        "layout": {"position_ids": positions, "option_ids": branches},
        "support": values,
        "task": task,
        "query_position": query,
        "canonical_to_row": order,
        "original_tokens": len(ids),
    }


def collate_modern(items, pad_id, device):
    n = len(items)
    length = max(len(it["ids"]) for it in items)
    k = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    pos = torch.zeros_like(ids)
    opt = torch.zeros_like(ids)
    markers = torch.zeros((n, k), dtype=torch.long, device=device)
    mask = torch.zeros((n, k), dtype=torch.bool, device=device)
    for i, it in enumerate(items):
        l = len(it["ids"])
        c = len(it["markers"])
        ids[i, :l] = torch.tensor(it["ids"], device=device)
        att[i, :l] = 1
        pos[i, :l] = torch.tensor(it["layout"]["position_ids"], device=device)
        opt[i, :l] = torch.tensor(it["layout"]["option_ids"], device=device)
        markers[i, :c] = torch.tensor(it["markers"], device=device)
        mask[i, :c] = True
    return dict(
        input_ids=ids,
        attention_mask=att,
        marker_pos=markers,
        marker_mask=mask,
        qtype=torch.tensor([it["qtype"] for it in items], device=device),
        position_ids=pos,
        option_ids=opt,
        items=items,
    )
