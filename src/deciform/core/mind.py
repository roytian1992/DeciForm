"""MIND-small TSV to complete-history, multi-positive decision records."""

import json
from pathlib import Path
from .data import atomic_json, sha256, training_view
from .schemas import normalize


def news_text(item):
    return "\n".join(
        f"{label}: {item[key]}"
        for key, label in [
            ("category", "Category"),
            ("subcategory", "Subcategory"),
            ("title", "Title"),
            ("abstract", "Abstract"),
        ]
        if item[key]
    )


def prepare_mind(news_path, behaviors_path, split, output, negative_count=3, seed=20261114):
    if split not in ("train", "dev"):
        raise ValueError("Use official labeled train or dev; Dev is not the hidden official test")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    news = {}
    with open(news_path) as f:
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 8:
                raise ValueError("MIND news.tsv requires eight fields")
            nid, category, subcategory, title, abstract, _, _, _ = fields
            if nid in news:
                raise ValueError("Duplicate news ID")
            news[nid] = dict(
                category=category, subcategory=subcategory, title=title, abstract=abstract
            )
    path = out / f"{split}.jsonl"
    views = out / "train.views.jsonl"
    seen = set()
    n = 0
    with open(behaviors_path) as source, path.open("x") as f:
        view = views.open("x") if split == "train" else None
        try:
            for line in source:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 5:
                    raise ValueError("MIND behaviors.tsv requires five fields")
                iid, uid, timestamp, history, impressions = fields
                if iid in seen:
                    raise ValueError("Duplicate impression ID")
                seen.add(iid)
                history = history.split()
                ids = []
                clicks = []
                for value in impressions.split():
                    nid, flag = value.rsplit("-", 1)
                    if flag not in ("0", "1"):
                        raise ValueError("Unlabeled or invalid impression")
                    ids.append(nid)
                    clicks.append(int(flag))
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate candidates")
                count = sum(clicks)
                if count == 0 or count == len(ids):
                    raise ValueError(
                        "Ranking metrics require clicks and non-clicks; source row retained for inspection, no silent exclusion"
                    )
                history_text = "\n\n".join(news_text(news[nid]) for nid in history)
                state = (
                    f"Previously clicked news ({len(history)} articles, in supplied order):\n"
                    + (history_text if history else "[No recorded reading history]")
                )
                row = {
                    "id": f"mind:small:{split}:{iid}",
                    "task": "choice",
                    "source": "mind_small",
                    "source_split": split,
                    "state": state,
                    "question": {
                        "instructions": "Given the previously clicked news, rank the candidate news by how likely this user is to click each in the current impression.",
                        "options": [news_text(news[nid]) for nid in ids],
                        "option_ids": ids,
                    },
                    "target_distribution": [v / count for v in clicks],
                    "click_labels": clicks,
                    "native_candidate_count": len(ids),
                    "history_ids": history,
                    "provenance": {
                        "impression_id": iid,
                        "timestamp": timestamp,
                        "source_row": n + 1,
                    },
                }
                # IDs of users never enter model input. Candidate IDs remain alignment keys.
                row = normalize(row, True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if view:
                    view.write(
                        json.dumps(training_view(row, negative_count, seed), ensure_ascii=False)
                        + "\n"
                    )
                n += 1
        finally:
            if view:
                view.close()
    if not n:
        raise ValueError("Empty behaviors file")
    result = {
        "rows": n,
        "source_split": split,
        "path": path.name,
        "sha256": sha256(path),
        "news_sha256": sha256(news_path),
        "behaviors_sha256": sha256(behaviors_path),
        "history": "complete",
        "evaluation_candidates": "complete_native",
        "dropped": 0,
        "truncated": 0,
    }
    if split == "train":
        result["training_views"] = {
            "path": views.name,
            "sha256": sha256(views),
            "negative_count": negative_count,
            "seed": seed,
        }
    atomic_json(out / "manifest.json", result)
    return result
