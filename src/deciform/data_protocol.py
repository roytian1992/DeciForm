"""Build audited, full-pool Choice adaptation episodes from local sources."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from deciform.choice import render_choice_prompts


ROLES = ("D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible")
TREC_DEFINITIONS = (
    "an abbreviation or its expansion",
    "an entity or object other than a person or place",
    "a definition, explanation, manner, or reason",
    "a person or group of people",
    "a location or place",
    "a number, date, quantity, or measurement",
)
RAW_INSTRUCTIONS = {
    "CogComp/trec": "What type of answer does this question ask for?",
    "stanfordnlp/imdb": "What is the sentiment of this movie review?",
    "fancyzhx/ag_news": "What is the topic of this news article?",
    "SetFit/amazon_reviews_multi_en": "What is the review rating class, from 0 to 4?",
    "fancyzhx/dbpedia_14": "Which encyclopedia category describes this entry?",
    "SetFit/sst5": "What is the overall sentiment of this text?",
    "nyu-mll/multi_nli": "What is the relation between the premise and the hypothesis?",
    "Yelp/yelp_review_full": "What star rating does this review express?",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_input(row: dict) -> str:
    text = " ".join(row["state"]["text"].casefold().split())
    question = " ".join(row["question"]["instructions"].casefold().split())
    return hashlib.sha256(f"{text}\n{question}".encode("utf-8")).hexdigest()


def text_sha256(row: dict) -> str:
    text = " ".join(row["state"]["text"].casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_choice(row: dict, *, source: str, name: str, index: int,
               labels: list[str], revision: str) -> dict | None:
    if source == "google/boolq":
        text = row["passage"]
        instructions = f"Based on the passage, {row['question'].strip()}?"
        options = ["no", "yes"]
        label_index = int(row["answer"])
    else:
        label_index = int(row["coarse_label"] if source == "CogComp/trec" else row["label"])
        if not 0 <= label_index < len(labels):
            return None
        options = labels
        instructions = RAW_INSTRUCTIONS[source]
        if source == "fancyzhx/dbpedia_14":
            text = f"{row['title'].strip()}\n{row['content'].strip()}"
        elif source == "nyu-mll/multi_nli":
            text = f"Premise: {row['premise'].strip()}\nHypothesis: {row['hypothesis'].strip()}"
        else:
            text = row["text"]
    question = {"type": "choice", "instructions": instructions, "options": options}
    if source == "CogComp/trec":
        question["option_definitions"] = list(TREC_DEFINITIONS)
    return {
        "id": f"{name}/train/{index:07d}", "source": source,
        "source_split": "train", "split": "unassigned",
        "state": {"text": text.strip()}, "question": question,
        "label": options[label_index], "label_index": label_index,
        "provenance": {"label_source": "native_label", "raw_row_index": index,
                       "source_revision": revision},
    }


@dataclass
class Pool:
    count: int
    read: Callable[[], Iterator[dict]]
    convert: Callable[[dict, int], dict | None]
    files: dict[str, str]


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _parquet_rows(paths: list[Path], columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as parquet

    for path in paths:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=8192, columns=columns):
            yield from batch.to_pylist()


def _raw_parquet_pool(entry: dict, data_root: Path) -> tuple[dict[str, Pool], dict]:
    import pyarrow.parquet as parquet

    source = entry["source"]
    if source == "google/civil_comments":
        root = data_root / "hf_fallback" / "google__civil_comments"
        manifest_path = data_root / "hf_fallback" / "google__civil_comments_manifest.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [root / item["path"] for item in source_manifest["splits"]["train"]["files"]]
        expected = {str(root / item["path"]): item["sha256"]
                    for item in source_manifest["splits"]["train"]["files"]}
        columns = ["text", "toxicity", "severe_toxicity", "obscene", "threat",
                   "insult", "identity_attack", "sexual_explicit"]
        labels = ["non_toxic", "toxic"]

        def convert(row: dict, index: int) -> dict | None:
            value = row.get("toxicity")
            if value is None or not math.isfinite(value):
                return None
            label_index = int(value >= 0.5)
            return {
                "id": f"civil_comments/train/{index:07d}", "source": source,
                "source_split": "train", "split": "unassigned",
                "state": {"text": (row.get("text") or "").strip()},
                "question": {"type": "choice", "instructions":
                             "Is this comment toxic under the selected toxicity threshold?",
                             "options": labels},
                "label": labels[label_index], "label_index": label_index,
                "provenance": {"label_source": "derived_threshold", "target_column": "toxicity",
                               "target_score": value, "threshold": 0.5,
                               "raw_row_index": index, "source_revision": source_manifest["revision"]},
            }

    elif source == "davidheineman/consumer-finance-complaints-large":
        root = data_root / "kev_upstream" / "raw_hf" / "davidheineman__consumer-finance-complaints-large"
        manifest_path = data_root / "kev_upstream" / "manual_sources.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = next(item for item in source_manifest["items"]
                    if item["id"] == "cfpb_consumer_finance_complaints_raw")
        files = sorted((root / "data").glob("train-*.parquet"))
        if len(files) != item["parquet_files"]:
            raise ValueError("CFPB raw shard count changed")
        expected = {}
        columns = ["product", "complaint_what_happened", "complaint_id",
                   "company", "state", "submitted_via", "date_received"]
        labels = sorted({value for row in _parquet_rows(files, ["product"])
                         if (value := row["product"])})

        def convert(row: dict, index: int) -> dict | None:
            product = row.get("product")
            complaint_id = row.get("complaint_id")
            if not product or not complaint_id:
                return None
            return {
                "id": f"cfpb/train/{complaint_id}", "source": source,
                "source_split": "train", "split": "unassigned",
                "state": {"text": (row.get("complaint_what_happened") or "").strip()},
                "question": {"type": "choice", "instructions":
                             "Which consumer-finance product best matches this complaint?",
                             "options": labels},
                "label": product, "label_index": labels.index(product),
                "provenance": {"label_source": "native_product_field",
                               "complaint_id": complaint_id, "company": row.get("company"),
                               "state": row.get("state"), "submitted_via": row.get("submitted_via"),
                               "date_received": row.get("date_received"),
                               "raw_row_index": index, "source_revision": item["revision"]},
            }

    else:
        raise ValueError(f"unregistered raw Parquet source: {source}")
    file_hashes = {str(path): sha256(path) for path in files}
    for filename, digest in expected.items():
        if file_hashes[filename] != digest:
            raise ValueError(f"raw source hash changed: {filename}")
    count = sum(parquet.ParquetFile(path).metadata.num_rows for path in files)
    if source == "google/civil_comments" and count != source_manifest["splits"]["train"]["rows"]:
        raise ValueError("Civil Comments raw train count changed")
    if source != "google/civil_comments" and count != item["rows_verified"]:
        raise ValueError("CFPB raw train count changed")
    pool = Pool(count=count, read=lambda: _parquet_rows(files, columns),
                convert=convert, files=file_hashes)
    return {"train": pool}, {"manifest": str(manifest_path),
                             "manifest_sha256": sha256(manifest_path),
                             "source_revision": source_manifest["revision"] if source == "google/civil_comments" else item["revision"],
                             "labels": labels,
                             "native_validation_and_test_accessed": False}


def _aegis_pool(entry: dict, data_root: Path) -> tuple[dict[str, Pool], dict]:
    from pyarrow.parquet import read_table

    manifest_path = data_root / "kev_upstream" / "devtools_sources_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = source_manifest["sources"]
    aegis_item = next(item for item in sources if item["group"] == "aegis" and item["filename"] == "train.json")
    aart_item = next(item for item in sources if item["group"] == "aart_filter")
    train_path = Path(aegis_item["path"])
    aart_path = Path(aart_item["path"])
    if sha256(train_path) != aegis_item["expected_sha256"] or sha256(aart_path) != aart_item["expected_sha256"]:
        raise ValueError("Aegis source or AART filter hash changed")
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    aart_prompts = {" ".join(text.casefold().split()) for text in
                    read_table(aart_path, columns=["prompt"])["prompt"].to_pylist() if text}
    labels = ["safe", "unsafe"]

    def convert(row: dict, index: int) -> dict | None:
        prompt = (row.get("prompt") or "").strip()
        label = row.get("prompt_label")
        if (not prompt or prompt == "REDACTED" or row.get("reconstruction_id_if_redacted")
                or label not in labels
                or row.get("prompt_label_source") != "human"
                or " ".join(prompt.casefold().split()) in aart_prompts):
            return None
        return {
            "id": f"aegis/train/{row['id']}", "source": entry["source"],
            "source_split": "train", "split": "unassigned",
            "state": {"text": prompt},
            "question": {"type": "choice", "instructions":
                         "Is this prompt safe under the content safety policy?",
                         "options": labels},
            "label": label, "label_index": labels.index(label),
            "provenance": {"label_source": "human_prompt_label", "raw_row_index": index,
                           "source_revision": aegis_item["revision"]},
        }

    pool = Pool(count=len(rows), read=lambda: iter(rows), convert=convert,
                files={str(train_path): aegis_item["expected_sha256"],
                       str(aart_path): aart_item["expected_sha256"]})
    return {"train": pool}, {"manifest": str(manifest_path),
                             "manifest_sha256": sha256(manifest_path),
                             "source_revision": aegis_item["revision"],
                             "labels": labels, "native_validation_and_test_accessed": False,
                             "excluded_aart_prompt_count": len(aart_prompts)}


def _generated_rule_pools(entry: dict, data_root: Path) -> tuple[dict[str, Pool], dict]:
    if entry["source"] not in {"generated/f2_rules", "generated/f2_decisions_v2"}:
        raise ValueError("unknown generated source")
    rule_id = entry["rule_id"]
    root = data_root / "typed_adapt_bench" / "generated" / entry.get("dataset_dir", "f2_rules")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family = rule_id.rsplit("_v", 1)[0]
    if manifest["dataset"] != entry["source"] or family not in manifest["rule_families"]:
        raise ValueError(f"unknown generated rule family: {rule_id}")
    pools = {}
    for pool_name, split in (("train", "train"), ("visible", "dev")):
        info = manifest["splits"][split]
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise ValueError(f"generated rule source hash changed: {path}")

        def read(path=path, rule_id=rule_id):
            return (row for row in _jsonl(path)
                    if row["provenance"]["rule_id"] == rule_id)

        count = sum(1 for _ in read())
        if count == 0:
            raise ValueError(f"empty generated rule family: {rule_id}/{split}")
        pools[pool_name] = Pool(count=count, read=read,
                                convert=lambda row, index: row,
                                files={str(path): info["sha256"]})
    return pools, {"manifest": str(manifest_path),
                   "manifest_sha256": sha256(manifest_path),
                   "rule_id": rule_id, "transfer_split_accessed": False}


def open_pools(entry: dict, data_root: Path) -> tuple[dict[str, Pool], dict]:
    source = entry["source"]
    if entry["kind"] == "raw_parquet":
        return _raw_parquet_pool(entry, data_root)
    if entry["kind"] == "aegis_json":
        return _aegis_pool(entry, data_root)
    if entry["kind"] == "generated_rule":
        return _generated_rule_pools(entry, data_root)
    if entry["kind"] == "hf":
        from datasets import load_from_disk

        manifest_path = data_root / "kev_upstream" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches = [item for item in manifest["datasets"] if item["repo"] == source]
        if len(matches) != 1:
            raise ValueError(f"source absent or duplicated in upstream manifest: {source}")
        info = matches[0]
        train_dir = Path(info["output_path"]) / "train"
        dataset = load_from_disk(str(train_dir))
        if len(dataset) != info["splits"]["train"]:
            raise ValueError(f"raw train count changed: {source}")
        if source == "google/boolq":
            labels = ["no", "yes"]
        elif source == "SetFit/amazon_reviews_multi_en":
            labels = [str(index) for index in range(5)]
        elif source == "SetFit/sst5":
            labels = ["very negative", "negative", "neutral", "positive", "very positive"]
        else:
            column = "coarse_label" if source == "CogComp/trec" else "label"
            labels = list(dataset.features[column].names)
        arrows = sorted(train_dir.glob("*.arrow"))
        if not arrows:
            raise FileNotFoundError(f"no pinned Arrow train files: {train_dir}")
        pool = Pool(
            count=len(dataset), read=lambda: iter(dataset),
            convert=lambda row, index: raw_choice(
                row, source=source, name=entry["name"], index=index,
                labels=labels, revision=info["revision"]),
            files={str(path): sha256(path) for path in arrows},
        )
        return {"train": pool}, {"manifest": str(manifest_path),
                                 "manifest_sha256": sha256(manifest_path),
                                 "source_revision": info["revision"], "labels": labels}

    manifest_path = data_root / "typed_adapt_bench" / "manifests" / "normalized_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest["datasets"][source]
    pools = {}
    for pool_name, split in (("train", entry["train_split"]),
                             ("visible", entry["visible_split"])):
        split_info = info["splits"][split]
        path = data_root / "typed_adapt_bench" / Path(split_info["path"]).relative_to("data/typed_adapt_bench")
        observed = sha256(path)
        if observed != split_info["sha256"]:
            raise ValueError(f"normalized source hash changed: {path}")
        pools[pool_name] = Pool(
            count=split_info["count"], read=lambda path=path: _jsonl(path),
            convert=lambda row, index: row,
            files={str(path): observed},
        )
    return pools, {"manifest": str(manifest_path),
                   "manifest_sha256": sha256(manifest_path)}


def scan_pools(pools: dict[str, Pool], tokenizer, max_prompt_tokens: int | None,
               excluded_text_hashes: set[str] | None = None
               ) -> tuple[dict[str, bytearray], dict[str, Counter]]:
    accepted = {name: bytearray(pool.count) for name, pool in pools.items()}
    counts = {name: Counter() for name in pools}
    seen: dict[str, tuple[str, int, str] | None] = {}
    exclusions = excluded_text_hashes or set()
    special_tokens = max(16, tokenizer.num_special_tokens_to_add(pair=False)) if (
        max_prompt_tokens is not None and hasattr(tokenizer, "num_special_tokens_to_add")
    ) else 16

    def process_batch(name: str, batch: list[tuple[int, dict]], pool: Pool) -> None:
        rows: dict[int, dict] = {}
        eligible: dict[int, bool] = {}
        long_indices = []
        long_prompts = []
        for index, native in batch:
            row = pool.convert(native, index)
            if row is None:
                counts[name]["invalid_label"] += 1
                continue
            if not row["state"]["text"] or not row["question"]["instructions"]:
                counts[name]["empty"] += 1
                continue
            if exclusions and text_sha256(row) in exclusions:
                counts[name]["cross_source_exclusion"] += 1
                continue
            option_ids, prompts = render_choice_prompts(row)
            if row["label"] not in option_ids:
                counts[name]["invalid_label"] += 1
                continue
            rows[index] = row
            if max_prompt_tokens is None:
                eligible[index] = True
                counts[name]["unbounded"] += 1
                continue
            longest = max(prompts, key=lambda prompt: len(prompt.encode("utf-8")))
            if len(longest.encode("utf-8")) + special_tokens <= max_prompt_tokens:
                eligible[index] = True
                counts[name]["byte_bound_fast_path"] += 1
                if index % 1000 == 999:
                    lengths = tokenizer(prompts, truncation=False,
                                        add_special_tokens=True)["input_ids"]
                    if max(map(len, lengths)) > max_prompt_tokens:
                        raise RuntimeError("byte-bound token check failed")
            else:
                long_indices.append(index)
                long_prompts.append(longest)

        if long_prompts:
            lengths = tokenizer(long_prompts, truncation=False,
                                add_special_tokens=True)["input_ids"]
            near_indices = []
            all_prompts = []
            for index, tokens in zip(long_indices, lengths, strict=True):
                if len(tokens) > max_prompt_tokens:
                    counts[name]["too_long"] += 1
                else:
                    near_indices.append(index)
                    _, prompts = render_choice_prompts(rows[index])
                    all_prompts.extend(prompts)
            if all_prompts:
                all_lengths = tokenizer(all_prompts, truncation=False,
                                        add_special_tokens=True)["input_ids"]
                cursor = 0
                for index in near_indices:
                    option_count = len(rows[index]["question"]["options"])
                    eligible[index] = max(map(len, all_lengths[cursor:cursor + option_count])) <= max_prompt_tokens
                    if not eligible[index]:
                        counts[name]["too_long"] += 1
                    cursor += option_count

        for index, _ in batch:
            if not eligible.get(index, False):
                continue
            row = rows[index]
            key = canonical_input(row)
            if key in seen:
                previous = seen[key]
                if previous is not None and previous[2] != row["label"]:
                    old_name, old_index, _ = previous
                    if accepted[old_name][old_index]:
                        accepted[old_name][old_index] = 0
                        counts[old_name]["eligible"] -= 1
                        counts[old_name]["label_conflict"] += 1
                    seen[key] = None
                    counts[name]["label_conflict"] += 1
                elif previous is None:
                    counts[name]["label_conflict"] += 1
                else:
                    counts[name]["duplicate"] += 1
                continue
            seen[key] = (name, index, row["label"])
            accepted[name][index] = 1
            counts[name]["eligible"] += 1

    for name, pool in pools.items():
        batch = []
        for index, native in enumerate(pool.read()):
            if index >= pool.count:
                raise ValueError(f"{name} has more records than its manifest")
            counts[name]["raw"] += 1
            batch.append((index, native))
            if len(batch) == 64:
                process_batch(name, batch, pool)
                batch.clear()
                if counts[name]["raw"] % 100000 < 64:
                    print(f"scanned {name}: raw={counts[name]['raw']} eligible={counts[name]['eligible']}", flush=True)
        if batch:
            process_batch(name, batch, pool)
        if counts[name]["raw"] != pool.count:
            raise ValueError(f"{name} has fewer records than its manifest")
    return accepted, counts


def select_visible(pools: dict[str, Pool], accepted: dict[str, bytearray],
                   seed: int, target_per_role: int, labels: list[str]
                   ) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    visible_pool = "visible" if "visible" in pools else "train"
    quota = max(1, math.ceil(target_per_role / len(labels)))
    keep = quota * len(ROLES)
    heaps: dict[str, list[tuple[int, int]]] = defaultdict(list)
    available: Counter = Counter()
    pool = pools[visible_pool]
    for index, native in enumerate(pool.read()):
        if not accepted[visible_pool][index]:
            continue
        row = pool.convert(native, index)
        available[row["label"]] += 1
        score = int.from_bytes(hashlib.sha256(
            f"{seed}:{row['id']}".encode("utf-8")).digest()[:8], "big")
        heap = heaps[row["label"]]
        candidate = (-score, index)
        if len(heap) < keep:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    selected = {}
    quotas = {}
    for label in labels:
        heap = heaps[label]
        label_quota = quota
        if visible_pool == "train":
            label_quota = min(quota, int(available[label] * 0.2) // len(ROLES))
        else:
            label_quota = min(quota, available[label] // len(ROLES))
        quotas[label] = label_quota
        if label_quota == 0:
            continue
        if len(heap) < label_quota * len(ROLES):
            raise ValueError(f"{label}: insufficient eligible rows for separate visible roles")
        ordered = [index for _, index in sorted(heap, reverse=True)]
        for role_index, role in enumerate(ROLES):
            for index in ordered[role_index * label_quota:(role_index + 1) * label_quota]:
                selected[(visible_pool, index)] = role
    if not selected:
        raise ValueError("no eligible visible records could be allocated")
    return selected, quotas


def build_episode(entry: dict, data_root: Path, output_dir: Path, tokenizer,
                  *, seed: int, max_prompt_tokens: int | None, target_per_role: int) -> dict:
    pools, source_info = open_pools(entry, data_root)
    exclusions = set(entry.get("exclude_text_hashes", []))
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in exclusions):
        raise ValueError("invalid cross-source exclusion hash")
    accepted, filtering = scan_pools(pools, tokenizer, max_prompt_tokens, exclusions)
    labels = source_info.get("labels")
    if labels is None:
        first = next((pools["train"].convert(row, index)
                      for index, row in enumerate(pools["train"].read())
                      if accepted["train"][index]), None)
        if first is None:
            raise ValueError("no eligible training records")
        labels = first["question"]["options"]
    selected, quotas = select_visible(pools, accepted, seed, target_per_role, labels)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {role: output_dir / f"{role}.jsonl" for role in ("D_train", *ROLES)}
    output_counts = {role: Counter() for role in paths}
    handles = {role: path.open("x", encoding="utf-8") for role, path in paths.items()}
    try:
        for pool_name, pool in pools.items():
            for index, native in enumerate(pool.read()):
                if not accepted[pool_name][index]:
                    continue
                role = selected.get((pool_name, index))
                if role is None and pool_name == "train":
                    role = "D_train"
                if role is None:
                    continue
                row = {**pool.convert(native, index), "split": role}
                handles[role].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output_counts[role][row["label"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if not output_counts["D_train"]:
        raise ValueError("no training rows after visible holdout allocation")
    outputs = {
        role: {"path": str(path), "count": sum(output_counts[role].values()),
               "sha256": sha256(path), "labels": dict(sorted(output_counts[role].items()))}
        for role, path in paths.items()
    }
    manifest = {
        "version": 2, "status": "complete", "episode": f"{entry['name']}_full_choice_v1",
        "source": entry["source"], "source_kind": entry["kind"],
        "initialization_overlap": bool(entry.get("initialization_overlap", False)),
        "question_type": "choice", "option_count": len(labels), "class_count": len(labels),
        "seed": seed, "max_prompt_tokens": max_prompt_tokens,
        "evaluation_target_per_role": target_per_role,
        "visible_allocation": "SHA256-smallest IDs per class; at most 20% of each train-pool label held out",
        "visible_quota_per_label_per_role": quotas,
        "visible_missing_labels": [label for label in labels if quotas[label] == 0],
        "train_missing_labels": [label for label in labels if not output_counts["D_train"][label]],
        "cross_source_excluded_text_sha256": sorted(exclusions),
        "source_info": source_info,
        "inputs": {name: {"count": pool.count, "files": pool.files}
                   for name, pool in pools.items()},
        "filtering": {name: dict(counts) for name, counts in filtering.items()},
        "outputs": outputs,
        "native_test_and_validation_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
