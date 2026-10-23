import json
from pathlib import Path

import pytest

from deciform.data_protocol import Pool, build_episode, open_pools, raw_choice, scan_pools, select_visible, sha256, text_sha256


class WordTokenizer:
    def __call__(self, prompts, **kwargs):
        return {"input_ids": [list(range(len(prompt.split()))) for prompt in prompts]}


def make_row(index: int, label: str, split: str, text: str | None = None) -> dict:
    return {
        "id": f"civil/{split}/{index}", "source": "google/civil_comments",
        "source_split": split, "split": split,
        "state": {"text": text or f"Comment text {split} {index}"},
        "question": {"type": "choice", "instructions": "Is this toxic?",
                     "options": ["non_toxic", "toxic"]},
        "label": label, "label_index": int(label == "toxic"),
        "provenance": {"label_source": "native_test_fixture"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_full_episode_uses_all_eligible_train_and_separate_visible_roles(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    normalized = data_root / "typed_adapt_bench" / "normalized" / "civil_comments"
    train = [make_row(i, "toxic" if i % 2 else "non_toxic", "train") for i in range(20)]
    visible = [make_row(i, "toxic" if i % 2 else "non_toxic", "dev") for i in range(20)]
    visible[0] = make_row(0, "non_toxic", "dev", "word " * 80)
    visible[1] = make_row(1, "toxic", "dev", train[0]["state"]["text"])
    write_jsonl(normalized / "train.jsonl", train)
    write_jsonl(normalized / "dev.jsonl", visible)
    manifest = {"datasets": {"google/civil_comments": {"splits": {
        split: {"path": f"data/typed_adapt_bench/normalized/civil_comments/{split}.jsonl",
                "count": 20, "sha256": sha256(normalized / f"{split}.jsonl")}
        for split in ("train", "dev")
    }}}}
    manifest_path = data_root / "typed_adapt_bench" / "manifests" / "normalized_sources.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    entry = {"name": "civil_comments", "kind": "normalized",
             "source": "google/civil_comments", "train_split": "train", "visible_split": "dev"}
    episode = build_episode(entry, data_root, tmp_path / "episode", WordTokenizer(),
                            seed=7, max_prompt_tokens=30, target_per_role=2)
    assert episode["outputs"]["D_train"]["count"] == 19
    assert [episode["outputs"][role]["count"] for role in (
        "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"
    )] == [2, 2, 2, 2]
    assert episode["filtering"]["visible"]["too_long"] == 1
    assert episode["filtering"]["train"]["label_conflict"] == 1
    role_ids = []
    for role in ("D_train", "D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"):
        role_ids.extend(json.loads(line)["id"] for line in
                        (tmp_path / "episode" / f"{role}.jsonl").read_text().splitlines())
    assert len(role_ids) == len(set(role_ids))


def test_raw_boolq_preserves_question_and_binary_label() -> None:
    row = raw_choice({"passage": "Evidence", "question": "is this true", "answer": True},
                     source="google/boolq", name="boolq", index=3,
                     labels=["no", "yes"], revision="pinned")
    assert row["state"]["text"] == "Evidence"
    assert row["question"]["instructions"] == "Based on the passage, is this true?"
    assert row["label"] == "yes"


def test_raw_invalid_label_is_not_remapped() -> None:
    assert raw_choice({"label": -1, "premise": "a", "hypothesis": "b"},
                      source="nyu-mll/multi_nli", name="mnli", index=0,
                      labels=["entailment", "neutral", "contradiction"],
                      revision="pinned") is None


def test_source_hash_change_fails_before_writing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    normalized = data_root / "typed_adapt_bench" / "normalized" / "civil_comments"
    write_jsonl(normalized / "train.jsonl", [make_row(0, "non_toxic", "train")])
    write_jsonl(normalized / "dev.jsonl", [make_row(0, "non_toxic", "dev")])
    manifest_path = data_root / "typed_adapt_bench" / "manifests" / "normalized_sources.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"datasets": {"google/civil_comments": {"splits": {
        split: {"path": f"data/typed_adapt_bench/normalized/civil_comments/{split}.jsonl",
                "count": 1, "sha256": "wrong"} for split in ("train", "dev")
    }}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        build_episode({"name": "civil_comments", "kind": "normalized",
                       "source": "google/civil_comments", "train_split": "train",
                       "visible_split": "dev"}, data_root, tmp_path / "episode",
                      WordTokenizer(), seed=7, max_prompt_tokens=30, target_per_role=1)
    assert not (tmp_path / "episode").exists()


def test_raw_pool_preserves_at_least_eighty_percent_per_label() -> None:
    rows = [make_row(i, "non_toxic", "train") for i in range(86)]
    pool = Pool(count=len(rows), read=lambda: iter(rows),
                convert=lambda row, index: row, files={})
    selected, quotas = select_visible({"train": pool},
                                      {"train": bytearray([1] * len(rows))},
                                      seed=7, target_per_role=120, labels=["non_toxic"])
    assert quotas == {"non_toxic": 4}
    assert len(selected) == 16
    assert len(rows) - len(selected) >= 0.8 * len(rows)


def test_short_prompt_uses_conservative_byte_bound() -> None:
    class NoTokenizerCall:
        def __call__(self, prompts, **kwargs):
            raise AssertionError("short prompt should not need tokenization")

    rows = [make_row(1, "non_toxic", "train")]
    pool = Pool(count=1, read=lambda: iter(rows), convert=lambda row, index: row, files={})
    accepted, counts = scan_pools({"train": pool}, NoTokenizerCall(), 1024)
    assert list(accepted["train"]) == [1]
    assert counts["train"]["byte_bound_fast_path"] == 1


def test_long_prompt_stops_after_one_over_limit_candidate() -> None:
    class CountTokenizer(WordTokenizer):
        calls = []

        def __call__(self, prompts, **kwargs):
            self.calls.append(len(prompts))
            return super().__call__(prompts, **kwargs)

    row = make_row(1, "non_toxic", "train", "word " * 100)
    row["question"]["options"] = ["non_toxic", "toxic", "other"]
    tokenizer = CountTokenizer()
    pool = Pool(count=1, read=lambda: iter([row]), convert=lambda item, index: item, files={})
    accepted, counts = scan_pools({"train": pool}, tokenizer, 30)
    assert list(accepted["train"]) == [0]
    assert counts["train"]["too_long"] == 1
    assert tokenizer.calls == [1]


def test_over_limit_record_does_not_hide_later_eligible_duplicate() -> None:
    first = make_row(0, "non_toxic", "train", "same text")
    second = make_row(1, "non_toxic", "train", "same text")
    first["question"]["options"][1] = "word " * 100
    pool = Pool(count=2, read=lambda: iter([first, second]),
                convert=lambda item, index: item, files={})
    accepted, counts = scan_pools({"train": pool}, WordTokenizer(), 30)
    assert list(accepted["train"]) == [0, 1]
    assert counts["train"]["eligible"] == 1
    assert counts["train"]["too_long"] == 1


def test_cross_source_text_exclusion_precedes_visible_allocation() -> None:
    rows = [make_row(0, "non_toxic", "train"), make_row(1, "toxic", "train")]
    pool = Pool(count=2, read=lambda: iter(rows), convert=lambda item, index: item, files={})
    accepted, counts = scan_pools({"train": pool}, WordTokenizer(), 100,
                                  {text_sha256(rows[0])})
    assert list(accepted["train"]) == [0, 1]
    assert counts["train"]["cross_source_exclusion"] == 1


def test_sparse_visible_class_uses_one_per_role() -> None:
    rows = [make_row(i, "non_toxic", "dev") for i in range(7)]
    pool = Pool(count=len(rows), read=lambda: iter(rows),
                convert=lambda row, index: row, files={})
    selected, quotas = select_visible({"visible": pool},
                                      {"visible": bytearray([1] * len(rows))},
                                      seed=7, target_per_role=2, labels=["non_toxic"])
    assert quotas == {"non_toxic": 1}
    assert set(selected.values()) == {"D_mv", "D_cal_fit", "D_cal_risk", "D_assess_visible"}


def test_visible_label_gap_is_explicit() -> None:
    rows = [make_row(i, "non_toxic", "dev") for i in range(7)]
    pool = Pool(count=len(rows), read=lambda: iter(rows),
                convert=lambda row, index: row, files={})
    selected, quotas = select_visible({"visible": pool},
                                      {"visible": bytearray([1] * len(rows))},
                                      seed=7, target_per_role=2,
                                      labels=["non_toxic", "toxic"])
    assert len(selected) == 4
    assert quotas == {"non_toxic": 1, "toxic": 0}


def test_raw_civil_pool_only_reads_native_train(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "hf_fallback" / "google__civil_comments"
    train = root / "data" / "train-00000-of-00001.parquet"
    train.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["first", "second"], "toxicity": [0.2, 0.8],
                             "severe_toxicity": [0.0, 0.0], "obscene": [0.0, 0.0],
                             "threat": [0.0, 0.0], "insult": [0.0, 0.0],
                             "identity_attack": [0.0, 0.0], "sexual_explicit": [0.0, 0.0]}), train)
    manifest = {"revision": "pinned", "splits": {"train": {"rows": 2, "files": [
        {"path": "data/train-00000-of-00001.parquet", "sha256": sha256(train)}]}}}
    (tmp_path / "hf_fallback" / "google__civil_comments_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    pools, info = open_pools({"kind": "raw_parquet", "source": "google/civil_comments"}, tmp_path)
    rows = [pools["train"].convert(row, index)
            for index, row in enumerate(pools["train"].read())]
    assert [row["label"] for row in rows] == ["non_toxic", "toxic"]
    assert info["native_validation_and_test_accessed"] is False


def test_aegis_pool_keeps_human_labels_and_filters_aart(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_path = tmp_path / "aegis.json"
    rows = [
        {"id": "1", "prompt": "ordinary prompt", "prompt_label": "safe",
         "prompt_label_source": "human"},
        {"id": "2", "prompt": "AART prompt", "prompt_label": "unsafe",
         "prompt_label_source": "human"},
        {"id": "3", "prompt": "REDACTED", "prompt_label": "unsafe",
         "prompt_label_source": "human"},
    ]
    train_path.write_text(json.dumps(rows), encoding="utf-8")
    aart_path = tmp_path / "aart.parquet"
    pq.write_table(pa.table({"prompt": ["AART prompt"]}), aart_path)
    manifest_path = tmp_path / "kev_upstream" / "devtools_sources_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"sources": [
        {"group": "aegis", "filename": "train.json", "path": str(train_path),
         "expected_sha256": sha256(train_path), "revision": "pinned"},
        {"group": "aart_filter", "path": str(aart_path),
         "expected_sha256": sha256(aart_path)},
    ]}), encoding="utf-8")
    pools, info = open_pools({"kind": "aegis_json", "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"}, tmp_path)
    converted = [pools["train"].convert(row, index)
                 for index, row in enumerate(pools["train"].read())]
    assert [row["label"] if row else None for row in converted] == ["safe", None, None]
    assert info["native_validation_and_test_accessed"] is False


def test_generated_rules_keep_rule_families_separate(tmp_path: Path) -> None:
    root = tmp_path / "typed_adapt_bench" / "generated" / "f2_rules"
    root.mkdir(parents=True)
    splits = {}
    for split in ("train", "dev"):
        rows = []
        for family in ("eligibility_v1", "refund_v1"):
            for index in range(8):
                row = make_row(index, "non_toxic" if index % 2 else "toxic", split,
                               f"{family} {split} {index}")
                row["provenance"]["rule_id"] = family
                rows.append(row)
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        splits[split] = {"path": str(path), "sha256": sha256(path)}
    (root / "manifest.json").write_text(json.dumps({
        "dataset": "generated/f2_rules",
        "rule_families": ["eligibility", "refund"], "splits": splits,
    }), encoding="utf-8")
    entry = {"name": "f2_eligibility", "kind": "generated_rule",
             "source": "generated/f2_rules", "rule_id": "eligibility_v1"}
    pools, info = open_pools(entry, tmp_path)
    assert pools["train"].count == pools["visible"].count == 8
    assert info["transfer_split_accessed"] is False
    episode = build_episode(entry, tmp_path, tmp_path / "episode", WordTokenizer(),
                            seed=7, max_prompt_tokens=100, target_per_role=2)
    assert episode["outputs"]["D_train"]["count"] == 8
    assert episode["outputs"]["D_mv"]["count"] == 2
