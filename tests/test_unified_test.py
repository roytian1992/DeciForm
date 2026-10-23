import json
import pytest
from deciform.joint_data import file_sha256
from deciform.evaluation import load_inventory, summarize_unified

def test_inventory_excludes_declared_rows_and_checks_hashes(tmp_path):
    rows = [{"id": str(i), "label": "a", "state": {"text": f"sample {i}"},
             "question": {"type": "choice", "options": ["a", "b"]}} for i in range(3)]
    stream = tmp_path/"records.jsonl"
    stream.write_text("".join(json.dumps(r)+"\n" for r in rows))
    manifest = {"status": "locked", "evaluation_only": True, "training_allowed": False,
        "checkpoint_selection_allowed": False, "record_count": 2,
        "unique_source_family_count": 1, "source_families": ["example"],
        "streams": [{"name": "one", "path": str(stream), "sha256": file_sha256(stream),
                     "count": 2, "raw_count": 3, "exclude_ids": ["1"], "source_families": ["example"]}]}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    _, loaded, counts = load_inventory(path)
    assert [r["id"] for r in loaded] == ["0", "2"]
    assert counts == {"example": 2}
    stream.write_text(stream.read_text()+" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inventory(path)

def test_aggregate_uses_each_record_once_and_pools_calibration():
    rows = [
        {"source_family": "x", "label": "a", "prediction": "a", "confidence": 0.9,
         "probabilities": {"a": 0.9, "b": 0.1}},
        {"source_family": "x", "label": "b", "prediction": "a", "confidence": 0.9,
         "probabilities": {"a": 0.9, "b": 0.1}},
        {"source_family": "y", "label": "a", "prediction": "a", "confidence": 0.9,
         "probabilities": {"a": 0.9, "b": 0.1}},
    ]
    result = summarize_unified(rows)
    assert result["record_count"] == 3
    assert result["aggregate"]["accuracy"] == pytest.approx(2/3)
    assert result["aggregate"]["brier"] == pytest.approx(1.66/3)
    assert result["aggregate"]["ece"] == pytest.approx(0.9-2/3)
    assert result["source_metrics"]["x"]["record_count"] == 2
