import copy
import pytest
from deciform.runtime_settings import validate_runtime_migration, worker_resize_state


def manifests():
    original = {"config": {"name": "old", "learning_rate": 2e-5, "global_batch_size":128,
                           "physical_rows_per_gpu": 4},
                "world_size": 2, "plan_sha256": "pool",
                "code": {"scripts/train.py": "v1", "model.py": "fixed"}}
    current = copy.deepcopy(original)
    current["config"].update(name="new", checkpoint_layer_stride=2)
    current["code"]["scripts/train.py"] = "v2"
    return original, current


def test_runtime_change_records_continuation_without_discarding_optimizer():
    old, new = manifests()
    result = validate_runtime_migration(old, new)
    assert result["optimizer_restored"] and result["data_cursor_restored"]
    assert result["config_changes"]["checkpoint_layer_stride"]["after"] == 2


@pytest.mark.parametrize("field,value", [("learning_rate", 3e-5), ("global_batch_size", 256)])
def test_runtime_migration_rejects_training_changes(field, value):
    old, new = manifests()
    new["config"][field] = value
    with pytest.raises(ValueError, match="training settings"):
        validate_runtime_migration(old, new)


def test_runtime_migration_rejects_changed_model_and_pool():
    old, new = manifests()
    new["code"]["model.py"] = "changed"
    with pytest.raises(ValueError, match="source code"):
        validate_runtime_migration(old, new)
    old, new = manifests()
    new["plan_sha256"] = "new pool"
    with pytest.raises(ValueError, match="plan_sha256"):
        validate_runtime_migration(old, new)


def test_worker_resize_is_explicit_and_preserves_global_training_settings():
    old, new = manifests()
    new["world_size"] = 4
    with pytest.raises(ValueError, match="world_size"):
        validate_runtime_migration(old, new)
    result = validate_runtime_migration(old, new, allow_worker_resize=True)
    assert result["worker_resize"]["before"] == 2
    assert result["worker_resize"]["after"] == 4
    assert result["optimizer_restored"] and result["data_cursor_restored"]
    assert not result["bitwise_replay"]
    new["config"]["global_batch_size"] = 256
    with pytest.raises(ValueError, match="training settings"):
        validate_runtime_migration(old, new, allow_worker_resize=True)


def test_worker_resize_rejects_incompatible_batch_partition():
    old, new = manifests()
    new["world_size"] = 3
    with pytest.raises(ValueError, match="batches"):
        validate_runtime_migration(old, new, allow_worker_resize=True)


def test_resize_preserves_prefix_once_through_multiple_migrations():
    saved = {"world_size": 2, "source_rows_seen": {"a": 5, "b": 3, "c": 0},
             "sample_visits": 8, "optimizer_steps": 1, "epoch": 0, "cursor": 8,
             "plan_sha256": "pool"}
    workers = [{"counts": {"a": 3, "b": 1}, "trace": "old0"},
               {"counts": {"a": 2, "b": 2}, "trace": "old1"}]
    first = worker_resize_state(saved, workers, {}, 4, 42)
    assert first["coverage_prefix"] == saved["source_rows_seen"]
    assert len(set(first["worker_seeds"])) == 4
    assert first == worker_resize_state(saved, workers, {}, 4, 42)
    # Four new workers each visit a fresh row; the historical eight rows are
    # represented once globally, not once per worker.
    later = {**saved, "world_size": 4, "source_rows_seen": {"a": 5, "b": 7, "c": 0},
             "sample_visits": 12, "cursor": 12, "optimizer_steps": 2}
    later_workers = [{"counts": {"b": 1}, "trace": t} for t in first["worker_traces"]]
    second = worker_resize_state(later, later_workers, first["coverage_prefix"], 2, 42)
    assert second["coverage_prefix"] == later["source_rows_seen"]
    assert sum(second["coverage_prefix"].values()) == 12


def test_resize_rejects_missing_worker_or_corrupt_counts():
    saved = {"world_size": 2, "source_rows_seen": {"a": 8}, "sample_visits": 8,
             "optimizer_steps": 1, "epoch": 0, "cursor": 8, "plan_sha256": "pool"}
    workers = [{"counts": {"a": 4}, "trace": str(i)} for i in range(2)]
    with pytest.raises(ValueError, match="population"):
        worker_resize_state(saved, workers[:1], {}, 4, 42)
    with pytest.raises(ValueError, match="coverage"):
        worker_resize_state(saved, workers, {"a": 1}, 4, 42)
    with pytest.raises(ValueError, match="sample visits"):
        worker_resize_state({**saved, "sample_visits": 9}, workers, {}, 4, 42)
