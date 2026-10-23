import hashlib
import json

from deciform.joint_data import JointRows, index_source, shard_batch


def test_joint_shards_cover_every_source_once_and_resume_without_overlap(tmp_path):
    sources = []
    indices = tmp_path / "indices"
    indices.mkdir()
    for name, count in [("one", 7), ("two", 12)]:
        episode = tmp_path / name
        episode.mkdir()
        payload = "".join(json.dumps({"id": f"{name}-{i}"}) + "\n" for i in range(count))
        (episode / "D_train.jsonl").write_text(payload)
        (episode / "manifest.json").write_text(json.dumps({
            "outputs": {"D_train": {"count": count, "sha256": hashlib.sha256(payload.encode()).hexdigest()}},
            "max_prompt_tokens": 1024, "option_count": 2,
        }))
        sources.append(index_source(name, episode, indices))
    data = JointRows({"sources": sources, "total_rows": 19})
    order = data.permutation(42, 0)
    visited = []
    for cursor in range(0, 19, 8):
        for rank in range(4):
            visited += [i for i in shard_batch(order, cursor, 8, rank, 4) if i is not None]
    assert sorted(visited) == list(range(19))
    assert len({data.get(i)[1]["id"] for i in visited}) == 19
    prefix = set(order[:8])
    resumed = {i for cursor in range(8, 19, 8) for rank in range(4)
               for i in shard_batch(order, cursor, 8, rank, 4) if i is not None}
    assert not prefix & resumed and len(prefix | resumed) == 19
    data.close()


def test_two_to_four_workers_preserves_global_sequence_and_partial_final_batch():
    order = list(range(29))[::-1]
    visited = []
    for cursor in range(0, len(order), 8):
        world = 2 if cursor < 16 else 4
        shards = [shard_batch(order, cursor, 8, rank, world) for rank in range(world)]
        global_rows = [value for row in zip(*shards, strict=True) for value in row if value is not None]
        assert global_rows == order[cursor:cursor + 8]
        visited.extend(global_rows)
    assert visited == order
