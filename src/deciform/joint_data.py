"""Audited, indexed access to every row of the joint P_train pool."""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def index_source(source: str, episode_dir: Path, index_dir: Path) -> dict:
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["outputs"]["D_train"]
    path = episode_dir / "D_train.jsonl"
    index_path = index_dir / f"{source}.npy"
    offsets = np.lib.format.open_memmap(index_path, mode="w+", dtype="uint64",
                                        shape=(expected["count"],))
    digest = hashlib.sha256()
    offset = 0
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                if count >= expected["count"]:
                    raise ValueError(f"{source}: more rows than manifest")
                offsets[count] = offset
                count += 1
            offset += len(line)
    offsets.flush()
    if count != expected["count"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"{source}: row count or SHA256 mismatch")
    return {"name": source, "path": str(path.resolve()), "count": count,
            "sha256": digest.hexdigest(), "index": str(index_path.resolve()),
            "index_sha256": file_sha256(index_path), "bytes": offset,
            "episode_manifest": str(manifest_path.resolve()),
            "episode_manifest_sha256": file_sha256(manifest_path),
            "max_prompt_tokens": (
                int(manifest["max_prompt_tokens"])
                if manifest.get("max_prompt_tokens") is not None else None
            ),
            "option_count": manifest["option_count"]}


class JointRows:
    """One deterministic global permutation, sharded without dropping a tail."""
    def __init__(self, plan: dict):
        self.sources = plan["sources"]
        self.ends = np.cumsum([s["count"] for s in self.sources]).tolist()
        self.total = self.ends[-1]
        if self.total != plan["total_rows"]:
            raise ValueError("joint plan count mismatch")
        self.offsets = []
        self.handles = []
        for source in self.sources:
            path = Path(source["path"])
            index = Path(source["index"])
            if path.stat().st_size != source["bytes"] or file_sha256(index) != source["index_sha256"]:
                raise ValueError(f"joint source changed: {source['name']}")
            self.offsets.append(np.load(index, mmap_mode="r"))
            self.handles.append(path.open("rb"))

    def get(self, index: int) -> tuple[str, dict]:
        if not 0 <= index < self.total:
            raise IndexError(index)
        source_index = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[source_index - 1] if source_index else 0)
        handle = self.handles[source_index]
        handle.seek(int(self.offsets[source_index][local]))
        row = json.loads(handle.readline())
        return self.sources[source_index]["name"], row

    def permutation(self, seed: int, epoch: int):
        return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(self.total)

    def close(self):
        for handle in self.handles:
            handle.close()


def shard_batch(order, cursor: int, global_batch: int, rank: int, world_size: int):
    if global_batch % world_size or not 0 <= rank < world_size:
        raise ValueError("global batch must divide evenly across workers")
    stop = min(cursor + global_batch, len(order))
    return [int(order[p]) if p < stop else None
            for p in range(cursor + rank, cursor + global_batch, world_size)]
