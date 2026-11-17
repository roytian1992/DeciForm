#!/usr/bin/env python3
"""Create a new allowlisted source tree without research data, runs or weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "README.md",
    "NOTICE",
    "AGENTS.md",
    ".gitignore",
    ".gitattributes",
    "environment.yml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-tested.txt",
    "src/deciform/__init__.py",
    "provenance/public_components.json",
    "provenance/public_verification.json",
]
DIRECTORIES = [
    "src/deciform/core",
    "configs/public",
    "examples/public",
    "tests/public",
    "docs/public",
    "provenance/data",
    "scripts/public",
]


def export(output):
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Export to a new directory outside the research checkout")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    candidates = [ROOT / p for p in FILES]
    candidates += [
        f
        for d in DIRECTORIES
        for f in (ROOT / d).rglob("*")
        if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo")
    ]
    candidates += [ROOT / ".github/workflows/public-tests.yml"]
    inventory = {}
    for source in sorted(set(candidates)):
        if source.is_symlink():
            raise ValueError("Source exports must not follow symlinks")
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[str(source.relative_to(ROOT))] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "files": inventory,
                "excluded": [
                    "data",
                    "runs",
                    "checkpoints",
                    "research scripts",
                    "server operation records",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"directory": str(output), "files": len(inventory)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    export(p.parse_args().output)
