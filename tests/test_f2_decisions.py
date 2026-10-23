import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_f2_decisions.py"
spec = importlib.util.spec_from_file_location("generate_f2_decisions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_policy_configurations_are_disjoint_and_labels_reexecute() -> None:
    for family in module.FAMILIES:
        groups = module.split_configs(family)
        hashes = [{module.canonical_hash(rule) for rule in rules} for rules in groups.values()]
        assert all(hashes[left].isdisjoint(hashes[right])
                   for left in range(3) for right in range(left + 1, 3))
        for split, count in (("train", 30), ("dev", 30), ("transfer", 30)):
            rows = module.generate_family(family, split, count, 17, groups[split])
            assert len({row["state"]["text"] for row in rows}) == count
            assert set(row["label"] for row in rows) == set(module.OPTIONS[family])
            for row in rows:
                _, label = module.policy_and_label(family, row["provenance"]["rule_config"],
                                                   row["provenance"]["facts"])
                assert row["label"] == label
