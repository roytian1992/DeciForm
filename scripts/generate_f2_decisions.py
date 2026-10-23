"""Generate balanced, executable F2 decisions with disjoint policy configurations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


FAMILIES = ("eligibility", "refund", "priority")
OPTIONS = {
    "eligibility": ("eligible", "not_eligible"),
    "refund": ("approve_refund", "deny_refund"),
    "priority": ("low", "medium", "high"),
}
TEMPLATES = {
    0: "Policy: {policy}\nCase facts: {facts}",
    1: "Apply this rule to the case. Rule: {policy}\nRecorded facts: {facts}",
    2: "Decision review. Governing rule: {policy}\nEvidence: {facts}",
    3: "Policy memo: {policy}\nCase record: {facts}",
}
SPLIT_TEMPLATES = {"train": (0, 1), "dev": (2,), "transfer": (3,)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def configs(family: str) -> list[dict]:
    if family == "eligibility":
        return [dict(zip(("min_age", "min_income", "resident_required", "verified_required"), values))
                for values in itertools.product((18, 21, 25, 30), (0, 20000, 40000),
                                                (False, True), (False, True))]
    if family == "refund":
        return [dict(zip(("max_days", "sealed_required", "receipt_required", "channel"), values))
                for values in itertools.product((7, 14, 30, 45), (False, True),
                                                (False, True), ("any", "online", "store"))]
    if family == "priority":
        return [dict(zip(("high_threshold", "medium_threshold", "vip_policy", "incident_threshold"), values))
                for values in itertools.product((5, 6, 7, 8), (2, 3, 4),
                                                ("high", "bump", "none"), (3, 5))]
    raise ValueError(f"unknown family: {family}")


def policy_and_label(family: str, rule: dict, facts: dict) -> tuple[str, str]:
    if family == "eligibility":
        policy = (f"Approve eligibility when age is at least {rule['min_age']} and income is at least "
                  f"{rule['min_income']}; residency required={str(rule['resident_required']).lower()}; "
                  f"identity verification required={str(rule['verified_required']).lower()}.")
        passed = (facts["age"] >= rule["min_age"] and facts["income"] >= rule["min_income"]
                  and (not rule["resident_required"] or facts["resident"])
                  and (not rule["verified_required"] or facts["identity_verified"]))
        return policy, "eligible" if passed else "not_eligible"
    if family == "refund":
        policy = (f"Approve a refund within {rule['max_days']} days; sealed item required="
                  f"{str(rule['sealed_required']).lower()}; receipt required="
                  f"{str(rule['receipt_required']).lower()}; allowed channel={rule['channel']}.")
        passed = (facts["days_since_purchase"] <= rule["max_days"]
                  and (not rule["sealed_required"] or facts["item_sealed"])
                  and (not rule["receipt_required"] or facts["receipt_present"])
                  and (rule["channel"] == "any" or facts["purchase_channel"] == rule["channel"]))
        return policy, "approve_refund" if passed else "deny_refund"
    if family == "priority":
        policy = (f"High priority at urgency >= {rule['high_threshold']} or incident count >= "
                  f"{rule['incident_threshold']}; medium at urgency >= {rule['medium_threshold']}; "
                  f"VIP policy={rule['vip_policy']} (high gives high, bump gives at least medium).")
        if (facts["urgency"] >= rule["high_threshold"]
                or facts["incident_count"] >= rule["incident_threshold"]
                or (rule["vip_policy"] == "high" and facts["vip"])):
            return policy, "high"
        if facts["urgency"] >= rule["medium_threshold"] or (rule["vip_policy"] == "bump" and facts["vip"]):
            return policy, "medium"
        return policy, "low"
    raise ValueError(f"unknown family: {family}")


def sample_facts(family: str, rng: random.Random) -> dict:
    if family == "eligibility":
        return {"age": rng.randint(16, 65), "income": rng.randint(0, 80000),
                "resident": rng.choice((False, True)),
                "identity_verified": rng.choice((False, True))}
    if family == "refund":
        return {"days_since_purchase": rng.randint(1, 60),
                "item_sealed": rng.choice((False, True)),
                "receipt_present": rng.choice((False, True)),
                "purchase_channel": rng.choice(("online", "store"))}
    return {"urgency": rng.randint(0, 9), "vip": rng.choice((False, True)),
            "incident_count": rng.randint(0, 6)}


def split_configs(family: str) -> dict[str, list[dict]]:
    partitions = {split: [] for split in SPLIT_TEMPLATES}
    for rule in configs(family):
        bucket = int(canonical_hash({"family": family, "rule": rule})[:8], 16) % 10
        split = "train" if bucket < 7 else "dev" if bucket < 9 else "transfer"
        partitions[split].append(rule)
    if any(len(items) < 2 for items in partitions.values()):
        raise ValueError(f"policy partition is too sparse: {family}")
    return partitions


def generate_family(family: str, split: str, count: int, seed: int,
                    rules: list[dict]) -> list[dict]:
    options = OPTIONS[family]
    if count % len(options):
        raise ValueError("per-family count must be divisible by label count")
    quota = count // len(options)
    labels = Counter()
    records = []
    seen = set()
    rng = random.Random(f"{seed}:{family}:{split}")
    for attempt in range(count * 1000):
        rule = rules[rng.randrange(len(rules))]
        facts = sample_facts(family, rng)
        policy, label = policy_and_label(family, rule, facts)
        if labels[label] >= quota:
            continue
        template_id = rng.choice(SPLIT_TEMPLATES[split])
        key = canonical_hash({"rule": rule, "facts": facts, "template_id": template_id})
        if key in seen:
            continue
        seen.add(key)
        fact_text = "; ".join(f"{name}={str(value).lower()}" for name, value in facts.items())
        records.append({
            "id": f"f2_decisions_v2/{family}/{split}/{len(records):05d}",
            "source": "generated/f2_decisions_v2", "source_split": split, "split": split,
            "state": {"text": TEMPLATES[template_id].format(policy=policy, facts=fact_text)},
            "question": {"type": "choice",
                         "instructions": "Which decision follows from the policy and case facts?",
                         "options": list(options)},
            "label": label, "label_index": options.index(label),
            "provenance": {"label_source": "deterministic_program",
                           "rule_id": f"{family}_v2", "rule_spec_sha256": canonical_hash(rule),
                           "rule_config": rule, "facts": facts,
                           "generator_seed": seed, "template_id": template_id},
        })
        labels[label] += 1
        if len(records) == count:
            return records
    raise RuntimeError(f"failed to fill balanced {family}/{split}: {dict(labels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--train-per-family", type=int, default=600)
    parser.add_argument("--dev-per-family", type=int, default=180)
    parser.add_argument("--transfer-per-family", type=int, default=180)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    partitions = {family: split_configs(family) for family in FAMILIES}
    outputs = {}
    for split, count in (("train", args.train_per_family), ("dev", args.dev_per_family),
                         ("transfer", args.transfer_per_family)):
        rows = [row for family in FAMILIES for row in generate_family(
            family, split, count, args.seed, partitions[family][split])]
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        outputs[split] = {"path": str(path), "count": len(rows), "sha256": sha256(path),
                          "labels": {family: dict(Counter(row["label"] for row in rows
                                                         if row["provenance"]["rule_id"] == f"{family}_v2"))
                                     for family in FAMILIES}}
    manifest = {
        "version": 2, "dataset": "generated/f2_decisions_v2",
        "generator": "DeciForm/scripts/generate_f2_decisions.py",
        "seed": args.seed, "rule_families": list(FAMILIES),
        "policy_split": "SHA256(family, rule_config) mod 10: 0-6 train, 7-8 dev, 9 transfer",
        "template_split": {name: list(templates) for name, templates in SPLIT_TEMPLATES.items()},
        "config_counts": {family: {split: len(rules) for split, rules in groups.items()}
                          for family, groups in partitions.items()},
        "splits": outputs, "final_hidden": "not_created",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_root": str(output),
                      "splits": {name: info["count"] for name, info in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
