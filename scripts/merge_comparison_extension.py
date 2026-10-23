"""Merge new shared-source evaluations into preserved three-model reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

METRICS = ("accuracy", "macro_f1", "brier", "ece", "nll")
COMPARISON_SCOPE = "exploratory_development_extension_not_locked_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_report(report: dict) -> dict:
    groups = defaultdict(list)
    families = set()
    for source, metrics in report["source_metrics"].items():
        family = metrics.get("source_family", source)
        # The DBpedia development manifest uses source IDs suffixed by panel
        # (dbpedia14__natural / dbpedia14__balanced). Canonicalize those back
        # to one dataset family before report-wide distinct-source counting.
        panel = metrics.get("panel")
        suffix = f"__{panel}" if panel in {"natural", "balanced"} else None
        if suffix and family.endswith(suffix):
            family = family[:-len(suffix)]
        metrics["source_family"] = family
        families.add(family)
        groups[metrics["panel"]].append((family, metrics))
    panel_metrics = {}
    panel_counts = {}
    for panel, entries in groups.items():
        panel_metrics[panel] = {
            "source_count": len({family for family, _ in entries}),
            "unique_source_count": len({family for family, _ in entries}),
            "source_panel_entry_count": len(entries),
            "record_count": sum(item["record_count"] for _, item in entries),
            "macro": {metric: sum(item[metric] for _, item in entries) / len(entries)
                      for metric in METRICS},
        }
        panel_counts[panel] = {key: panel_metrics[panel][key]
                              for key in ("source_count", "unique_source_count",
                                          "source_panel_entry_count", "record_count")}
    return {
        **report,
        "comparison_scope": COMPARISON_SCOPE,
        "source_count": len(families),
        "unique_source_count": len(families),
        "source_panel_entry_count": len(report["source_metrics"]),
        "record_count": sum(item["record_count"] for item in report["source_metrics"].values()),
        "panel_metrics": panel_metrics,
        "panel_counts": panel_counts,
        "count_semantics": {
            "source_count": "unique source families across this report",
            "source_panel_entry_count": "source-family-by-panel metric entries",
            "record_count": "sum of rows across disjoint source-panel entries",
            "panel_counts": "source-family and record totals within each panel",
        },
    }


def main(args) -> None:
    model_inputs = {
        "RSI-Static-Full": (args.static_base, args.static_extension),
        "Open-JEV": (args.openjev_base, args.openjev_extension),
        "KEV-9B": (args.kev_base, args.kev_extension),
    }
    loaded = {}
    for name, (base_path, extension_path) in model_inputs.items():
        base, extension = json.loads(base_path.read_text()), json.loads(extension_path.read_text())
        if base.get("status") != "complete" or extension.get("status") != "complete":
            raise ValueError(f"incomplete component report for {name}")
        base_sources = base["source_metrics"]
        extra_sources = extension["source_metrics"]
        duplicate = set(base_sources) & set(extra_sources)
        if duplicate:
            raise ValueError(f"source metric collision for {name}: {sorted(duplicate)}")
        if name == "Open-JEV" and extension.get("model") != "open_jev":
            raise ValueError("Open-JEV extension report has the wrong model id")
        if name == "KEV-9B" and extension.get("model") != "kev":
            raise ValueError("KEV extension report has the wrong model id")
        for key in ("checkpoint", "adapter_sha256", "head_sha256"):
            if base.get(key) and extension.get(key) and base[key] != extension[key]:
                raise ValueError(f"checkpoint identity changed for {name}: {key}")
        merged_sources = {**base_sources, **extra_sources}
        loaded[name] = summarize_report({
            **base,
            "source_metrics": merged_sources,
            "component_reports": [
                {"path": str(base_path.resolve()), "sha256": sha256(base_path)},
                {"path": str(extension_path.resolve()), "sha256": sha256(extension_path)},
            ],
        })

    source_maps = [report["source_metrics"] for report in loaded.values()]
    names = set(source_maps[0])
    if any(set(mapping) != names for mapping in source_maps[1:]):
        raise ValueError("the three models do not cover identical source-panel entries")
    for source in names:
        signatures = {(mapping[source]["panel"], mapping[source]["input_sha256"])
                      for mapping in source_maps}
        if len(signatures) != 1:
            raise ValueError(f"models were evaluated on different rows for {source}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    report_paths = {}
    for name, report in loaded.items():
        safe = name.lower().replace("-", "_")
        path = output_dir / f"{safe}_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        report_paths[name] = path

    panel_names = sorted(loaded["RSI-Static-Full"]["panel_metrics"])
    comparison_panels = {}
    for panel in panel_names:
        static = loaded["RSI-Static-Full"]["panel_metrics"][panel]["macro"]
        comparison_panels[panel] = {
            name: {
                "metrics": report["panel_metrics"][panel]["macro"],
                "delta_vs_RSI_Static_Full": {
                    metric: report["panel_metrics"][panel]["macro"][metric] - static[metric]
                    for metric in METRICS
                },
            }
            for name, report in loaded.items()
        }
    static = loaded["RSI-Static-Full"]
    summary = {
        "status": "complete",
        "comparison_scope": COMPARISON_SCOPE,
        "aggregation": (
            "Within each panel, reported macro metrics are unweighted means across source families. "
            "DBpedia14 inclusion changes natural/balanced denominators from 9 to 10; "
            "compare the extension macro only with this caveat, and retain the prior 9-family report."
        ),
        "interpretation_limits": [
            "This is a development-monitoring extension, not the locked final benchmark result.",
            "The original 2026-10-20 comparison is preserved unchanged as the prior 9-family natural/balanced result.",
            "DBpedia14 was absent from our 18-family training plan; its extension panels are source-family-held-out for RSI-Static-Full, but they are now development-evaluated and should not be presented as untouched final test data.",
            "KEV-9B's model card lists DBpedia14 among training datasets; Open-JEV family-level exposure is unverified.",
            "The DBpedia split builder read source labels to create balanced and complementary partitions; the sealed remainder was not passed to model evaluators or used for model selection."
        ],
        "extension_source": "DBpedia14 official test",
        "source_count": static["unique_source_count"],
        "unique_source_count": static["unique_source_count"],
        "source_panel_entry_count": static["source_panel_entry_count"],
        "record_count": static["record_count"],
        "panel_counts": static["panel_counts"],
        "count_semantics": static["count_semantics"],
        "source_exposure": {
            "dbpedia14": {
                "RSI-Static-Full": "559,477 DBpedia14 D_train rows were prepared in the 12-source Choice protocol, but DBpedia14 was omitted from both the original 10-family Static-Full plan and the expanded 18-family continuation; both completed reports record 0 DBpedia14 training visits",
                "Open-JEV": "checkpoint provenance names release-v1 and reasoning-control-v1 but does not enumerate constituent datasets; exposure is unverified",
                "KEV-9B": "the released model card lists fancyzhx/dbpedia_14 among its training datasets",
            }
        },
        "panels": comparison_panels,
        "reports": {name: str(path.resolve()) for name, path in report_paths.items()},
        "report_sha256": {name: sha256(path) for name, path in report_paths.items()},
        "merge_code": str(Path(__file__).resolve()),
        "merge_code_sha256": sha256(Path(__file__).resolve()),
        "component_reports": {
            name: [str(path.resolve()) for path in paths]
            for name, paths in model_inputs.items()
        },
        "input_sha256": {
            name: [sha256(path) for path in paths] for name, paths in model_inputs.items()
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for prefix in ("static", "openjev", "kev"):
        parser.add_argument(f"--{prefix}-base", type=Path, required=True)
        parser.add_argument(f"--{prefix}-extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
