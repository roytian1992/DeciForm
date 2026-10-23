"""Shared counting rules for source families and evaluation panels."""
from __future__ import annotations


def source_family(name: str, panel: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for suffix in ("__natural", "__balanced"):
        if panel in {"natural", "balanced"} and name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def population_counts(source_metrics: dict[str, dict]) -> dict:
    families = set()
    panels: dict[str, list[tuple[str, dict]]] = {}
    for name, metrics in source_metrics.items():
        panel = metrics["panel"]
        family = source_family(name, panel, metrics.get("source_family"))
        metrics["source_family"] = family
        families.add(family)
        panels.setdefault(panel, []).append((family, metrics))

    panel_counts = {}
    for panel, entries in panels.items():
        panel_families = [family for family, _ in entries]
        if len(set(panel_families)) != len(panel_families):
            raise ValueError(f"panel {panel!r} contains duplicate source families")
        panel_counts[panel] = {
            "source_count": len(panel_families),
            "unique_source_count": len(panel_families),
            "source_panel_entry_count": len(entries),
            "record_count": sum(metrics["record_count"] for _, metrics in entries),
        }

    entry_count = len(source_metrics)
    unique_count = len(families)
    return {
        "source_count": unique_count,
        "unique_source_count": unique_count,
        "source_panel_entry_count": entry_count,
        "source_families": sorted(families),
        "panel_counts": panel_counts,
        "count_semantics": {
            "source_count": "unique dataset/source-family count across all panels",
            "source_panel_entry_count": "number of source-by-panel metric entries; repeated families across panels count repeatedly",
            "panel_counts": "source_count is the number of unique source families within that panel",
        },
    }
