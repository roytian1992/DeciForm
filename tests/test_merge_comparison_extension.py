import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_comparison_extension.py"
SPEC = importlib.util.spec_from_file_location("merge_comparison_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dbpedia_panel_ids_count_as_one_source_family():
    metric = {"record_count": 1, "accuracy": 1.0, "macro_f1": 1.0,
              "brier": 0.0, "ece": 0.0, "nll": 0.0}
    report = {"source_metrics": {
        "dbpedia14": {**metric, "source_family": "dbpedia14",
                       "panel": "D_assess_visible"},
        "dbpedia14__natural": {**metric, "source_family": "dbpedia14__natural",
                                "panel": "natural"},
        "dbpedia14__balanced": {**metric, "source_family": "dbpedia14__balanced",
                                "panel": "balanced"},
    }}

    merged = MODULE.summarize_report(report)

    assert merged["unique_source_count"] == 1
    assert merged["source_panel_entry_count"] == 3
    assert {item["source_family"] for item in merged["source_metrics"].values()} == {"dbpedia14"}
    assert {item["source_count"] for item in merged["panel_metrics"].values()} == {1}
