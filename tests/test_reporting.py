import pytest

from deciform.reporting import population_counts


def test_population_counts_distinguish_source_families_from_panel_entries():
    metrics = {
        "ag_news": {"panel": "D_assess_visible", "record_count": 120},
        "ag_news__natural": {"panel": "natural", "record_count": 1000},
        "ag_news__balanced": {"panel": "balanced", "record_count": 1000},
        "cfpb": {"panel": "D_assess_visible", "record_count": 120},
    }

    result = population_counts(metrics)

    assert result["source_count"] == 2
    assert result["unique_source_count"] == 2
    assert result["source_panel_entry_count"] == 4
    assert result["panel_counts"]["natural"]["unique_source_count"] == 1
    assert result["source_families"] == ["ag_news", "cfpb"]
    assert metrics["ag_news__balanced"]["source_family"] == "ag_news"


def test_population_counts_reject_duplicate_family_in_one_panel():
    metrics = {
        "ag_news__natural": {"panel": "natural", "record_count": 10, "source_family": "ag_news"},
        "ag_news_copy": {"panel": "natural", "record_count": 10, "source_family": "ag_news"},
    }

    with pytest.raises(ValueError, match="duplicate source families"):
        population_counts(metrics)
