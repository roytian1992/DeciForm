import json
from deciform.core.mind import prepare_mind


def test_mind_all_history_and_all_clicks(tmp_path):
    news = tmp_path / "news.tsv"
    news.write_text(
        "".join(f"N{i}\tcategory\tsub\tTitle {i}\tBody {i}\turl\t[]\t[]\n" for i in range(6))
    )
    behavior = tmp_path / "behaviors.tsv"
    behavior.write_text("1\tUser42\t11/01/2019 01:00:00 PM\tN0 N1\tN2-1 N3-0 N4-1 N5-0\n")
    m = prepare_mind(news, behavior, "train", tmp_path / "prepared", negative_count=1)
    row = json.loads((tmp_path / "prepared/train.jsonl").read_text())
    view = json.loads((tmp_path / "prepared/train.views.jsonl").read_text())
    assert m["rows"] == 1 and row["history_ids"] == ["N0", "N1"]
    assert row["target_distribution"] == [0.5, 0, 0.5, 0] and len(view["question"]["options"]) == 3
    assert "User42" not in json.dumps(row) and view["target_distribution"].count(0.5) == 2
