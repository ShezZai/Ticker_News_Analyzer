"""Offline unit tests for the classification eval. No DB, no network."""

import pytest

from ticker_news.evals.classify_eval import (
    act_accuracy_evaluator,
    build_items,
    load_ground_truth,
    predicted_label_evaluator,
)


def _write_csv(tmp_path, text, *, bom=True, name="gt.csv"):
    path = tmp_path / name
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(text, encoding=encoding)
    return path


GOOD_CSV = (
    "article id,header,Act_GT\n"
    "595,Some headline,NO\n"
    "14682,KraneShares Cross-Lists KOID,YES\n"
)


class TestLoadGroundTruth:
    def test_loads_rows_with_bom(self, tmp_path):
        rows = load_ground_truth(_write_csv(tmp_path, GOOD_CSV))
        assert rows == [
            {"article_id": 595, "header": "Some headline", "act": "NO"},
            {"article_id": 14682, "header": "KraneShares Cross-Lists KOID", "act": "YES"},
        ]

    def test_normalizes_case_and_whitespace(self, tmp_path):
        csv_text = "article id,header,Act_GT\n 595 ,H, yes \n"
        rows = load_ground_truth(_write_csv(tmp_path, csv_text, bom=False))
        assert rows == [{"article_id": 595, "header": "H", "act": "YES"}]

    def test_duplicate_id_raises_with_line_number(self, tmp_path):
        csv_text = "article id,header,Act_GT\n595,A,NO\n595,B,YES\n"
        with pytest.raises(ValueError, match="line 3.*duplicate"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_bad_act_value_raises(self, tmp_path):
        csv_text = "article id,header,Act_GT\n595,A,MAYBE\n"
        with pytest.raises(ValueError, match="line 2.*MAYBE"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_non_integer_id_raises(self, tmp_path):
        csv_text = "article id,header,Act_GT\nabc,A,NO\n"
        with pytest.raises(ValueError, match="line 2.*abc"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_missing_column_raises(self, tmp_path):
        csv_text = "id,header,label\n595,A,NO\n"
        with pytest.raises(ValueError, match="Act_GT"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_empty_csv_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no rows"):
            load_ground_truth(_write_csv(tmp_path, "article id,header,Act_GT\n"))


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    def __init__(self, rows=None):
        self.executed = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)


GT_ROWS = [
    {"article_id": 595, "header": "H595", "act": "NO"},
    {"article_id": 14682, "header": "H14682", "act": "YES"},
]


class TestBuildItems:
    def test_builds_dataset_items(self):
        conn = FakeConn(rows=[
            (595, "Title 595", "ok", True),
            (14682, "Title 14682", "ok", True),
        ])
        items = build_items(conn, GT_ROWS)
        assert items == [
            {
                "id": "article-595",
                "input": {"article_id": 595, "title": "Title 595"},
                "expected_output": {"act": "NO"},
                "metadata": {"gt_header": "H595"},
            },
            {
                "id": "article-14682",
                "input": {"article_id": 14682, "title": "Title 14682"},
                "expected_output": {"act": "YES"},
                "metadata": {"gt_header": "H14682"},
            },
        ]

    def test_missing_ids_raise(self):
        conn = FakeConn(rows=[(595, "T", "ok", True)])
        with pytest.raises(ValueError, match="not found.*14682"):
            build_items(conn, GT_ROWS)

    def test_unscraped_articles_raise(self):
        conn = FakeConn(rows=[
            (595, "T", "error", False),
            (14682, "T", "ok", True),
        ])
        with pytest.raises(ValueError, match="no scraped content.*595"):
            build_items(conn, GT_ROWS)


class TestItemEvaluators:
    def test_correct_act_scores_one(self):
        ev = act_accuracy_evaluator(
            output={"predicted": "real news", "act": "YES"},
            expected_output={"act": "YES"},
        )
        assert ev.name == "act_accuracy"
        assert ev.value == 1.0
        assert "real news" in ev.comment

    def test_wrong_act_scores_zero(self):
        ev = act_accuracy_evaluator(
            output={"predicted": "conference-PR", "act": "NO"},
            expected_output={"act": "YES"},
        )
        assert ev.value == 0.0
        assert "gt=YES" in ev.comment

    def test_missing_output_scores_zero_with_comment(self):
        ev = act_accuracy_evaluator(output=None, expected_output={"act": "NO"})
        assert ev.value == 0.0
        assert "no output" in ev.comment

    def test_predicted_label_is_categorical(self):
        ev = predicted_label_evaluator(output={"predicted": "legal-call", "act": "NO"})
        assert ev.name == "predicted_label"
        assert ev.value == "legal-call"

    def test_predicted_label_handles_missing_output(self):
        ev = predicted_label_evaluator(output=None)
        assert ev.value == "<none>"
