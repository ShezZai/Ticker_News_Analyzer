"""Offline unit tests for the classification eval. No DB, no network."""

import pytest

from ticker_news.evals.classify_eval import load_ground_truth


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
