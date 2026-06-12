"""Offline unit tests for the classification eval. No DB, no network."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ticker_news.classification.variants import (
    BinaryClassification,
    VariantRunner,
    is_act_binary,
)
from ticker_news.evals import classify_eval
from ticker_news.evals.classify_eval import (
    act_accuracy_evaluator,
    act_metrics_run_evaluator,
    build_items,
    load_ground_truth,
    make_task,
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

    def close(self):
        pass


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
                "input": {"article_id": 595, "title": "Title 595"},
                "expected_output": {"act": "NO"},
                "metadata": {"gt_header": "H595"},
            },
            {
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

    def test_no_ground_truth_skips_accuracy(self):
        ev = act_accuracy_evaluator(
            output={"predicted": "real news", "act": "YES"},
            expected_output=None,
        )
        assert ev.name == "act_accuracy_skip"
        assert "no ground truth" in ev.value

    def test_predicted_label_is_categorical(self):
        ev = predicted_label_evaluator(output={"predicted": "legal-call", "act": "NO"})
        assert ev.name == "predicted_label"
        assert ev.value == "legal-call"

    def test_predicted_label_handles_missing_output(self):
        ev = predicted_label_evaluator(output=None)
        assert ev.value == "<none>"


def _item_result(expected_act, predicted_act):
    return SimpleNamespace(
        item={"expected_output": {"act": expected_act} if expected_act else None},
        output={"predicted": "x", "act": predicted_act} if predicted_act else None,
    )


def _by_name(evaluations):
    return {e.name: e for e in evaluations}


class TestRunEvaluator:
    def test_confusion_metrics(self):
        # TP=2 FP=1 FN=1 TN=2 -> acc 4/6, precision 2/3, recall 2/3
        results = [
            _item_result("YES", "YES"), _item_result("YES", "YES"),  # TP
            _item_result("NO", "YES"),                               # FP
            _item_result("YES", "NO"),                               # FN
            _item_result("NO", "NO"), _item_result("NO", "NO"),      # TN
        ]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_accuracy_avg"].value == pytest.approx(4 / 6)
        assert evals["act_precision"].value == pytest.approx(2 / 3)
        assert evals["act_recall"].value == pytest.approx(2 / 3)
        assert evals["act_f1"].value == pytest.approx(2 / 3)
        assert "TP=2 FP=1 FN=1 TN=2" in evals["act_accuracy_avg"].comment

    def test_no_yes_predictions_skips_precision(self):
        results = [_item_result("YES", "NO"), _item_result("NO", "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert "act_precision" not in evals
        assert "no YES predictions" in evals["act_precision_skip"].value
        assert evals["act_recall"].value == 0.0
        assert "act_f1" not in evals

    def test_no_yes_items_skips_recall(self):
        results = [_item_result("NO", "YES"), _item_result("NO", "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert "act_recall" not in evals
        assert "no YES items" in evals["act_recall_skip"].value
        assert evals["act_precision"].value == 0.0

    def test_failed_items_are_counted_as_wrong(self):
        # an errored task (output=None) on a YES item counts as FN
        results = [_item_result("YES", None), _item_result("YES", "YES")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_recall"].value == pytest.approx(0.5)

    def test_failed_no_item_counts_as_false_positive(self):
        # an errored task (output=None) on a NO item must hurt, not count as TN
        results = [_item_result("NO", None), _item_result("NO", "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_accuracy_avg"].value == pytest.approx(0.5)
        assert "TP=0 FP=1 FN=0 TN=1" in evals["act_accuracy_avg"].comment

    def test_empty_results_skip_everything(self):
        evals = _by_name(act_metrics_run_evaluator(item_results=[]))
        assert set(evals) == {"act_metrics_skip"}

    def test_items_without_ground_truth_are_excluded(self):
        # unlabeled items (expected_output None) must not enter the confusion
        # counts — only the two labeled ones below do (one TP, one TN)
        results = [
            _item_result(None, "YES"), _item_result(None, "NO"),
            _item_result("YES", "YES"), _item_result("NO", "NO"),
        ]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_accuracy_avg"].value == pytest.approx(1.0)
        assert "TP=1 FP=0 FN=0 TN=1" in evals["act_accuracy_avg"].comment

    def test_all_unlabeled_skips_everything(self):
        results = [_item_result(None, "YES"), _item_result(None, "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert set(evals) == {"act_metrics_skip"}


class StubChain:
    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        return self._verdicts.pop(0)


class TestMakeTask:
    def test_task_reads_db_and_returns_verdict_dict(self, monkeypatch):
        conn = FakeConn(rows=[("Title", "Body text")])
        monkeypatch.setattr(classify_eval, "connect_eval", lambda dsn: conn)
        runner = VariantRunner(
            lite=StubChain(BinaryClassification(label="real news", confidence=0.7,
                                                reason="earnings")),
            confirm=None, is_act=is_act_binary, label_of=lambda v: v.label,
        )
        task = make_task(runner, dsn=None)
        out = asyncio.run(task(item={"input": {"article_id": 595, "title": "T"}}))
        assert out == {
            "predicted": "real news", "act": "YES", "confidence": 0.7,
            "reason": "earnings", "confirmed": False,
        }
        # the SELECT was parametrized on the article id
        assert any(p == (595,) for _, p in conn.executed)

    def test_task_raises_for_missing_article(self, monkeypatch):
        conn = FakeConn(rows=[])
        monkeypatch.setattr(classify_eval, "connect_eval", lambda dsn: conn)
        runner = VariantRunner(
            lite=StubChain(), confirm=None,
            is_act=is_act_binary, label_of=lambda v: v.label,
        )
        task = make_task(runner, dsn=None)
        with pytest.raises(ValueError, match="595"):
            asyncio.run(task(item={"input": {"article_id": 595, "title": "T"}}))

    def test_task_names_the_trace_after_variant_and_article(self, monkeypatch):
        import langfuse

        seen = {}

        @contextmanager
        def fake_propagate(**kwargs):
            seen.update(kwargs)
            yield

        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
        conn = FakeConn(rows=[("Title", "Body")])
        monkeypatch.setattr(classify_eval, "connect_eval", lambda dsn: conn)
        runner = VariantRunner(
            lite=StubChain(BinaryClassification(label="none news")),
            confirm=None, is_act=is_act_binary, label_of=lambda v: v.label,
        )
        task = make_task(runner, dsn=None, trace_prefix="classify-binary")
        asyncio.run(task(item={"input": {"article_id": 595, "title": "T"}}))
        assert seen["trace_name"] == "classify-binary:article-595"
