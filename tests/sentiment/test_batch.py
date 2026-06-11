"""Offline tests for the batch sentiment runner (stubbed DB, trace, and stage).

Pins the eval contract documented in CLAUDE.md: the root trace output carries
the verdict on the batch path too, not just in the service worker.
"""

from contextlib import contextmanager

from ticker_news.sentiment import batch


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        return FakeCursor(self._rows)

    def commit(self):
        pass

    def close(self):
        pass


class FakeRoot:
    """Captures root.update kwargs the way a Langfuse span would receive them."""

    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def _stub_batch(monkeypatch, *, rows, root, stage_result):
    """Stub DB, schema, stage, and trace; return the article_trace call log."""
    trace_calls = []

    @contextmanager
    def fake_article_trace(url, *, ticker=None, entrypoint="service"):
        trace_calls.append((url, ticker, entrypoint))
        yield root

    monkeypatch.setattr(batch, "connect", lambda vector=True: FakeConn(rows))
    monkeypatch.setattr(batch.store, "ensure_schema", lambda conn: None)
    monkeypatch.setattr(batch, "sentiment_stage", lambda conn, url: stage_result)
    monkeypatch.setattr(batch.obs, "article_trace", fake_article_trace)
    return trace_calls


def test_run_batch_root_output_carries_verdict(monkeypatch):
    root = FakeRoot()
    verdict = {"ticker": "NVDA", "action": "buy", "confidence": 0.9}
    trace_calls = _stub_batch(
        monkeypatch, rows=[("u1", "NVDA")], root=root, stage_result=verdict)

    assert batch.run_batch() == 1
    assert trace_calls == [("u1", "NVDA", "batch")]
    (kw,) = root.updates
    assert kw["output"] == {"ok": True, "verdict": verdict}
    assert "metadata" in kw


def test_run_batch_non_dict_result_omits_verdict(monkeypatch):
    root = FakeRoot()
    _stub_batch(monkeypatch, rows=[("u1", "NVDA")], root=root, stage_result=None)

    assert batch.run_batch() == 1
    (kw,) = root.updates
    assert kw["output"] == {"ok": True}


def test_run_batch_keyless_root_none_skips_update(monkeypatch):
    # Langfuse disabled -> article_trace yields None; the loop must not touch it.
    _stub_batch(monkeypatch, rows=[("u1", "NVDA")], root=None,
                stage_result={"action": "buy"})
    assert batch.run_batch() == 1
