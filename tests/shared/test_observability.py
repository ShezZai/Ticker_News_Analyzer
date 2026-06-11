from ticker_news.shared import observability as obs
from ticker_news.shared.config import get_settings


def _disable(monkeypatch):
    get_settings.cache_clear()
    obs.client.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_disabled_without_keys(monkeypatch):
    _disable(monkeypatch)
    assert obs.enabled() is False
    assert obs.client() is None


def test_chain_config_empty_when_disabled(monkeypatch):
    _disable(monkeypatch)
    assert obs.chain_config() == {}
    assert obs.chain_config(run_name="x") == {"run_name": "x"}


def test_article_trace_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with obs.article_trace("https://example.com/a", ticker="NVDA") as t:
        assert t is None


def test_article_trace_accepts_entrypoint_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with obs.article_trace("https://example.com/a", ticker="NVDA", entrypoint="batch") as t:
        assert t is None


def test_stage_span_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with obs.stage_span("classify") as s:
        assert s is None


def test_flush_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    obs.flush()  # must not raise


def test_enabled_with_keys(monkeypatch):
    get_settings.cache_clear()
    obs.client.cache_clear()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert obs.enabled() is True
    obs.client.cache_clear()
    get_settings.cache_clear()
