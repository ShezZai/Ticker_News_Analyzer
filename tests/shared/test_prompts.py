from ticker_news.shared import prompts
from ticker_news.shared.config import get_settings


def _disable(monkeypatch):
    get_settings.cache_clear()
    from ticker_news.shared import observability as obs
    obs.client.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_fallback_when_disabled(monkeypatch):
    _disable(monkeypatch)
    assert prompts.get_prompt("classify-article", "FALLBACK {x}") == "FALLBACK {x}"


def test_registry_covers_all_llm_prompts(monkeypatch):
    _disable(monkeypatch)
    reg = prompts.registry()
    assert set(reg) == {
        "classify-article", "extract-insights",
        "analyst-fundamentals", "analyst-market_context",
        "analyst-historical_precedent", "synthesize-verdict",
    }
    assert all(isinstance(v, str) and v for v in reg.values())
