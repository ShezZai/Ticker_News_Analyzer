import pytest

from ticker_news.shared import llm


def test_gemini_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        llm.gemini_chat(llm.GEMINI_FLASH_LITE)


def test_openai_embeddings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        llm.openai_embeddings()


def test_gemini_chat_builds_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    model = llm.gemini_chat(llm.GEMINI_FLASH_LITE, timeout_s=30.0)
    assert model.model is not None
    assert llm.GEMINI_FLASH_LITE in model.model


def test_rate_limiter_is_shared(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    a = llm.gemini_chat(llm.GEMINI_FLASH_LITE)
    b = llm.gemini_chat(llm.GEMINI_FLASH)
    assert a.rate_limiter is b.rate_limiter
