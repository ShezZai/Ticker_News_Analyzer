import os

import pytest

from ticker_news.shared.config import AppSettings, get_settings

# Tests must never read a developer's local .env — neutralize it for the
# whole session. Individual tests control config via monkeypatched env vars.
AppSettings.model_config["env_file"] = None

_LANGFUSE_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch):
    from ticker_news.shared import observability
    from ticker_news.shared import prompts as prompts_mod

    for var in _LANGFUSE_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    observability.client.cache_clear()
    prompts_mod._seen_versions.clear()
    yield
    get_settings.cache_clear()
    observability.client.cache_clear()
    prompts_mod._seen_versions.clear()
