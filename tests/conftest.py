import pytest

from ticker_news.shared.config import AppSettings, get_settings

# Tests must never read a developer's local .env — neutralize it for the
# whole session. Individual tests control config via monkeypatched env vars.
AppSettings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from ticker_news.shared import observability

    get_settings.cache_clear()
    observability.client.cache_clear()
    yield
    get_settings.cache_clear()
    observability.client.cache_clear()
