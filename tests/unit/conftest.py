import os

import pytest

from ecommerce_search.config import Settings, get_settings

_ISOLATED_PREFIXES = ("POSTGRES_", "DB_", "APP_ENV", "LOG_LEVEL", "EMBEDDING_", "SEARCH_")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Unit tests never read the developer's real .env or ambient database settings.

    Tests that need settings build them explicitly (`make_settings`) or set their own env vars.
    """
    for key in list(os.environ):
        if key.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
