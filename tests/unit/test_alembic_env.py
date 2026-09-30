import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config

from ecommerce_search.config import get_settings


class _Stop(Exception):
    pass


def test_alembic_online_engine_uses_connect_timeout(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-password")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_SECONDS", "4")
    get_settings.cache_clear()
    captured = {}

    def fake_create_engine(url, **kwargs):
        captured["kwargs"] = kwargs
        raise _Stop

    # env.py does `from sqlalchemy import create_engine` each time Alembic loads it.
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)
    config = Config("alembic.ini")
    config.attributes["database"] = "some_scratch_db"
    try:
        with pytest.raises(_Stop):
            command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
    assert captured["kwargs"]["connect_args"] == {"connect_timeout": 4}
