from ecommerce_search.db import engine as engine_module
from ecommerce_search.db.engine import create_db_engine


def test_engine_applies_pool_timeout(make_settings):
    engine = create_db_engine(make_settings(db_pool_timeout_seconds=7))
    try:
        assert engine.pool.timeout() == 7
    finally:
        engine.dispose()


def test_engine_passes_safe_connect_args(make_settings, monkeypatch):
    captured = {}

    def fake_create_engine(url, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(engine_module, "create_engine", fake_create_engine)
    settings = make_settings(
        db_pool_timeout_seconds=7,
        db_connect_timeout_seconds=4,
        db_statement_timeout_ms=1500,
    )
    engine_module.create_db_engine(settings)
    assert captured["pool_timeout"] == 7
    assert captured["connect_args"] == {
        "connect_timeout": 4,
        "options": "-c statement_timeout=1500",
    }
