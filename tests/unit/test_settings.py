import pytest
from pydantic import ValidationError

from ecommerce_search.config import Settings

ENV_KEYS = [
    "APP_ENV",
    "LOG_LEVEL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "DB_CONNECT_TIMEOUT_SECONDS",
    "DB_POOL_TIMEOUT_SECONDS",
    "DB_STATEMENT_TIMEOUT_MS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults(make_settings):
    s = make_settings()
    assert s.postgres_host == "127.0.0.1"
    assert s.postgres_port == 5432
    assert s.postgres_db == "ecommerce_search"
    assert s.log_level == "INFO"
    assert s.db_connect_timeout_seconds == 3
    assert s.db_pool_timeout_seconds == 10
    assert s.db_statement_timeout_ms == 30_000


def test_environment_override(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("APP_ENV", "production")
    s = Settings(_env_file=None)
    assert (s.postgres_host, s.postgres_port, s.app_env) == ("db", 6543, "production")


def test_password_is_required():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"app_env": "staging"},
        {"log_level": "LOUD"},
        {"postgres_port": 0},
        {"postgres_port": 70000},
        {"postgres_password": ""},
        {"db_pool_timeout_seconds": 0},
        {"db_statement_timeout_ms": 0},
        {"db_statement_timeout_ms": -5},
        {"db_statement_timeout_ms": 2_147_483_648},
    ],
)
def test_invalid_values_rejected(make_settings, overrides):
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_password_not_in_repr(make_settings):
    s = make_settings(postgres_password="super-secret-value")  # noqa: S106
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in str(s)


def test_database_url_escapes_special_characters(make_settings):
    s = make_settings(postgres_password="p@ss:w/rd#1")  # noqa: S106
    url = s.database_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.password == "p@ss:w/rd#1"  # noqa: S105
    rendered = url.render_as_string(hide_password=False)
    assert "p%40ss%3Aw%2Frd%231" in rendered
    assert "p@ss:w/rd#1" not in rendered


def test_database_url_hides_password_by_default(make_settings):
    assert "unit-test-password" not in str(make_settings().database_url())


def test_database_override(make_settings):
    assert make_settings().database_url(database="other").database == "other"


def test_timeouts_read_from_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("DB_STATEMENT_TIMEOUT_MS", "1500")
    s = Settings(_env_file=None)
    assert (s.db_pool_timeout_seconds, s.db_statement_timeout_ms) == (7, 1500)


def test_statement_timeout_must_be_an_integer(make_settings):
    with pytest.raises(ValidationError):
        make_settings(db_statement_timeout_ms="1; DROP TABLE x")
