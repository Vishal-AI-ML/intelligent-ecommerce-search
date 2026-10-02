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
    "SEARCH_LEXICAL_K",
    "SEARCH_DEFAULT_TOP_K",
    "SEARCH_MAX_QUERY_LENGTH",
    "SEARCH_DENSE_K",
    "SEARCH_CANDIDATE_K",
    "SEARCH_RRF_K",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_REVISION",
    "EMBEDDING_MODELS_DIR",
    "EMBEDDING_BATCH_SIZE",
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
        {"search_lexical_k": 0},
        {"search_default_top_k": 0},
        {"search_max_query_length": 0},
        {"search_lexical_k": 5, "search_default_top_k": 6},
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


def test_search_defaults(make_settings):
    s = make_settings()
    assert (s.search_lexical_k, s.search_default_top_k, s.search_max_query_length) == (50, 10, 200)


def test_search_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SEARCH_LEXICAL_K", "20")
    monkeypatch.setenv("SEARCH_DEFAULT_TOP_K", "5")
    monkeypatch.setenv("SEARCH_MAX_QUERY_LENGTH", "80")
    s = Settings(_env_file=None)
    assert (s.search_lexical_k, s.search_default_top_k, s.search_max_query_length) == (20, 5, 80)


# ---- Milestone 4: dense search and embedding settings ---------------------------------------------


def test_dense_and_embedding_defaults_pin_the_reviewed_model(make_settings):
    s = make_settings()
    assert s.search_dense_k == 50 and s.embedding_batch_size == 32
    assert s.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embedding_model_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert s.embedding_spec().dimension == 384
    assert s.embedding_models_dir is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model_id": "BAAI/bge-small-en-v1.5"},  # not in the production registry
        {"embedding_model_id": "org/unreviewed"},
        {"embedding_model_revision": "main"},
        {"embedding_model_revision": "f" * 40},
        {"search_dense_k": 0},
        {"search_dense_k": 5, "search_default_top_k": 6},
        {"embedding_batch_size": 0},
        {"embedding_batch_size": 257},
    ],
)
def test_invalid_embedding_settings_rejected(make_settings, overrides):
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_models_dir_defaults_to_the_repository_root_and_can_be_overridden(make_settings, tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert make_settings().resolved_models_dir() == root / "models"
    assert make_settings(embedding_models_dir=tmp_path).resolved_models_dir() == tmp_path


def test_dense_settings_read_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SEARCH_DENSE_K", "20")
    monkeypatch.setenv("EMBEDDING_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    s = Settings(_env_file=None)
    assert (s.search_dense_k, s.embedding_models_dir, s.embedding_batch_size) == (20, tmp_path, 8)


# ---- Milestone 5: hybrid search settings ----------------------------------------------------------


def test_hybrid_defaults(make_settings):
    s = make_settings()
    assert (s.search_lexical_k, s.search_dense_k, s.search_candidate_k) == (50, 50, 50)
    assert s.search_rrf_k == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"search_candidate_k": 0},
        {"search_candidate_k": 2001},
        {"search_rrf_k": 0},
        {"search_rrf_k": 1001},
        {"search_rrf_k": "sixty"},
        {"search_candidate_k": 5, "search_default_top_k": 6},
        {"search_lexical_k": 10, "search_dense_k": 10, "search_candidate_k": 21},
    ],
)
def test_invalid_hybrid_settings_rejected(make_settings, overrides):
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_candidate_k_may_equal_the_sum_of_source_depths(make_settings):
    s = make_settings(search_lexical_k=10, search_dense_k=15, search_candidate_k=25)
    assert s.search_candidate_k == 25
    assert make_settings(search_rrf_k=1).search_rrf_k == 1
    assert make_settings(search_rrf_k=1000).search_rrf_k == 1000


def test_hybrid_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SEARCH_CANDIDATE_K", "30")
    monkeypatch.setenv("SEARCH_RRF_K", "20")
    s = Settings(_env_file=None)
    assert (s.search_candidate_k, s.search_rrf_k) == (30, 20)


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("60", 60), ("1000", 1000)])
def test_rrf_k_environment_override_within_range(monkeypatch, raw, expected):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SEARCH_RRF_K", raw)
    assert Settings(_env_file=None).search_rrf_k == expected


@pytest.mark.parametrize("raw", ["0", "1001"])
def test_rrf_k_environment_override_out_of_range_rejected(monkeypatch, raw):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("SEARCH_RRF_K", raw)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
