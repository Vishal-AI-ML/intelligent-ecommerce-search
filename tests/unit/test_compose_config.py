import re
from pathlib import Path

from ecommerce_search.config.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")

FORWARDED = {
    "DB_CONNECT_TIMEOUT_SECONDS": "db_connect_timeout_seconds",
    "DB_POOL_TIMEOUT_SECONDS": "db_pool_timeout_seconds",
    "DB_STATEMENT_TIMEOUT_MS": "db_statement_timeout_ms",
    # Milestone 3: the api service reads these; db and migrate do not.
    "SEARCH_LEXICAL_K": "search_lexical_k",
    "SEARCH_DEFAULT_TOP_K": "search_default_top_k",
    "SEARCH_MAX_QUERY_LENGTH": "search_max_query_length",
    # Milestone 4: dense search (api only).
    "SEARCH_DENSE_K": "search_dense_k",
}


def test_compose_forwarded_defaults_match_settings_and_env_example():
    for env_name, field in FORWARDED.items():
        default = Settings.model_fields[field].default
        assert f"{env_name}: ${{{env_name}:-{default}}}" in COMPOSE
        assert re.search(rf"^{env_name}={default}$", ENV_EXAMPLE, re.MULTILINE)


def test_migrate_does_not_receive_api_only_timeouts():
    migrate = COMPOSE.split("  migrate:")[1].split("  api:")[0]
    assert "<<:" not in migrate and "DB_POOL_TIMEOUT_SECONDS" not in migrate
    assert "DB_STATEMENT_TIMEOUT_MS" not in migrate
    assert "SEARCH_" not in migrate


def test_search_settings_reach_only_the_api_service():
    db = COMPOSE.split("  db:")[1].split("  migrate:")[0]
    api = COMPOSE.split("  api:")[1].split("volumes:")[0]
    assert "SEARCH_" not in db
    for name in ("SEARCH_LEXICAL_K", "SEARCH_DEFAULT_TOP_K", "SEARCH_MAX_QUERY_LENGTH"):
        assert name in api


def _service(name: str, following: str) -> str:
    return COMPOSE.split(f"  {name}:")[1].split(following)[0]


def test_only_the_api_gets_the_read_only_model_mount_and_embedding_settings():
    db = _service("db", "  migrate:")
    migrate = _service("migrate", "  api:")
    # the api service runs until the top-level `volumes:` key (column 0)
    api = re.split(r"^volumes:", COMPOSE.split("  api:")[1], flags=re.MULTILINE)[0]
    for section in (db, migrate):
        assert "/models" not in section and "EMBEDDING_" not in section
        assert "SEARCH_DENSE_K" not in section
    assert "      - ./models:/models:ro" in api  # read-only bind mount, never copied in
    assert "EMBEDDING_MODELS_DIR: /models" in api


def test_model_weights_are_never_copied_into_the_image():
    root_files = {
        name: (ROOT / name).read_text(encoding="utf-8") for name in ("Dockerfile", ".dockerignore")
    }
    assert "models" not in root_files["Dockerfile"]
    assert "models/" in root_files[".dockerignore"].splitlines()
