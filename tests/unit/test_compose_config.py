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
