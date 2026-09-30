import pytest
from fastapi.testclient import TestClient

from ecommerce_search.api.app import create_app
from ecommerce_search.api.schemas import HealthResponse

pytestmark = pytest.mark.integration


def test_health_ok_against_migrated_scratch_database(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database)
    scratch_settings = settings.model_copy(update={"postgres_db": scratch_database})
    with TestClient(create_app(scratch_settings)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.checks.pgvector.version


def test_health_503_when_database_unreachable(settings):
    unreachable = settings.model_copy(update={"postgres_port": 1})
    with TestClient(create_app(unreachable)) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = HealthResponse.model_validate(response.json())
    assert body.status == "unavailable"
    assert body.checks.database.detail == "database unreachable"
    assert settings.postgres_password.get_secret_value() not in response.text
