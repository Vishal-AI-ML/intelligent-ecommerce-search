import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ecommerce_search import __version__
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_db_session
from ecommerce_search.api.schemas import HealthResponse
from ecommerce_search.db.checks import CheckResult, check_database, check_pgvector

SENSITIVE = "host=internal-db.example password=hunter2 Traceback"


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeSession:
    def __init__(self, db_down=False, vector_version="0.8.0", vector_error=False):
        self.db_down = db_down
        self.vector_error = vector_error
        self.vector_version = vector_version

    def execute(self, statement):
        if self.db_down:
            raise OperationalError(str(statement), {}, Exception(SENSITIVE))
        if self.vector_error and "pg_extension" in str(statement):
            raise OperationalError(str(statement), {}, Exception(SENSITIVE))
        return FakeResult(self.vector_version if "pg_extension" in str(statement) else 1)

    def close(self):
        pass


@pytest.fixture
def client_for(make_settings):
    def _make(session: FakeSession) -> TestClient:
        app = create_app(make_settings())
        app.dependency_overrides[get_db_session] = lambda: session
        return TestClient(app)

    return _make


def test_check_database_ok():
    result = check_database(FakeSession())
    assert result.ok and result.latency_ms is not None


def test_check_database_failure_is_generic():
    result = check_database(FakeSession(db_down=True))
    assert not result.ok
    assert result.detail == "database unreachable"


def test_check_pgvector_present_and_missing():
    assert check_pgvector(FakeSession(vector_version="0.8.0")).version == "0.8.0"
    missing = check_pgvector(FakeSession(vector_version=None))
    assert not missing.ok and missing.detail == "extension not installed"


def test_check_pgvector_sqlalchemy_error_uses_fixed_contract(caplog):
    with caplog.at_level("ERROR"):
        result = check_pgvector(FakeSession(vector_error=True))
    assert result == CheckResult(ok=False, detail="pgvector check failed")
    for leaked in ("internal-db", "hunter2", "Traceback", "SELECT"):
        assert leaked not in repr(result)
        assert leaked not in caplog.text
    assert "OperationalError" in caplog.text


def test_health_pgvector_query_error_returns_503_without_leaking(client_for):
    with client_for(FakeSession(vector_error=True)) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = HealthResponse.model_validate(response.json())
    assert body.checks.database.status == "ok"
    assert body.checks.pgvector.status == "error"
    assert body.checks.pgvector.detail == "pgvector check failed"
    for leaked in ("internal-db", "hunter2", "Traceback", "unit-test-password", "SELECT"):
        assert leaked not in response.text


def test_health_ok(client_for):
    with client_for(FakeSession()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.service == "ecommerce-search"
    assert body.version == __version__
    assert body.environment == "test"
    assert body.checks.database.status == "ok"
    assert body.checks.pgvector.status == "ok"
    assert body.checks.pgvector.version == "0.8.0"


def test_health_database_down_returns_503_without_leaking(client_for):
    with client_for(FakeSession(db_down=True)) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = HealthResponse.model_validate(response.json())
    assert body.status == "unavailable"
    assert body.checks.database.status == "error"
    assert body.checks.pgvector.detail == "not checked"
    for leaked in ("internal-db", "hunter2", "Traceback", "unit-test-password", "postgresql"):
        assert leaked not in response.text


def test_health_pgvector_missing_returns_503(client_for):
    with client_for(FakeSession(vector_version=None)) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = HealthResponse.model_validate(response.json())
    assert body.checks.database.status == "ok"
    assert body.checks.pgvector.status == "error"


def test_openapi_documents_health(client_for):
    with client_for(FakeSession()) as client:
        spec = client.get("/openapi.json").json()
    responses = spec["paths"]["/health"]["get"]["responses"]
    assert {"200", "503"} <= set(responses)
    assert "HealthResponse" in spec["components"]["schemas"]


def test_fake_session_sanity():
    # Guards the fake itself: text() statements are stringified the way the checks expect.
    assert FakeSession().execute(text("SELECT 1")) is not None
