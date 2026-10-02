"""`/search/dense` end to end against real PostgreSQL scratch databases (fake model)."""

import logging

import pytest
from dense_support import change_product, embed_fake
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_embedder
from ecommerce_search.api.schemas import DenseSearchResponse
from ecommerce_search.ingestion.service import ingest_file
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

PID = "SYN-SHO-0001"


def make_client(settings, engine, fake):
    scratch = settings.model_copy(update={"postgres_db": engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    app = create_app(scratch)
    app.dependency_overrides[get_embedder] = lambda: fake
    return TestClient(app)


@pytest.fixture
def catalog(migrated_engine):
    from catalog_support import PROVENANCE, SEED

    ingest_file(migrated_engine, SEED, PROVENANCE)
    return migrated_engine


def test_dense_search_contract_on_a_real_database(settings, catalog):
    embed_fake(catalog)
    with make_client(settings, catalog, FakeEmbedder()) as client:
        response = client.get("/search/dense", params={"q": "running shoes", "top_k": 5})
    assert response.status_code == 200
    body = DenseSearchResponse.model_validate(response.json())
    assert body.result_count == 5 and [r.rank for r in body.results] == [1, 2, 3, 4, 5]
    scores = [r.dense_score for r in body.results]
    assert scores == sorted(scores, reverse=True)
    assert body.latency_ms.vector_ms > 0 and body.latency_ms.query_embedding_ms >= 0
    assert body.embedding_model_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert all(r.category for r in body.results)  # rows joined to current products


def test_changed_product_disappears_until_embed_heals_it(settings, catalog):
    embed_fake(catalog)
    query = {"q": "Strato running shoes", "top_k": 50}
    with make_client(settings, catalog, FakeEmbedder()) as client:
        before = [
            r["product_id"] for r in client.get("/search/dense", params=query).json()["results"]
        ]
        assert PID in before
        change_product(catalog, PID, "Nike Strato Running Shoes for Men - Blue Model SH101")
        stale = [
            r["product_id"] for r in client.get("/search/dense", params=query).json()["results"]
        ]
        assert PID not in stale  # the old vector never ranks the updated product
        embed_fake(catalog)
        healed = client.get("/search/dense", params=query).json()["results"]
    assert PID in [r["product_id"] for r in healed]
    assert next(r for r in healed if r["product_id"] == PID)["title"].endswith("Blue Model SH101")


def test_unembedded_database_returns_empty_results_not_an_error(settings, catalog):
    with make_client(settings, catalog, FakeEmbedder()) as client:
        body = client.get("/search/dense", params={"q": "laptop"}).json()
    assert body["result_count"] == 0 and body["results"] == []


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_database_below_0004_is_a_fixed_503(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database, "0003")
    # Alembic's fileConfig replaces the root handlers (pytest's caplog included), so the dense
    # logger gets its own capturing handler.
    records = _Records()
    dense_logger = logging.getLogger("ecommerce_search.api.dense")
    dense_logger.addHandler(records)
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        with make_client(settings, engine, FakeEmbedder()) as client:
            response = client.get("/search/dense", params={"q": "laptop"})
            lexical = client.get("/search", params={"q": "laptop"})
            health = client.get("/health")
    finally:
        dense_logger.removeHandler(records)
        engine.dispose()
    assert response.status_code == 503 and response.json() == {"detail": "search unavailable"}
    assert any("table_missing" in message for message in records.messages)
    assert "product_embeddings" not in response.text
    assert lexical.status_code == 200 and health.status_code == 200


def test_missing_model_snapshot_keeps_lexical_and_health_working(settings, catalog, tmp_path):
    scratch = settings.model_copy(
        update={"postgres_db": catalog.url.database, "embedding_models_dir": tmp_path}
    )
    with TestClient(create_app(scratch)) as client:  # the real provider, empty models dir
        dense = client.get("/search/dense", params={"q": "laptop"})
        lexical = client.get("/search", params={"q": "laptop"})
        health = client.get("/health")
    assert dense.status_code == 503 and dense.json() == {"detail": "search unavailable"}
    assert str(tmp_path) not in dense.text
    assert lexical.status_code == 200 and lexical.json()["result_count"] == 10
    assert health.status_code == 200 and health.json()["status"] == "ok"


def test_lexical_responses_are_unchanged_by_embeddings(settings, catalog):
    queries = ["hp laptop", "wireless headphones", "iphone", "!!!"]
    with make_client(settings, catalog, FakeEmbedder()) as client:
        before = [client.get("/search", params={"q": q, "top_k": 50}).json() for q in queries]
        embed_fake(catalog)
        after = [client.get("/search", params={"q": q, "top_k": 50}).json() for q in queries]
    for payload in before + after:
        payload.pop("latency_ms")
    assert before == after
