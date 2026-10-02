"""`GET/POST /search/dense` with a fake embedder and a fake retrieval function (no DB, no model)."""

import logging
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, ProgrammingError

from ecommerce_search.api import dense as dense_module
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_db_session, get_embedder
from ecommerce_search.api.schemas import DenseSearchResponse
from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.search.dense import DenseHit, DenseResult
from fake_embedder import FakeEmbedder

SENSITIVE = "host=internal-db.example password=hunter2 SELECT * FROM products D:\\secret\\path"
FIXED_503 = {"detail": "search unavailable"}


def hit(rank: int, product_id: str, score: float) -> DenseHit:
    return DenseHit(
        rank=rank,
        dense_score=score,
        product_id=product_id,
        title="Nike Test Running Shoes",
        brand="Nike",
        category="shoes",
        subcategory="running",
        description="Shoes.",
        price=Decimal("4299.00"),
        currency="INR",
        rating=Decimal("4.4"),
        review_count=7,
        availability="in_stock",
    )


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client_for(make_settings, monkeypatch, calls):
    def _make(embedder="fake", error=None, hits=None, **settings):
        fake = FakeEmbedder() if embedder == "fake" else embedder

        def fake_search(session, vector, spec, limit):
            calls.append((len(vector), spec.model_id, limit))
            if error is not None:
                raise error
            chosen = [hit(1, "P1", 0.91), hit(2, "P2", 0.42)] if hits is None else hits
            return DenseResult(hits=chosen[:limit], vector_ms=2.5)

        monkeypatch.setattr(dense_module, "dense_search", fake_search)
        app = create_app(make_settings(**settings))
        app.dependency_overrides[get_db_session] = lambda: object()
        app.dependency_overrides[get_embedder] = lambda: fake
        client = TestClient(app)
        client.fake = fake
        return client

    return _make


def test_get_success_shape_and_truthful_metadata(client_for, calls):
    with client_for() as client:
        response = client.get("/search/dense", params={"q": "  running   shoes ", "top_k": 5})
    assert response.status_code == 200
    body = DenseSearchResponse.model_validate(response.json())
    assert calls == [(384, ALL_MINILM_L6_V2.model_id, 5)]
    assert body.query == "running shoes" and body.search_version == "dense_only"
    assert body.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert body.embedding_model_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert body.embedding_dimension == 384 and body.embedding_text_version == "1"
    assert body.distance_metric == "cosine" and body.applied_filters == []
    assert body.top_k == 5 and body.result_count == len(body.results) == 2
    assert [(r.rank, r.product_id, r.dense_score) for r in body.results] == [
        (1, "P1", 0.91),
        (2, "P2", 0.42),
    ]
    latency = body.latency_ms
    assert latency.model_load_ms == 1.0  # loaded by this first request
    assert latency.vector_ms == 2.5 and latency.query_embedding_ms >= 0
    assert latency.total_ms >= 0
    raw = response.json()
    assert raw["results"][0]["price"] == "4299.00"
    assert set(raw) == {
        "query",
        "search_version",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dimension",
        "embedding_text_version",
        "distance_metric",
        "top_k",
        "result_count",
        "results",
        "applied_filters",
        "latency_ms",
    }
    assert set(raw["results"][0]) == {
        "rank",
        "dense_score",
        "product_id",
        "title",
        "brand",
        "category",
        "subcategory",
        "description",
        "price",
        "currency",
        "rating",
        "review_count",
        "availability",
    }


def test_model_is_loaded_once_and_reused(client_for):
    with client_for() as client:
        first = client.get("/search/dense", params={"q": "shoes"}).json()
        second = client.post("/search/dense", json={"query": "shoes"}).json()
    assert first["latency_ms"]["model_load_ms"] == 1.0
    assert second["latency_ms"]["model_load_ms"] is None


def test_get_and_post_share_one_path(client_for):
    with client_for() as client:
        got = client.get("/search/dense", params={"q": "wireless headphones", "top_k": 7}).json()
        posted = client.post(
            "/search/dense", json={"query": "wireless headphones", "top_k": 7}
        ).json()
    for payload in (got, posted):
        payload.pop("latency_ms")
    assert got == posted


def test_default_top_k_and_dense_limit_come_from_settings(client_for, calls):
    with client_for(search_default_top_k=3, search_dense_k=4) as client:
        assert client.get("/search/dense", params={"q": "x1"}).json()["top_k"] == 3
        assert client.get("/search/dense", params={"q": "x1", "top_k": 4}).status_code == 200
        assert client.get("/search/dense", params={"q": "x1", "top_k": 5}).status_code == 422
    assert [limit for *_, limit in calls] == [3, 4]


@pytest.mark.parametrize("query", ["!!!", "-", '"', "...  ???", "\u2014"])
def test_punctuation_only_is_an_empty_200_without_loading_the_model(client_for, calls, query):
    with client_for() as client:
        response = client.get("/search/dense", params={"q": query})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["result_count"] == 0
    assert body["latency_ms"]["model_load_ms"] is None
    assert body["latency_ms"]["query_embedding_ms"] is None
    assert body["latency_ms"]["vector_ms"] is None
    assert client.fake.load_calls == 0 and calls == []


def test_any_script_letters_are_searchable(client_for, calls):
    with client_for() as client:
        assert (
            client.get(
                "/search/dense", params={"q": "\u043d\u043e\u0443\u0442\u0431\u0443\u043a"}
            ).status_code
            == 200
        )
        assert (
            client.get("/search/dense", params={"q": "\u0968\u0966"}).status_code == 200
        )  # digits
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("params", "location"),
    [
        ({"q": ""}, "q"),
        ({"q": "   "}, "q"),
        ({"q": "shoes\x00"}, "q"),
        ({"q": "a\u200bb"}, "q"),
        ({"q": "x" * 201}, "q"),
        ({"q": "shoes", "top_k": 0}, "top_k"),
        ({"q": "shoes", "top_k": 51}, "top_k"),
        ({"q": "shoes", "top_k": "many"}, "top_k"),
    ],
)
def test_get_validation_errors_are_422_and_never_echo_the_query(
    client_for, calls, params, location
):
    with client_for() as client:
        response = client.get("/search/dense", params=params)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"][-1] == location
    assert "shoes\x00" not in response.text and "x" * 201 not in response.text
    assert calls == [] and client.fake.load_calls == 0


@pytest.mark.parametrize(
    "body",
    [
        {"query": "\ud800"},
        {"query": "shoes", "top_k": 0},
        {"query": "shoes", "extra": 1},
        {"top_k": 5},
        {"query": 5},
    ],
)
def test_post_validation_errors_are_422(client_for, calls, body):
    import json

    with client_for() as client:
        response = client.post(
            "/search/dense",
            content=json.dumps(body, ensure_ascii=True),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert calls == []


def test_query_over_the_model_token_limit_is_422(client_for, calls):
    with client_for() as client:
        client.fake.token_overrides["toolong"] = 257
        response = client.get("/search/dense", params={"q": "toolong query"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"].endswith("query is too long for the embedding model")
    assert "toolong" not in response.text and calls == []


@pytest.mark.parametrize(
    "reason",
    ["snapshot_missing", "load_failed", "dimension_mismatch", "config_mismatch", "encode_failed"],
)
def test_model_failures_are_a_fixed_503_with_a_sanitized_log(client_for, calls, caplog, reason):
    fake = FakeEmbedder()
    fake.fail_after_batches = 0
    fake.failure = EmbedderUnavailable(reason)
    with caplog.at_level(logging.ERROR), client_for(embedder=fake) as client:
        response = client.get("/search/dense", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    assert calls == []
    assert f"dense search unavailable: {reason}" in caplog.text


def test_no_models_directory_is_a_fixed_503(client_for, caplog):
    with caplog.at_level(logging.ERROR), client_for(embedder=None) as client:
        response = client.get("/search/dense", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    assert "snapshot_missing" in caplog.text and "model-fetch" in caplog.text


@pytest.mark.parametrize(
    ("error", "logged"),
    [
        (OperationalError(SENSITIVE, {"p": "hunter2"}, Exception(SENSITIVE)), "database_error"),
        (
            ProgrammingError(SENSITIVE, {}, type("E", (Exception,), {"sqlstate": "42P01"})()),
            "table_missing",
        ),
    ],
)
def test_database_failures_are_a_fixed_503_and_leak_nothing(client_for, caplog, error, logged):
    with caplog.at_level(logging.DEBUG), client_for(error=error) as client:
        response = client.get("/search/dense", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    for secret in ("hunter2", "internal-db", "SELECT", "secret\\path", "Traceback"):
        assert secret not in response.text
        assert secret not in caplog.text
    assert f"dense search unavailable: {logged}" in caplog.text


def test_openapi_documents_dense_and_keeps_lexical(client_for):
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    assert {"get", "post"} <= set(schema["paths"]["/search/dense"])
    assert {"get", "post"} <= set(schema["paths"]["/search"])
    dense = schema["paths"]["/search/dense"]["get"]
    assert "not a probability" in dense["description"] and "threshold" in dense["description"]
    lexical = schema["paths"]["/search"]["get"]
    assert lexical["summary"] == "Lexical search (V0)"
    assert lexical["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SearchResponse"
    }


def test_lexical_search_never_touches_the_embedder(client_for, monkeypatch):
    from ecommerce_search.api import search as search_module
    from ecommerce_search.search.lexical import LexicalResult

    monkeypatch.setattr(
        search_module,
        "lexical_search",
        lambda session, query, limit: LexicalResult(tsquery="'x'", hits=[], lexical_ms=0.1),
    )
    with client_for() as client:
        assert client.get("/search", params={"q": "x"}).status_code == 200
    assert client.fake.load_calls == 0
