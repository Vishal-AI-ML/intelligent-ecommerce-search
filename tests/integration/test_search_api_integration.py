"""`GET /search` and `POST /search` against real PostgreSQL scratch databases."""

import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from ecommerce_search.api.app import create_app
from ecommerce_search.api.schemas import HealthResponse, SearchResponse
from ecommerce_search.search.documents import DOCUMENT_VERSION

pytestmark = pytest.mark.integration


@pytest.fixture
def client(settings, seeded_engine):
    scratch = settings.model_copy(update={"postgres_db": seeded_engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    with TestClient(create_app(scratch)) as test_client:
        yield test_client


def test_get_returns_typed_ranked_products(client):
    response = client.get("/search", params={"q": "  hp   laptop ", "top_k": 5})
    assert response.status_code == 200
    body = SearchResponse.model_validate(response.json())
    assert body.query == "hp laptop" and body.search_version == "v0_lexical"
    assert body.document_version == DOCUMENT_VERSION and body.top_k == 5
    assert body.result_count == len(body.results) == 5
    assert body.tsquery == "'hp' & 'laptop'" and body.applied_filters == []
    assert [r.rank for r in body.results] == [1, 2, 3, 4, 5]
    assert all(r.brand == "HP" and r.category == "laptop" for r in body.results)
    scores = [r.lexical_score for r in body.results]
    assert scores == sorted(scores, reverse=True) and all(s >= 0 for s in scores)
    assert body.latency_ms.lexical_ms > 0 and body.latency_ms.total_ms >= body.latency_ms.lexical_ms
    price = response.json()["results"][0]["price"]
    assert isinstance(price, str) and price.count(".") == 1  # decimal string, no float loss


def test_post_and_get_share_validation_and_retrieval(client):
    got = client.get("/search", params={"q": "wireless headphones", "top_k": 50}).json()
    posted = client.post("/search", json={"query": "wireless headphones", "top_k": 50}).json()
    for payload in (got, posted):
        payload.pop("latency_ms")
    assert got == posted and got["result_count"] == 40  # complete set fits in top_k=50


def test_result_count_counts_returned_results_not_matches(client):
    body = client.get("/search", params={"q": "laptop", "top_k": 3}).json()
    assert body["result_count"] == 3 == len(body["results"])
    default = client.get("/search", params={"q": "laptop"}).json()
    assert default["top_k"] == 10 and default["result_count"] == 10
    widest = client.get("/search", params={"q": "laptop", "top_k": 50}).json()
    assert widest["result_count"] == 50  # 80 products match; the endpoint never returns more


def test_zero_result_punctuation_and_nonsense_queries_are_clean_200s(client):
    for query in ("iphone", "zzqxv", "!!!", '"', "-"):
        for response in (
            client.get("/search", params={"q": query}),
            client.post("/search", json={"query": query}),
        ):
            assert response.status_code == 200, query
            body = response.json()
            assert body["results"] == [] and body["result_count"] == 0


def test_validation_errors_are_422(client):
    bad_requests = [
        client.get("/search", params={"q": ""}),
        client.get("/search", params={"q": "   "}),
        client.get("/search", params={"q": "a\x00b"}),
        client.get("/search", params={"q": "x" * 201}),
        client.get("/search", params={"q": "laptop", "top_k": 0}),
        client.get("/search", params={"q": "laptop", "top_k": 51}),
        client.get("/search"),
        client.post("/search", json={"query": ""}),
        client.post("/search", json={"query": "a\x00b"}),
        client.post("/search", json={"query": "x" * 201}),
        client.post("/search", json={"query": "laptop", "top_k": 0}),
        client.post("/search", json={"query": "laptop", "top_k": 51}),
        client.post("/search", json={"query": "laptop", "filters": {}}),
    ]
    assert [r.status_code for r in bad_requests] == [422] * len(bad_requests)
    assert client.get("/search", params={"q": "x" * 200}).status_code == 200  # the boundary


def test_case_and_whitespace_variants_return_the_same_list(client):
    def ids(query):
        return [
            r["product_id"] for r in client.get("/search", params={"q": query}).json()["results"]
        ]

    assert ids("HP Laptop") == ids("hp   laptop") == ids("  hp\tlaptop ") == ids("hp laptop")


def test_repeated_requests_are_deterministic(client):
    first = client.get("/search", params={"q": "laptop", "top_k": 50}).json()["results"]
    for _ in range(3):
        assert client.get("/search", params={"q": "laptop", "top_k": 50}).json()["results"] == first


def test_health_contract_is_unchanged_next_to_search(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert HealthResponse.model_validate(response.json()).status == "ok"


def test_missing_search_table_is_a_fixed_generic_503(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database, "0002")  # no search index table exists
    scratch = settings.model_copy(update={"postgres_db": scratch_database})
    with TestClient(create_app(scratch)) as test_client:
        responses = [
            test_client.get("/search", params={"q": "laptop"}),
            test_client.post("/search", json={"query": "laptop"}),
        ]
        health = test_client.get("/health")
    for response in responses:
        assert response.status_code == 503
        assert response.json() == {"detail": "search unavailable"}
        for leaked in (
            "product_search_documents",
            "relation",
            "SELECT",
            "psycopg",
            settings.postgres_password.get_secret_value(),
        ):
            assert leaked not in response.text
    assert health.status_code == 200  # /health keeps its own contract (database and pgvector ok)


def test_unreachable_database_is_a_fixed_generic_503(settings):
    unreachable = settings.model_copy(update={"postgres_port": 1})
    with TestClient(create_app(unreachable)) as test_client:
        response = test_client.get("/search", params={"q": "laptop"})
        health = test_client.get("/health")
    assert response.status_code == 503 and response.json() == {"detail": "search unavailable"}
    assert health.status_code == 503  # the existing health 503 contract is untouched
    assert HealthResponse.model_validate(health.json()).status == "unavailable"


def test_missing_documents_shrink_results_and_search_does_not_claim_completeness(
    settings, migrated_engine
):
    """Until `reindex` succeeds, a missing document simply means fewer results; the response
    carries no completeness claim (document_version is the code's version, not a status)."""
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    ingest_file(migrated_engine, SEED, PROVENANCE)
    scratch = settings.model_copy(update={"postgres_db": migrated_engine.url.database})
    with TestClient(create_app(scratch)) as test_client:
        full = test_client.get("/search", params={"q": "hp laptop", "top_k": 50}).json()
        with migrated_engine.begin() as conn:
            conn.execute(
                text("DELETE FROM product_search_documents WHERE product_id = :p"),
                {"p": full["results"][0]["product_id"]},
            )
        shrunk = test_client.get("/search", params={"q": "hp laptop", "top_k": 50}).json()
    assert full["result_count"] == 12 and shrunk["result_count"] == 11
    assert not {"complete", "index_status", "total_matches"} & set(shrunk)


# ---- I1/I6 D: Unicode handling against real PostgreSQL ---------------------------------------

UNSAFE_JSON = {
    "lone high surrogate": "\\ud800",
    "lone low surrogate": "\\udc00",
    "embedded surrogate": "lap\\ud800top",
    "NUL": "lap\\u0000top",
    "unsafe control character": "lap\\u0007top",
    "zero-width space": "lap\\u200btop",
    "right-to-left override": "\\u202elaptop",
}


@pytest.mark.parametrize("escaped", UNSAFE_JSON.values(), ids=UNSAFE_JSON.keys())
def test_unsafe_unicode_is_422_on_post_never_500(client, escaped):
    body = ('{"query": "' + escaped + '"}').encode("ascii")
    response = client.post("/search", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert escaped.lstrip("\\") not in response.text


@pytest.mark.parametrize(
    "encoded",
    ["lap%E2%80%8Btop", "%E2%80%AElaptop", "lap%00top", "lap%07top", "%ED%A0%80"],
)
def test_get_never_returns_500_for_unsafe_or_invalid_utf8_text(client, encoded):
    response = client.get("/search?q=" + encoded)
    assert response.status_code in (200, 422)
    if encoded != "%ED%A0%80":  # replacement characters are ordinary text; the rest is unsafe
        assert response.status_code == 422


@pytest.mark.parametrize(
    "text_value",
    [
        "नमस्ते",
        "می‌خواهم",
        "café",
        "ラップトップ",
    ],
)
def test_valid_non_ascii_text_reaches_postgresql_and_matches_nothing(client, text_value):
    got = client.get("/search", params={"q": text_value})
    posted = client.post("/search", json={"query": text_value})
    assert got.status_code == posted.status_code == 200
    assert got.json()["query"] == posted.json()["query"] == text_value
    assert got.json()["results"] == posted.json()["results"] == []


def test_missing_table_503_logs_an_operator_hint_but_not_details(
    settings, scratch_database, migrator, caplog
):
    migrator.upgrade(scratch_database, "0002")
    # Alembic's fileConfig (run by the migration) removes root handlers, including caplog's.
    logging.getLogger().addHandler(caplog.handler)
    scratch = settings.model_copy(update={"postgres_db": scratch_database})
    with TestClient(create_app(scratch)) as test_client, caplog.at_level("ERROR"):
        response = test_client.get("/search", params={"q": "laptop"})
    assert response.status_code == 503 and response.json() == {"detail": "search unavailable"}
    assert "alembic upgrade head" in caplog.text and "reindex" in caplog.text
    for leaked in (
        "SELECT",
        "product_search_documents",
        settings.postgres_password.get_secret_value(),
    ):
        assert leaked not in caplog.text
