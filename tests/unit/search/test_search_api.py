from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from ecommerce_search.api import search as search_module
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_db_session, get_decision_provider
from ecommerce_search.api.schemas import SearchResponse
from ecommerce_search.search.lexical import LexicalHit, LexicalResult

SENSITIVE = "host=internal-db.example password=hunter2 SELECT * FROM products Traceback"


def hit(rank: int, product_id: str = "P1", score: float = 0.5) -> LexicalHit:
    return LexicalHit(
        rank=rank,
        lexical_score=score,
        product_id=product_id,
        title="HP Test Laptop",
        brand="HP",
        category="laptop",
        subcategory="business",
        description="A laptop.",
        price=Decimal("33999.00"),
        currency="INR",
        rating=Decimal("4.4"),
        review_count=12,
        availability="in_stock",
    )


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client_for(make_settings, monkeypatch, calls):
    def _make(hits=None, error=None, **settings):
        def fake(session, query, limit):
            calls.append((query, limit))
            if error is not None:
                raise error
            chosen = [hit(i + 1, f"P{i + 1}") for i in range(2)] if hits is None else hits
            return LexicalResult(tsquery="'laptop'", hits=chosen[:limit], lexical_ms=1.25)

        monkeypatch.setattr(search_module, "lexical_search", fake)
        app = create_app(make_settings(**settings))
        app.dependency_overrides[get_db_session] = lambda: object()
        return TestClient(app)

    return _make


def test_get_success_shape(client_for, calls):
    with client_for() as client:
        response = client.get("/search", params={"q": "  hp   laptop ", "top_k": 5})
    assert response.status_code == 200
    body = SearchResponse.model_validate(response.json())
    assert calls == [("hp laptop", 5)]
    assert body.query == "hp laptop"
    assert body.search_version == "v0_lexical" and body.document_version == "1"
    assert body.top_k == 5 and body.result_count == len(body.results) == 2
    assert body.tsquery == "'laptop'" and body.applied_filters == []
    assert body.latency_ms.lexical_ms == 1.25 and body.latency_ms.total_ms >= 0
    assert [r.rank for r in body.results] == [1, 2]
    raw = response.json()
    assert raw["results"][0]["price"] == "33999.00"  # a decimal string, never a float
    assert set(raw) == {
        "query",
        "search_version",
        "document_version",
        "top_k",
        "result_count",
        "results",
        "tsquery",
        "applied_filters",
        "latency_ms",
    }
    assert set(raw["results"][0]) == {
        "rank",
        "lexical_score",
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


def test_post_matches_get(client_for):
    with client_for() as client:
        got = client.get("/search", params={"q": "laptop"}).json()
        posted = client.post("/search", json={"query": "laptop"}).json()
    for payload in (got, posted):
        payload["latency_ms"] = None
    assert got == posted
    assert got["top_k"] == 10  # settings default


def test_result_count_is_the_number_returned_not_the_number_of_matches(client_for):
    many = [hit(i + 1, f"P{i + 1}") for i in range(30)]
    with client_for(hits=many) as client:
        body = client.get("/search", params={"q": "laptop", "top_k": 3}).json()
    assert body["result_count"] == 3 == len(body["results"]) and body["top_k"] == 3


def test_zero_results_and_punctuation_only_are_200(client_for):
    with client_for(hits=[]) as client:
        for query in ("zzqxv", "!!!"):
            response = client.get("/search", params={"q": query})
            assert response.status_code == 200
            body = response.json()
            assert body["results"] == [] and body["result_count"] == 0


@pytest.mark.parametrize("query", ["", "   ", "\t", "lap\x00top", "a" * 201])
def test_invalid_query_text_is_422_on_get_and_post(client_for, calls, query):
    with client_for() as client:
        assert client.get("/search", params={"q": query}).status_code == 422
        assert client.post("/search", json={"query": query}).status_code == 422
    assert calls == []  # validation happens before any retrieval


@pytest.mark.parametrize("top_k", [0, -1, 51, 1000])
def test_top_k_out_of_bounds_is_422(client_for, calls, top_k):
    with client_for() as client:
        assert client.get("/search", params={"q": "laptop", "top_k": top_k}).status_code == 422
        response = client.post("/search", json={"query": "laptop", "top_k": top_k})
        assert response.status_code == 422
    assert calls == []


def test_top_k_boundaries_follow_settings(client_for):
    with client_for(search_lexical_k=7, search_default_top_k=3) as client:
        assert client.get("/search", params={"q": "x", "top_k": 7}).status_code == 200
        assert client.get("/search", params={"q": "x", "top_k": 8}).status_code == 422
        assert client.get("/search", params={"q": "x"}).json()["top_k"] == 3
    with client_for(search_max_query_length=5) as client:
        assert client.get("/search", params={"q": "abcde"}).status_code == 200
        assert client.get("/search", params={"q": "abcdef"}).status_code == 422


def test_missing_and_unknown_inputs_are_422(client_for):
    with client_for() as client:
        assert client.get("/search").status_code == 422
        assert client.post("/search", json={}).status_code == 422
        assert client.post("/search", json={"query": "x", "filters": {}}).status_code == 422
        assert client.post("/search", content="not json").status_code == 422


def test_database_failure_is_a_fixed_generic_503(client_for, caplog):
    boom = OperationalError("SELECT secret", {}, Exception(SENSITIVE))
    with client_for(error=boom) as client, caplog.at_level("ERROR"):
        for response in (
            client.get("/search", params={"q": "laptop"}),
            client.post("/search", json={"query": "laptop"}),
        ):
            assert response.status_code == 503
            assert response.json() == {"detail": "search unavailable"}
            for leaked in ("internal-db", "hunter2", "Traceback", "SELECT", "unit-test-password"):
                assert leaked not in response.text
    assert "OperationalError" in caplog.text
    for leaked in ("internal-db", "hunter2", "Traceback", "SELECT"):
        assert leaked not in caplog.text


def test_search_503_handler_is_route_local_so_health_keeps_its_contract(client_for):
    with client_for() as client:
        # No global exception handler is registered for database errors.
        assert not client.app.exception_handlers.keys() & {OperationalError, Exception}


def test_openapi_documents_both_routes_truthfully(client_for):
    with client_for() as client:
        spec = client.get("/openapi.json").json()
    for method in ("get", "post"):
        responses = spec["paths"]["/search"][method]["responses"]
        assert {"200", "422", "503"} <= set(responses)
        assert "SearchUnavailable" in str(responses["503"])
    schemas = spec["components"]["schemas"]
    assert {"SearchRequest", "SearchResponse", "SearchResult", "SearchLatency"} <= set(schemas)
    properties = set(schemas["SearchResponse"]["properties"])
    for placeholder in ("parsed_query", "decision_provider", "reranker", "filters"):
        assert placeholder not in properties
    assert "applied_filters" in properties
    assert "not a" in schemas["SearchResult"]["properties"]["lexical_score"]["description"]
    assert "not the total" in schemas["SearchResponse"]["properties"]["result_count"]["description"]


# JSON escape sequences (ASCII text that decodes to the unsafe character on the server).
UNSAFE_JSON_QUERIES = {
    "lone high surrogate": "\\ud800",
    "lone low surrogate": "\\udc00",
    "embedded surrogate": "lap\\ud800top",
    "NUL": "lap\\u0000top",
    "unsafe control character": "lap\\u0007top",
    "zero-width space": "lap\\u200btop",
    "right-to-left override": "\\u202elaptop",
}


@pytest.mark.parametrize("escaped", UNSAFE_JSON_QUERIES.values(), ids=UNSAFE_JSON_QUERIES.keys())
def test_post_rejects_invalid_unicode_with_422_never_500(client_for, calls, escaped):
    body = ('{"query": "' + escaped + '"}').encode("ascii")
    with client_for() as client:
        response = client.post(
            "/search", content=body, headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert "surrogate" in response.text
    assert escaped.lstrip("\\") not in response.text  # the offending text is never echoed
    assert calls == []  # the database layer never sees the text


@pytest.mark.parametrize(
    "encoded",
    ["lap%E2%80%8Btop", "%E2%80%AElaptop", "lap%00top", "lap%07top"],
    ids=["zero-width space", "RTL override", "NUL", "unsafe control character"],
)
def test_get_rejects_the_same_unsafe_text_with_422(client_for, calls, encoded):
    with client_for() as client:
        response = client.get("/search?q=" + encoded)
    assert response.status_code == 422 and calls == []


def test_get_cannot_carry_a_lone_surrogate_and_invalid_utf8_never_causes_a_500(client_for, calls):
    # Percent-encoded invalid UTF-8 (an encoded surrogate) is decoded with replacement
    # characters by the HTTP layer, so a GET cannot deliver a lone surrogate. What matters is
    # that nothing unencodable reaches the database driver and there is no 500.
    with client_for() as client:
        response = client.get("/search?q=%ED%A0%80")
    assert response.status_code in (200, 422)
    for query, _limit in calls:
        query.encode("utf-8")


@pytest.mark.parametrize("text", ["नमस्ते", "café", "ラップ"])
def test_valid_non_ascii_text_is_accepted_on_get_and_post(client_for, calls, text):
    with client_for() as client:
        got = client.get("/search", params={"q": text})
        posted = client.post("/search", json={"query": text})
    assert got.status_code == posted.status_code == 200
    assert got.json()["query"] == posted.json()["query"] == text
    assert [c[0] for c in calls] == [text, text]


def test_unsafe_text_is_never_logged_or_echoed(client_for, caplog):
    with client_for() as client, caplog.at_level("DEBUG"):
        response = client.post(
            "/search",
            content=b'{"query": "lap\\ud800top"}',
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422
    assert "ud800" not in response.text and "ud800" not in caplog.text


class RaisingProvider:
    """Installed to prove `/search` never invokes the Milestone 6 decision provider."""

    def __init__(self) -> None:
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        raise RuntimeError("the lexical endpoint must not call the decision provider")


def test_lexical_search_never_invokes_the_decision_provider(client_for, calls):
    provider = RaisingProvider()
    with client_for() as client:
        client.app.state.decision_provider = provider
        client.app.dependency_overrides[get_decision_provider] = lambda: provider
        got = client.get("/search", params={"q": "hp laptop 8gb"})
        posted = client.post("/search", json={"query": "hp laptop 8gb"})
        punctuation = client.get("/search", params={"q": "!!!"})
    assert got.status_code == posted.status_code == punctuation.status_code == 200
    assert provider.calls == 0
    assert calls == [("hp laptop 8gb", 10), ("hp laptop 8gb", 10), ("!!!", 10)]
    assert "query_understanding" not in got.text and "query_understanding" not in posted.text


def test_openapi_lexical_components_keep_their_m5_fields(client_for):
    with client_for() as client:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(schemas["SearchResponse"]["properties"]) == {
        "query",
        "search_version",
        "document_version",
        "top_k",
        "result_count",
        "results",
        "tsquery",
        "applied_filters",
        "latency_ms",
    }
    assert set(schemas["SearchLatency"]["properties"]) == {"lexical_ms", "total_ms"}
    assert set(schemas["SearchResult"]["properties"]) == {
        "rank",
        "lexical_score",
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
    assert set(schemas["SearchRequest"]["properties"]) == {"query", "top_k"}
