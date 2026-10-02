"""`GET/POST /search/hybrid` with a fake embedder, an instrumented fake session and fake source
retrieval functions (no DB, no model). Records the order of model work and transaction events."""

import json
import logging
from contextlib import contextmanager
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, ProgrammingError

from ecommerce_search.api import hybrid as hybrid_api
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_db_session, get_embedder
from ecommerce_search.api.schemas import HybridSearchResponse
from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.search import hybrid as hybrid_search
from ecommerce_search.search.dense import DenseHit, DenseResult
from ecommerce_search.search.lexical import LexicalHit, LexicalResult
from fake_embedder import FakeEmbedder

SENSITIVE = "host=internal-db.example password=hunter2 SELECT * FROM products D:\\secret\\path"
FIXED_503 = {"detail": "search unavailable"}
SNAPSHOT = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"


def _fields(product_id: str, source: str) -> dict:
    return {
        "product_id": product_id,
        "title": f"{source} title {product_id}",
        "brand": "Nike",
        "category": "shoes",
        "subcategory": "running",
        "description": "Shoes.",
        "price": Decimal("4299.00"),
        "currency": "INR",
        "rating": Decimal("4.4"),
        "review_count": 7,
        "availability": "in_stock",
    }


def lexical_hits(*ids: str) -> list[LexicalHit]:
    return [
        LexicalHit(rank=r, lexical_score=1.0 / r, **_fields(p, "lexical"))
        for r, p in enumerate(ids, start=1)
    ]


def dense_hits(*ids: str) -> list[DenseHit]:
    return [
        DenseHit(rank=r, dense_score=1.0 - r / 100, **_fields(p, "dense"))
        for r, p in enumerate(ids, start=1)
    ]


class FakeSession:
    """Records transaction events; `in_transaction()` mirrors SQLAlchemy's Session."""

    def __init__(self, events: list, execute_error: Exception | None = None) -> None:
        self.events = events
        self.execute_error = execute_error
        self._open = False
        self.closed = False

    def in_transaction(self) -> bool:
        return self._open

    @contextmanager
    def begin(self):
        assert not self._open, "nested transaction"
        self.events.append("begin")
        self._open = True
        try:
            yield self
        except BaseException:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")
        finally:
            self._open = False

    def execute(self, statement, params=None):
        assert self._open, "statement outside the transaction"
        self.events.append(("execute", str(statement)))
        if self.execute_error is not None:
            raise self.execute_error

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class RecordingEmbedder(FakeEmbedder):
    """Records whether a transaction was open while the model loaded or encoded."""

    def __init__(self, events: list, sessions: list) -> None:
        super().__init__()
        self.events = events
        self.sessions = sessions

    def _in_tx(self) -> bool:
        return any(session.in_transaction() for session in self.sessions)

    def load(self):
        self.events.append(("load", self._in_tx()))
        return super().load()

    def embed_query(self, text):
        self.events.append(("embed", self._in_tx()))
        return super().embed_query(text)


@pytest.fixture
def events():
    return []


@pytest.fixture
def client_for(make_settings, monkeypatch, events):
    def _make(
        embedder="recording",
        lexical=None,
        dense=None,
        lexical_error=None,
        dense_error=None,
        execute_error=None,
        **settings,
    ):
        sessions: list[FakeSession] = []
        fake = RecordingEmbedder(events, sessions) if embedder == "recording" else embedder

        def fake_lexical(session, query, limit):
            assert session.in_transaction()
            events.append(("lexical", query, limit))
            if lexical_error is not None:
                raise lexical_error
            hits = lexical_hits("P1", "P2", "P3") if lexical is None else lexical
            return LexicalResult(tsquery="'x'", hits=hits[:limit], lexical_ms=1.5)

        def fake_dense(session, vector, spec, limit):
            assert session.in_transaction()
            events.append(("dense", len(vector), limit))
            if dense_error is not None:
                raise dense_error
            hits = dense_hits("P3", "P4") if dense is None else dense
            return DenseResult(hits=hits[:limit], vector_ms=2.5)

        monkeypatch.setattr(hybrid_search, "lexical_search", fake_lexical)
        monkeypatch.setattr(hybrid_search, "dense_search", fake_dense)

        def session_dependency():
            session = FakeSession(events, execute_error)
            sessions.append(session)
            try:
                yield session
            finally:
                session.close()

        app = create_app(make_settings(**settings))
        app.dependency_overrides[get_db_session] = session_dependency
        app.dependency_overrides[get_embedder] = lambda: fake
        client = TestClient(app)
        client.fake = fake
        client.sessions = sessions
        return client

    return _make


def test_get_success_shape_and_truthful_metadata(client_for, events):
    with client_for() as client:
        response = client.get("/search/hybrid", params={"q": "  running   shoes ", "top_k": 3})
    assert response.status_code == 200
    body = HybridSearchResponse.model_validate(response.json())
    assert body.query == "running shoes" and body.search_version == "v1_hybrid"
    assert body.document_version and body.tsquery == "'x'"
    assert body.embedding_model_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert body.embedding_dimension == 384 and body.distance_metric == "cosine"
    fusion = body.fusion
    assert (fusion.method, fusion.rrf_k, fusion.rrf_k_status) == ("rrf", 100, "provisional")
    assert (fusion.lexical_k, fusion.dense_k, fusion.candidate_k) == (50, 50, 50)
    assert body.dense_status == "used"
    assert (body.lexical_hit_count, body.dense_hit_count, body.overlap_count) == (3, 2, 1)
    assert (body.fused_count, body.candidate_count) == (4, 4)
    assert body.top_k == 3 and body.result_count == len(body.results) == 3
    # P3: lexical 3 + dense 1; P1: lexical 1; then P2 (lexical 2) and P4 (dense 2) tie -> P2.
    assert [(r.rank, r.product_id, r.lexical_rank, r.dense_rank) for r in body.results] == [
        (1, "P3", 3, 1),
        (2, "P1", 1, None),
        (3, "P2", 2, None),
    ]
    assert body.results[0].rrf_score == pytest.approx(1 / 103 + 1 / 101)
    assert body.results[0].title == "lexical title P3"  # lexical display fields win
    assert body.results[1].dense_score is None and body.results[1].lexical_score == 1.0
    latency = body.latency_ms
    assert (latency.lexical_ms, latency.vector_ms, latency.model_load_ms) == (1.5, 2.5, 1.0)
    assert latency.rrf_ms >= 0 and latency.query_embedding_ms >= 0 and latency.total_ms >= 0
    raw = response.json()
    assert raw["results"][0]["price"] == "4299.00" and raw["applied_filters"] == []
    assert "fallback" not in response.text
    assert ("lexical", "running shoes", 50) in events and ("dense", 384, 50) in events


def test_model_work_completes_before_the_transaction_and_it_ends_before_fusion(
    client_for, events, monkeypatch
):
    original = hybrid_api.fuse_rrf

    def recording_fuse(*args, **kwargs):
        events.append("fuse")
        return original(*args, **kwargs)

    monkeypatch.setattr(hybrid_api, "fuse_rrf", recording_fuse)
    with client_for() as client:
        assert client.get("/search/hybrid", params={"q": "shoes"}).status_code == 200
    names = [e if isinstance(e, str) else e[0] for e in events]
    assert names == [
        "load",
        "load",  # count_tokens
        "embed",
        "load",  # inside embed_query
        "begin",
        "execute",
        "lexical",
        "dense",
        "commit",
        "fuse",
        "close",
    ]
    assert all(e[1] is False for e in events if e[0] in ("load", "embed"))
    assert ("execute", SNAPSHOT) in events  # the first statement of the transaction


def test_get_and_post_share_one_path(client_for):
    with client_for() as client:
        got = client.get("/search/hybrid", params={"q": "wireless headphones", "top_k": 4}).json()
        posted = client.post(
            "/search/hybrid", json={"query": "wireless headphones", "top_k": 4}
        ).json()
    for payload in (got, posted):
        payload.pop("latency_ms")
        assert (payload["fusion"]["rrf_k"], payload["fusion"]["rrf_k_status"]) == (
            100,
            "provisional",
        )
    assert got == posted


def test_an_explicit_rrf_k_setting_is_used_and_reported(client_for):
    with client_for(search_rrf_k=60) as client:
        got = client.get("/search/hybrid", params={"q": "shoes", "top_k": 1}).json()
        posted = client.post("/search/hybrid", json={"query": "shoes", "top_k": 1}).json()
    for payload in (got, posted):
        assert (payload["fusion"]["rrf_k"], payload["fusion"]["rrf_k_status"]) == (
            60,
            "provisional",
        )
        assert payload["results"][0]["rrf_score"] == pytest.approx(1 / 63 + 1 / 61)


def test_top_k_results_are_a_prefix_of_larger_top_k(client_for):
    lexical = lexical_hits(*[f"L{n:02d}" for n in range(30)])
    dense = dense_hits(*[f"L{n:02d}" for n in range(29, -1, -1)], *[f"D{n:02d}" for n in range(20)])
    with client_for(lexical=lexical, dense=dense) as client:
        bodies = {
            k: client.get("/search/hybrid", params={"q": "x", "top_k": k}).json()
            for k in (1, 5, 10, 50)
        }
    full = [r["product_id"] for r in bodies[50]["results"]]
    assert len(full) == 50 == bodies[50]["candidate_count"] and bodies[50]["fused_count"] == 50
    for k in (1, 5, 10):
        assert [r["product_id"] for r in bodies[k]["results"]] == full[:k]
        assert bodies[k]["candidate_count"] == 50  # truncation is independent of top_k


def test_candidate_k_truncates_the_fused_list(client_for):
    with client_for(search_candidate_k=3, search_default_top_k=3) as client:
        body = client.get("/search/hybrid", params={"q": "x"}).json()
        too_many = client.get("/search/hybrid", params={"q": "x", "top_k": 4})
    assert body["fused_count"] == 4 and body["candidate_count"] == 3 and body["top_k"] == 3
    assert too_many.status_code == 422


def test_source_depths_come_from_settings_independent_of_top_k(client_for, events):
    with client_for(
        search_lexical_k=7, search_dense_k=9, search_candidate_k=16, search_default_top_k=2
    ) as client:
        body = client.get("/search/hybrid", params={"q": "shoes"}).json()
    assert ("lexical", "shoes", 7) in events and ("dense", 384, 9) in events
    assert body["fusion"]["lexical_k"] == 7 and body["fusion"]["dense_k"] == 9
    assert body["top_k"] == 2


def test_lexical_zero_hits_returns_dense_ranked_results(client_for):
    with client_for(lexical=[]) as client:
        body = client.get("/search/hybrid", params={"q": "zzqxv"}).json()
    assert body["lexical_hit_count"] == 0 and body["dense_hit_count"] == 2
    assert [(r["product_id"], r["lexical_rank"]) for r in body["results"]] == [
        ("P3", None),
        ("P4", None),
    ]
    assert body["results"][0]["title"] == "dense title P3"


def test_zero_dense_hits_is_a_200_with_fewer_candidates_and_no_fallback_label(client_for):
    with client_for(dense=[]) as client:
        response = client.get("/search/hybrid", params={"q": "shoes"})
    body = response.json()
    assert response.status_code == 200 and body["dense_status"] == "used"
    assert body["dense_hit_count"] == 0 and body["candidate_count"] == 3
    assert "fallback" not in response.text.lower()


@pytest.mark.parametrize("query", ["!!!", "-", '"', "...  ???", "\u2014"])
def test_no_searchable_text_skips_model_and_database(client_for, events, query):
    with client_for() as client:
        response = client.get("/search/hybrid", params={"q": query})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["result_count"] == 0
    assert body["dense_status"] == "skipped_no_searchable_text" and body["tsquery"] is None
    assert body["lexical_hit_count"] == body["dense_hit_count"] == body["candidate_count"] == 0
    assert all(value is None for key, value in body["latency_ms"].items() if key != "total_ms")
    assert client.fake.load_calls == 0
    assert events == ["close"]  # no begin, no statement, no model work


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
    client_for, events, params, location
):
    with client_for() as client:
        response = client.get("/search/hybrid", params=params)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == location
    assert "shoes\x00" not in response.text and "x" * 201 not in response.text
    assert "begin" not in events and client.fake.load_calls == 0


@pytest.mark.parametrize(
    "body",
    [
        {"query": "\ud800"},
        {"query": "shoes", "top_k": 0},
        {"query": "shoes", "top_k": 51},
        {"query": "shoes", "extra": 1},
        {"query": "shoes", "rrf_k": 10},  # no per-request fusion overrides
        {"top_k": 5},
        {"query": 5},
    ],
)
def test_post_validation_errors_are_422(client_for, events, body):
    with client_for() as client:
        response = client.post(
            "/search/hybrid",
            content=json.dumps(body, ensure_ascii=True),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert "begin" not in events


def test_get_ignores_unknown_fusion_parameters(client_for):
    with client_for() as client:
        body = client.get("/search/hybrid", params={"q": "shoes", "rrf_k": 1, "lexical_k": 2})
    assert body.json()["fusion"]["rrf_k"] == 100 and body.json()["fusion"]["lexical_k"] == 50


def test_query_over_the_model_token_limit_is_422_before_any_transaction(client_for, events):
    with client_for() as client:
        client.fake.token_overrides["toolong"] = 257
        response = client.get("/search/hybrid", params={"q": "toolong query"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"].endswith("query is too long for the embedding model")
    assert "toolong" not in response.text
    assert "begin" not in events and events[-1] == "close"


@pytest.mark.parametrize(
    "reason",
    ["snapshot_missing", "load_failed", "dimension_mismatch", "config_mismatch", "encode_failed"],
)
def test_model_failures_are_a_fixed_503_before_any_transaction(client_for, events, caplog, reason):
    fake = FakeEmbedder()
    fake.fail_after_batches = 0
    fake.failure = EmbedderUnavailable(reason)
    with caplog.at_level(logging.ERROR), client_for(embedder=fake) as client:
        response = client.get("/search/hybrid", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    assert f"hybrid search unavailable: {reason}" in caplog.text
    assert events == ["close"]


def test_no_models_directory_is_a_fixed_503(client_for, events, caplog):
    with caplog.at_level(logging.ERROR), client_for(embedder=None) as client:
        response = client.get("/search/hybrid", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    assert "snapshot_missing" in caplog.text and "model-fetch" in caplog.text
    assert events == ["close"]


def _operational():
    return OperationalError(SENSITIVE, {"p": "hunter2"}, Exception(SENSITIVE))


def _missing_table():
    return ProgrammingError(SENSITIVE, {}, type("E", (Exception,), {"sqlstate": "42P01"})())


@pytest.mark.parametrize(
    ("failure", "logged", "expected_tail"),
    [
        ({"execute_error": _operational()}, "database_error", []),
        ({"lexical_error": _operational()}, "database_error", ["lexical"]),
        ({"lexical_error": _missing_table()}, "table_missing", ["lexical"]),
        ({"dense_error": _operational()}, "database_error", ["lexical", "dense"]),
        ({"dense_error": _missing_table()}, "table_missing", ["lexical", "dense"]),
    ],
)
def test_database_failures_roll_back_and_are_a_fixed_503_that_leaks_nothing(
    client_for, events, caplog, failure, logged, expected_tail
):
    with caplog.at_level(logging.DEBUG), client_for(**failure) as client:
        response = client.get("/search/hybrid", params={"q": "shoes"})
    assert response.status_code == 503 and response.json() == FIXED_503
    for secret in ("hunter2", "internal-db", "SELECT", "secret\\path", "Traceback"):
        assert secret not in response.text
        assert secret not in caplog.text
    assert f"hybrid search unavailable: {logged}" in caplog.text
    names = [e if isinstance(e, str) else e[0] for e in events]
    assert names == ["load", "load", "embed", "load", "begin", "execute"] + expected_tail + [
        "rollback",
        "close",
    ]
    assert not client.sessions[0].in_transaction() and client.sessions[0].closed


def test_fusion_error_happens_after_the_transaction_ended(client_for, events, monkeypatch):
    def broken_fuse(*args, **kwargs):
        events.append("fuse")
        raise ValueError("lexical ranks must be contiguous 1..n in list order")

    monkeypatch.setattr(hybrid_api, "fuse_rrf", broken_fuse)
    with client_for() as client:
        client_no_raise = TestClient(client.app, raise_server_exceptions=False)
        response = client_no_raise.get("/search/hybrid", params={"q": "shoes"})
    assert response.status_code == 500 and "contiguous" not in response.text
    names = [e if isinstance(e, str) else e[0] for e in events]
    assert names[names.index("begin") :] == [
        "begin",
        "execute",
        "lexical",
        "dense",
        "commit",
        "fuse",
        "close",
    ]


@pytest.mark.parametrize(
    ("lexical", "dense"),
    [
        (lexical_hits("P1", "P1"), None),  # duplicate product id
        (None, dense_hits("P3", "P4")[::-1]),  # ranks out of list order
    ],
)
def test_malformed_source_list_is_a_generic_500_not_a_503(
    client_for, events, caplog, lexical, dense
):
    # A malformed source list is a programming error: it must not be disguised as the
    # service-unavailable 503, and neither the body nor the app log may leak details.
    with caplog.at_level(logging.DEBUG), client_for(lexical=lexical, dense=dense) as client:
        client_no_raise = TestClient(client.app, raise_server_exceptions=False)
        response = client_no_raise.get("/search/hybrid", params={"q": "hunter2 shoes"})
    assert response.status_code == 500 and response.text == "Internal Server Error"
    # The application's own log records (the test client's request log is not the app's).
    app_log = "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("ecommerce_")
    )
    for secret in ("hunter2", "contiguous", "unique", "Traceback"):
        assert secret not in response.text
        assert secret not in app_log
    assert "unavailable" not in app_log
    names = [e if isinstance(e, str) else e[0] for e in events]
    assert names[names.index("begin") :] == [
        "begin",
        "execute",
        "lexical",
        "dense",
        "commit",
        "close",
    ]


def test_openapi_states_when_latency_fields_are_null(client_for):
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    latency = schema["components"]["schemas"]["HybridSearchLatency"]["properties"]
    for field in ("lexical_ms", "model_load_ms", "query_embedding_ms", "vector_ms", "rrf_ms"):
        assert "null" in latency[field]["description"]
        assert {"type": "null"} in latency[field]["anyOf"]
    assert latency["total_ms"]["type"] == "number"


def test_read_sources_refuses_a_session_with_an_open_transaction(events):
    session = FakeSession(events)
    with session.begin(), pytest.raises(RuntimeError, match="open transaction"):
        hybrid_search.read_sources(session, "x", [0.0], None, 1, 1)


def test_openapi_documents_hybrid_and_keeps_existing_routes(client_for):
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    for path in ("/search", "/search/dense", "/search/hybrid"):
        assert {"get", "post"} <= set(schema["paths"][path])
    hybrid = schema["paths"]["/search/hybrid"]["get"]
    assert "Reciprocal Rank Fusion" in hybrid["description"]
    assert "not probabilities" in hybrid["description"]
    assert set(hybrid["responses"]) >= {"200", "422", "503"}
    assert schema["paths"]["/search"]["get"]["summary"] == "Lexical search (V0)"
    assert (
        schema["paths"]["/search/dense"]["get"]["summary"] == "Dense semantic search (Milestone 4)"
    )
    params = {p["name"] for p in hybrid["parameters"]}
    assert params == {"q", "top_k"}
