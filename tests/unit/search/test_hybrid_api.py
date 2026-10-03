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
from ecommerce_search.api.dependencies import (
    get_db_session,
    get_decision_provider,
    get_embedder,
)
from ecommerce_search.api.schemas import HybridSearchResponse
from ecommerce_search.catalog.taxonomy import Category, StorageType
from ecommerce_search.decision import DecisionResult, DeterministicDecisionProvider
from ecommerce_search.decision import provider as decision_module
from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.query_understanding import (
    Ambiguity,
    AmbiguityReason,
    Conflict,
    ConflictReason,
    Intent,
    MatchedTerm,
    MatchRule,
    QueryAttributes,
    QueryUnderstanding,
    SourceSpan,
    UnderstandingField,
    parse_normalized_query,
)
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

    def connection(self, *args, **kwargs):
        self.events.append("connection")
        raise AssertionError("the hybrid route never asks the session for a connection")

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
        provider=None,
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
        if provider is not None:
            app.dependency_overrides[get_decision_provider] = lambda: provider
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
    executed = ("total_ms", "query_understanding_ms")  # understanding runs before this check
    assert all(value is None for key, value in body["latency_ms"].items() if key not in executed)
    assert body["latency_ms"]["total_ms"] >= body["latency_ms"]["query_understanding_ms"] >= 0
    block = body["query_understanding"]
    assert block["usage"] == "informational" and block["provider"] == "deterministic"
    assert block["understanding"]["raw_query"] == body["query"]
    assert body["applied_filters"] == []
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


# ---- query understanding (Milestone 6): informational only -------------------------------------

PARSED_TYPES = (
    QueryUnderstanding,
    DecisionResult,
    MatchedTerm,
    SourceSpan,
    Ambiguity,
    Conflict,
    QueryAttributes,
    Decimal,
    Category,
    StorageType,
    Intent,
)
QUERIES = ["  hp   gaming laptop ", "sasta phone 50k ke andar", "16gb ram under 50000", "!!!"]


def stub_understanding(raw_query: str, variant: str) -> QueryUnderstanding:
    """A valid parse of any query that differs between variants in every field a provider can
    change (`provider`, `provider_version` and the parser/lexicon versions are fixed literals)."""
    a = variant == "A"
    span = SourceSpan(start=0, end=1, text=raw_query[0])
    term = MatchedTerm(
        rule=MatchRule.BRAND_TERM if a else MatchRule.INTENT_TERM,
        field=UnderstandingField.BRAND if a else UnderstandingField.SEMANTIC_INTENT,
        value="stub-a" if a else "stub-b",
        span=span,
    )
    return QueryUnderstanding(
        raw_query=raw_query,
        category=Category.LAPTOP if a else Category.SHOES,
        brand="StubBrandA" if a else "StubBrandB",
        ram_gb=8 if a else 4096,
        storage_gb=512 if a else 1,
        storage_type=StorageType.SSD if a else StorageType.HDD,
        min_price=Decimal("100.00") if a else Decimal("0.01"),
        max_price=Decimal("999.99") if a else Decimal("9999999999.99"),
        semantic_intent=Intent.GAMING if a else Intent.COMFORT,
        attributes=(
            QueryAttributes(storage_interface="NVME")
            if a
            else QueryAttributes(price_preference="low")
        ),
        matched_terms=(term,),
        ambiguities=(
            Ambiguity(
                reason=AmbiguityReason.BARE_CAPACITY if a else AmbiguityReason.UNSUPPORTED_RANGE,
                span=span,
                value="a" if a else None,
            ),
        ),
        conflicts=(
            Conflict(
                field=term.field,
                reason=(
                    ConflictReason.REPEATED_DIFFERENT_VALUES
                    if a
                    else ConflictReason.MIN_PRICE_EXCEEDS_MAX_PRICE
                ),
                candidates=(term,),
            ),
        ),
        unresolved=(span,) if a else (),
    )


class StubProvider:
    """Records each call; returns a valid decision that differs per variant."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.calls: list[tuple[str, QueryUnderstanding]] = []

    def understand(self, query, deterministic_result):
        self.calls.append((query, deterministic_result))
        return DecisionResult(understanding=stub_understanding(query, self.variant))


class RaisingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        raise RuntimeError(SENSITIVE)


class InvalidProvider:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        return self.result


def install_spies(monkeypatch) -> list:
    """Wrap every retrieval and fusion entry point; record (name, args, kwargs, result)."""
    records: list = []
    targets = [
        (hybrid_api, "encode_query"),
        (hybrid_api, "read_sources"),
        (hybrid_api, "fuse_rrf"),
        (hybrid_search, "lexical_search"),
        (hybrid_search, "dense_search"),
    ]
    for module, name in targets:
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            records.append((_name, args, kwargs, result))
            return result

        monkeypatch.setattr(module, name, spy)
    return records


def count_normalizations(monkeypatch) -> list:
    seen: list = []
    original = hybrid_api.normalize_query

    def spy(raw, max_length):
        seen.append(raw)
        return original(raw, max_length)

    monkeypatch.setattr(hybrid_api, "normalize_query", spy)
    return seen


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _walk(item)


def assert_retrieval_saw_only(records: list, query: str) -> None:
    assert {name for name, *_ in records} == {
        "encode_query",
        "read_sources",
        "fuse_rrf",
        "lexical_search",
        "dense_search",
    }
    for name, args, kwargs, _result in records:
        arguments = list(args) + list(kwargs.values())
        for argument in arguments:
            if isinstance(argument, str):
                assert argument == query, name  # the exact normalized query, nothing else
            for nested in _walk(argument):
                assert not isinstance(nested, PARSED_TYPES), (name, type(nested))
        if name in ("encode_query", "read_sources", "lexical_search"):
            assert query in arguments, name


def retrieval_view(records: list) -> list:
    """Source lists and fused output as the spies saw them (comparable across requests)."""
    view = []
    for name, args, kwargs, result in records:
        if name == "read_sources":
            view.append((name, args[1], result[0].tsquery, result[0].hits, result[1].hits))
        elif name == "fuse_rrf":
            view.append((name, args, kwargs, result))
        elif name in ("lexical_search", "dense_search"):
            view.append((name, args[1:], result.hits))
    return view


def without_understanding(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("query_understanding")
    payload.pop("latency_ms")
    return payload


def test_default_provider_reports_the_deterministic_parse(client_for):
    with client_for() as client:
        assert isinstance(client.app.state.decision_provider, DeterministicDecisionProvider)
        response = client.get("/search/hybrid", params={"q": "  hp  laptop 8gb 256gb ssd "})
    assert response.status_code == 200
    body = response.json()
    expected = parse_normalized_query("hp laptop 8gb 256gb ssd").model_dump(mode="json")
    assert expected["brand"] is not None  # a real parse, not an empty one
    assert body["query_understanding"] == {
        "usage": "informational",
        "provider": "deterministic",
        "provider_version": "qu-1",
        "understanding": expected,
    }
    assert body["query_understanding"]["understanding"]["parser_version"] == "qu-1"
    assert body["query_understanding"]["understanding"]["lexicon_version"] == "1"
    assert body["applied_filters"] == []
    latency = body["latency_ms"]
    assert 0 <= latency["query_understanding_ms"] <= latency["total_ms"]
    HybridSearchResponse.model_validate(body)


def test_create_app_owns_one_provider_and_the_dependency_returns_it(make_settings):
    app = create_app(make_settings())
    provider = app.state.decision_provider
    assert isinstance(provider, DeterministicDecisionProvider)
    request = type("Request", (), {"app": app})()
    assert get_decision_provider(request) is provider
    assert get_decision_provider(request) is provider  # the same singleton every time
    assert create_app(make_settings()).state.decision_provider is not provider  # per app


def test_app_state_provider_is_used_by_the_route(client_for):
    stub = StubProvider("A")
    with client_for() as client:
        client.app.state.decision_provider = stub
        first = client.get("/search/hybrid", params={"q": "shoes"}).json()
        client.post("/search/hybrid", json={"query": "shoes"})
    assert [query for query, _ in stub.calls] == ["shoes", "shoes"]
    assert first["query_understanding"]["understanding"]["brand"] == "StubBrandA"


@pytest.mark.parametrize("raw", QUERIES)
def test_normalize_and_provider_run_exactly_once_per_get_and_post(client_for, monkeypatch, raw):
    stub = StubProvider("A")
    normalized = " ".join(raw.split())
    with client_for(provider=stub) as client:
        seen = count_normalizations(monkeypatch)
        got = client.get("/search/hybrid", params={"q": raw})
        assert seen == [raw] and [query for query, _ in stub.calls] == [normalized]
        posted = client.post("/search/hybrid", json={"query": raw})
        assert seen == [raw, raw]
        assert [query for query, _ in stub.calls] == [normalized, normalized]
    assert got.status_code == posted.status_code == 200
    # The provider receives the deterministic parse of the same normalized query.
    for query, deterministic in stub.calls:
        assert deterministic == parse_normalized_query(query)
    block = got.json()["query_understanding"]
    assert block == posted.json()["query_understanding"]
    assert block["understanding"] == stub_understanding(normalized, "A").model_dump(mode="json")
    assert without_understanding(got.json()) == without_understanding(posted.json())


@pytest.mark.parametrize("raw", QUERIES)
def test_retrieval_and_fusion_are_independent_of_the_provider_output(client_for, monkeypatch, raw):
    normalized = " ".join(raw.split())
    responses, views = {}, {}
    providers = {
        "A": StubProvider("A"),
        "B": StubProvider("B"),
        "deterministic": DeterministicDecisionProvider(),
    }
    for name, provider in providers.items():
        with client_for(provider=provider) as client:
            records = install_spies(monkeypatch)
            responses[name, "get"] = client.get("/search/hybrid", params={"q": raw}).json()
            responses[name, "post"] = client.post("/search/hybrid", json={"query": raw}).json()
        if normalized == "!!!":
            assert records == []  # no searchable text: nothing retrieval-side ran
        else:
            assert_retrieval_saw_only(records, normalized)
        views[name] = retrieval_view(records)
        monkeypatch.undo()
    a_understanding = responses["A", "get"]["query_understanding"]["understanding"]
    b_understanding = responses["B", "get"]["query_understanding"]["understanding"]
    changed = {key for key in a_understanding if a_understanding[key] != b_understanding[key]}
    assert changed == set(a_understanding) - {
        "raw_query",
        "retrieval_strategy",
        "parser_version",
        "lexicon_version",
    }
    reference = without_understanding(responses["deterministic", "get"])
    for key, payload in responses.items():
        assert without_understanding(payload) == reference, key
        assert payload["applied_filters"] == []
        assert payload["latency_ms"]["query_understanding_ms"] is not None
    assert views["A"] == views["B"] == views["deterministic"]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_m5_ids_ranks_scores_counts_and_fusion_are_unchanged(client_for, variant):
    with client_for(provider=StubProvider(variant)) as client:
        body = client.get("/search/hybrid", params={"q": "running shoes", "top_k": 4}).json()
    assert [
        (r["rank"], r["product_id"], r["lexical_rank"], r["dense_rank"]) for r in body["results"]
    ] == [
        (1, "P3", 3, 1),
        (2, "P1", 1, None),
        (3, "P2", 2, None),
        (4, "P4", None, 2),
    ]
    assert body["results"][0]["rrf_score"] == pytest.approx(1 / 103 + 1 / 101)
    assert body["results"][3]["rrf_score"] == pytest.approx(1 / 102)
    counts = ("lexical_hit_count", "dense_hit_count", "overlap_count", "fused_count")
    assert [body[key] for key in counts] == [3, 2, 1, 4] and body["candidate_count"] == 4
    assert body["fusion"] == {
        "method": "rrf",
        "rrf_k": 100,
        "rrf_k_status": "provisional",
        "lexical_k": 50,
        "dense_k": 50,
        "candidate_k": 50,
    }
    assert body["search_version"] == "v1_hybrid" and body["applied_filters"] == []


def _parser_failure(monkeypatch):
    def broken(query):
        raise ValueError(SENSITIVE)

    monkeypatch.setattr(decision_module, "parse_normalized_query", broken)
    return None  # the default (real) provider is used and never reached


PROVIDER_FAILURES = {
    "raising provider": lambda monkeypatch: RaisingProvider(),
    "non-DecisionResult": lambda monkeypatch: InvalidProvider({"provider": "deterministic"}),
    "result for another query": lambda monkeypatch: InvalidProvider(
        DecisionResult(understanding=QueryUnderstanding(raw_query="something else"))
    ),
    "parser raises": _parser_failure,
}


@pytest.mark.parametrize("make_provider", PROVIDER_FAILURES.values(), ids=PROVIDER_FAILURES.keys())
@pytest.mark.parametrize("method", ["get", "post"])
def test_query_understanding_failure_is_a_fixed_503_before_any_model_or_database_work(
    client_for, events, caplog, monkeypatch, make_provider, method
):
    provider = make_provider(monkeypatch)
    with caplog.at_level(logging.DEBUG), client_for(provider=provider) as client:
        records = install_spies(monkeypatch)
        if method == "get":
            response = client.get("/search/hybrid", params={"q": "hunter2 laptop"})
        else:
            response = client.post("/search/hybrid", json={"query": "hunter2 laptop"})
    assert response.status_code == 503 and response.json() == FIXED_503
    assert records == []  # no encode, read_sources, lexical, dense or fusion
    assert client.fake.load_calls == 0  # no model load or encoding (they would add events)
    assert events == ["close"]  # no begin, execute or connection on the session
    if provider is not None:
        assert provider.calls == 1
    app_records = [record for record in caplog.records if record.name.startswith("ecommerce_")]
    app_log = [record.getMessage() for record in app_records]
    # Exactly one error, with the fixed reason; the rest is app start/stop lifecycle logging.
    errors = [r.getMessage() for r in app_records if r.levelno >= logging.WARNING]
    assert errors == ["hybrid search unavailable: query_understanding_failed"]
    assert [r.name for r in app_records if r.levelno >= logging.WARNING] == [hybrid_api.__name__]
    assert all(record.exc_info is None and record.exc_text is None for record in app_records)
    for secret in (
        "hunter2",
        "internal-db",
        "SELECT",
        "secret\\path",
        "Traceback",
        "RuntimeError",
        "ValueError",
        "QueryUnderstandingFailed",
        "something else",
    ):
        assert secret not in response.text
        assert secret not in "\n".join(app_log)


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "   "},
        {"q": "shoes\x00"},
        {"q": "x" * 201},
        {"q": "shoes", "top_k": 0},
        {"q": "shoes", "top_k": 51},
        {"q": "shoes", "top_k": "many"},
    ],
)
def test_validation_422s_happen_before_the_provider(client_for, events, params):
    provider = RaisingProvider()
    body = {"query": params["q"]} | ({"top_k": params["top_k"]} if "top_k" in params else {})
    with client_for(provider=provider) as client:
        got = client.get("/search/hybrid", params=params)
        posted = client.post("/search/hybrid", json=body)
    assert got.status_code == posted.status_code == 422
    assert provider.calls == 0 and "begin" not in events
    assert "search unavailable" not in got.text + posted.text


def test_openapi_hybrid_schema_changes_are_additive(client_for):
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    response = components["HybridSearchResponse"]
    assert set(response["properties"]) == {
        "query",
        "search_version",
        "document_version",
        "tsquery",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dimension",
        "embedding_text_version",
        "distance_metric",
        "fusion",
        "dense_status",
        "lexical_hit_count",
        "dense_hit_count",
        "overlap_count",
        "fused_count",
        "candidate_count",
        "top_k",
        "result_count",
        "results",
        "applied_filters",
        "latency_ms",
        "query_understanding",  # the only addition (Milestone 6)
    }
    assert "query_understanding" in response["required"]
    assert response["properties"]["query_understanding"] == {
        "$ref": "#/components/schemas/QueryUnderstandingBlock"
    }
    block = components["QueryUnderstandingBlock"]
    assert set(block["properties"]) == {"usage", "provider", "provider_version", "understanding"}
    assert set(block["required"]) == set(block["properties"])
    assert block["additionalProperties"] is False
    assert block["properties"]["usage"]["const"] == "informational"
    assert block["properties"]["understanding"] == {
        "$ref": "#/components/schemas/QueryUnderstanding"
    }
    latency = components["HybridSearchLatency"]
    assert set(latency["properties"]) == {
        "lexical_ms",
        "model_load_ms",
        "query_embedding_ms",
        "vector_ms",
        "rrf_ms",
        "total_ms",
        "query_understanding_ms",  # the only addition (Milestone 6)
    }
    understanding_ms = latency["properties"]["query_understanding_ms"]
    assert understanding_ms["type"] == "number" and "anyOf" not in understanding_ms
    assert "query_understanding_ms" in latency["required"]
    assert set(components["HybridFusion"]["properties"]) == {
        "method",
        "rrf_k",
        "rrf_k_status",
        "lexical_k",
        "dense_k",
        "candidate_k",
    }
    assert set(components["HybridSearchResult"]["properties"]) == {
        "rank",
        "rrf_score",
        "lexical_rank",
        "lexical_score",
        "dense_rank",
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
    for method in ("get", "post"):
        responses = schema["paths"]["/search/hybrid"][method]["responses"]
        assert "Query understanding failed" in responses["503"]["description"]
        assert responses["503"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/SearchUnavailable"
        }
    assert "informational" in schema["paths"]["/search/hybrid"]["get"]["description"]
