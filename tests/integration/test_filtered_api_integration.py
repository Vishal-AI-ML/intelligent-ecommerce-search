"""`/search/filtered` (Milestone 7, V2) end to end against real PostgreSQL scratch databases (fake
embedder only; the development database is never touched).

Independent evidence:

* Expected filter specs come from the predeclared Session C truth table
  (`filter_support.TRUTH_TABLE`); expected ignored constraints and the applied-filter JSON are
  written here by hand from the fp-1 rules, not produced by the policy.
* Eligibility is the catalog oracle of `filter_support` (plain SELECTs + Python predicates).
* Source lists are compared with direct calls of the committed filtered lexical / dense reads,
  and fusion with an exact-fraction RRF oracle written here.
"""

import logging
from decimal import Decimal
from fractions import Fraction

import pytest
from dense_support import embed_fake
from fastapi.testclient import TestClient
from filter_support import (
    STALE_EMBEDDING,
    TRUTH_TABLE,
    apply_mutations,
    catalog_facts,
    current_embedding_ids,
    oracle_eligible,
    satisfies,
)
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from ecommerce_search.api import filtered as filtered_api
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_decision_provider, get_embedder
from ecommerce_search.api.schemas import FilteredSearchResponse
from ecommerce_search.decision import DecisionResult
from ecommerce_search.filtering import FilterSpec
from ecommerce_search.query_understanding import QueryUnderstanding
from ecommerce_search.search import filtered as filtered_search
from ecommerce_search.search.filtered import filtered_dense_search, filtered_lexical_search
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

ROWS = {row.id: row for row in TRUTH_TABLE}


def _ignored(*pairs: tuple[str, str]) -> list[dict]:
    return [{"field": field, "reason": reason} for field, reason in pairs]


# truth-table row id -> (label, expected ignored constraints, written from fp-1)
QUERIES = {
    "english": ("cat-brand", []),
    "hinglish-intent": (
        "informational-intent",
        _ignored(("semantic_intent", "informational_only")),
    ),
    "hinglish-sasta": ("informational-price", _ignored(("price_preference", "informational_only"))),
    "hinglish-impossible-price": ("exact-price-none", []),
    "numeric-ram": ("ram-only", []),
    "numeric-storage": ("storage-ssd", []),
    "numeric-nvme": ("nvme", []),
    "numeric-min-price": ("min-price", []),
    "numeric-max-price": ("max-price", []),
    "combined": ("full", []),
    "conflict-storage": (
        "conflict-storage",
        _ignored(("storage_type", "conflict"), ("storage_interface", "conflict")),
    ),
    "conflict-brand": ("conflict-brand", _ignored(("brand", "conflict"))),
    "conflict-price": (
        "conflict-price",
        _ignored(("min_price", "conflict"), ("max_price", "conflict")),
    ),
    "ambiguity": ("ambiguous-ram", _ignored(("ram_gb", "ambiguous_family"))),
    "impossible-brand": ("impossible-brand", []),
    "impossible-phone-ssd": ("impossible-phone-ssd", []),
    "punctuation": ("punctuation", []),
}
QUERY_PARAMS = pytest.mark.parametrize("label", QUERIES, ids=QUERIES)
FIELD_ORDER = (
    "category",
    "brand",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "storage_interface",
    "min_price",
    "max_price",
)
OPERATORS = {"min_price": "gte", "max_price": "lte"}


def _plain(value):
    return value.value if hasattr(value, "value") else value


def expected_applied(spec: FilterSpec) -> list[dict]:
    """The applied-filter JSON for a spec, written from fp-1: fixed field order, `eq` except
    inclusive price bounds, canonical strings (prices with exactly two decimals)."""
    out = []
    for name in FIELD_ORDER:
        value = getattr(spec, name)
        if value is None:
            continue
        rendered = f"{value:.2f}" if isinstance(value, Decimal) else str(_plain(value))
        out.append({"field": name, "operator": OPERATORS.get(name, "eq"), "value": rendered})
    return out


def rrf_oracle(lexical: list[str], dense: list[str], rrf_k: int) -> list[tuple[str, Fraction]]:
    scores: dict[str, Fraction] = {}
    for ranking in (lexical, dense):
        for position, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, Fraction(0)) + Fraction(1, rrf_k + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class RecordingEmbedder(FakeEmbedder):
    """Records the pool's checked-out connections while the model loads and encodes."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = None
        self.events: list = []

    def load(self):
        checked_out = None if self.engine is None else self.engine.pool.checkedout()
        self.events.append(("load", checked_out))
        return super().load()

    def embed_query(self, text):
        self.events.append(("embed", self.engine.pool.checkedout()))
        return super().embed_query(text)


@pytest.fixture(scope="module")
def world(writable_seeded_engine):
    """Module-owned scratch catalog: seed + fake embeddings + the Session C mutations."""
    engine = writable_seeded_engine
    assert engine.url.database.startswith("ecommerce_search_test_")
    embed_fake(engine)
    apply_mutations(engine)
    return {"engine": engine, "facts": catalog_facts(engine)}


def make_client(settings, engine, fake, provider=None) -> TestClient:
    scratch = settings.model_copy(update={"postgres_db": engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    assert scratch.postgres_db.startswith("ecommerce_search_test_")
    app = create_app(scratch)
    app.dependency_overrides[get_embedder] = lambda: fake
    if provider is not None:
        app.dependency_overrides[get_decision_provider] = lambda: provider
    return TestClient(app)


def record_statements(engine) -> list:
    log: list = []
    event.listen(engine, "begin", lambda conn: log.append("BEGIN"))
    event.listen(engine, "commit", lambda conn: log.append("COMMIT"))
    event.listen(engine, "rollback", lambda conn: log.append("ROLLBACK"))
    event.listen(
        engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, params, context, many: log.append(
            " ".join(statement.split())[:40]
        ),
    )
    return log


def spy_reads(monkeypatch) -> list:
    calls: list = []
    original = filtered_api.read_filtered_sources

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(filtered_api, "read_filtered_sources", spy)
    return calls


# ------------------------------------------------------------------------- oracle agreement


@QUERY_PARAMS
def test_filtered_search_matches_the_independent_oracles(settings, world, monkeypatch, label):
    row_id, ignored = QUERIES[label]
    row = ROWS[row_id]
    assert row.parsed  # the expected spec is the predeclared parse -> fp-1 expectation
    fake = RecordingEmbedder()
    with make_client(settings, world["engine"], fake) as client:
        engine = client.app.state.engine
        fake.engine = engine
        app_settings = client.app.state.settings
        reads = spy_reads(monkeypatch)
        log = record_statements(engine)
        got = client.get("/search/filtered", params={"q": row.query, "top_k": 50})
        posted = client.post("/search/filtered", json={"query": row.query, "top_k": 50})
        assert engine.pool.checkedout() == 0
    assert got.status_code == posted.status_code == 200
    a, b = got.json(), posted.json()
    a.pop("latency_ms"), b.pop("latency_ms")
    assert a == b  # GET/POST parity
    body = FilteredSearchResponse.model_validate(got.json())
    assert body.search_version == "v2_filtered" and body.filter_policy.version == "fp-1"
    assert a["applied_filters"] == expected_applied(row.spec)
    assert a["ignored_constraints"] == ignored
    assert a["query"] == row.query and body.query_understanding.usage == "filter_source"
    assert body.latency_ms.query_understanding_ms >= 0
    assert body.latency_ms.filter_translation_ms >= 0
    if row.query == "!!!":
        assert reads == [] and log == [] and fake.load_calls == 0
        assert a["results"] == [] and a["dense_status"] == "skipped_no_searchable_text"
        assert a["lexical_hit_count"] == a["dense_hit_count"] == a["candidate_count"] == 0
        return
    # The route passed the predeclared spec, the unchanged query and the configured depths.
    assert len(reads) == 2  # one read per request (GET, POST), never a retry
    for args, kwargs in reads:
        assert kwargs == {} and args[1] == row.query and args[4] == row.spec
        assert args[5:] == (app_settings.search_lexical_k, app_settings.search_dense_k)
    assert log.count("BEGIN") == 2 and log.count("COMMIT") == 2 and "ROLLBACK" not in log
    assert all(checked_out == 0 for _, checked_out in fake.events if checked_out is not None)
    # Direct, committed filtered reads are the source oracle.
    vector = FakeEmbedder().embed_query(row.query)
    with Session(world["engine"]) as session:
        lexical = filtered_lexical_search(
            session, row.query, row.spec, app_settings.search_lexical_k
        ).hits
        dense = filtered_dense_search(
            session, vector, app_settings.embedding_spec(), row.spec, app_settings.search_dense_k
        ).hits
    assert a["lexical_hit_count"] == len(lexical) and a["dense_hit_count"] == len(dense)
    lexical_ids = [hit.product_id for hit in lexical]
    dense_ids = [hit.product_id for hit in dense]
    assert a["overlap_count"] == len(set(lexical_ids) & set(dense_ids))
    assert a["fused_count"] == len(set(lexical_ids) | set(dense_ids))
    rrf_k = a["fusion"]["rrf_k"]
    assert rrf_k == app_settings.search_rrf_k
    fused = rrf_oracle(lexical_ids, dense_ids, rrf_k)[: app_settings.search_candidate_k]
    assert a["candidate_count"] == len(fused)
    expected = fused[:50]
    assert [(r["product_id"], r["rrf_score"]) for r in a["results"]] == [
        (pid, float(score)) for pid, score in expected
    ]
    assert a["result_count"] == len(a["results"])
    by_lexical = {hit.product_id: hit for hit in lexical}
    by_dense = {hit.product_id: hit for hit in dense}
    for position, result in enumerate(a["results"], start=1):
        pid = result["product_id"]
        assert result["rank"] == position
        assert satisfies(world["facts"][pid], row.spec), pid  # independent catalog predicate
        lex, den = by_lexical.get(pid), by_dense.get(pid)
        assert result["lexical_rank"] == (lex and lex.rank)
        assert result["lexical_score"] == (lex and lex.lexical_score)
        assert result["dense_rank"] == (den and den.rank)
        assert result["dense_score"] == (den and den.dense_score)
        source = lex or den
        assert result["title"] == source.title and result["price"] == f"{source.price:.2f}"
    if row.id == "full":
        # Its only eligible product is the deliberately stale embedding, and the strict-AND
        # lexical query also needs "under"/"40k": an empty list is the correct, unrelaxed result.
        assert oracle_eligible(world["facts"], row.spec) == {STALE_EMBEDDING}
        assert STALE_EMBEDDING not in current_embedding_ids(
            world["engine"], app_settings.embedding_spec()
        )
        assert lexical == [] and dense == [] and a["results"] == []
    elif row.expect == "nonempty":
        assert a["results"] and a["dense_hit_count"] > 0  # non-vacuous comparison
    if row.expect == "empty":
        # Valid filters with no eligible product: 200, no results, filters preserved.
        assert a["results"] == [] and a["candidate_count"] == 0
        assert a["applied_filters"] and a["applied_filters"] == expected_applied(row.spec)


# ------------------------------------------------------------------------- V1 equivalence


@pytest.mark.parametrize(
    "query", ["nvme hdd", "above 30k under 20k", "memory 8gb", "zzqxv", "comfortable gift"]
)
def test_empty_filter_v2_retrieval_equals_v1(settings, world, query):
    with make_client(settings, world["engine"], FakeEmbedder()) as client:
        v1 = client.get("/search/hybrid", params={"q": query, "top_k": 50}).json()
        v2 = client.get("/search/filtered", params={"q": query, "top_k": 50}).json()
    assert v2["applied_filters"] == []
    assert v2["dense_hit_count"] > 0  # non-vacuous: the comparison covers real candidates
    for payload in (v1, v2):
        for key in (
            "search_version",
            "filter_policy",
            "applied_filters",
            "ignored_constraints",
            "query_understanding",
            "latency_ms",
        ):
            payload.pop(key, None)
    assert v1 == v2


# ------------------------------------------------------------------------- failures


class UnevidencedProvider:
    """A well-formed decision whose brand has no matched-term evidence: fp-1 refuses it."""

    def __init__(self) -> None:
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        return DecisionResult(understanding=QueryUnderstanding(raw_query=query, brand="HP"))


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def api_log():
    # Alembic's fileConfig replaces the root handlers, so the route logger gets its own.
    records = _Records()
    route_logger = logging.getLogger(filtered_api.__name__)
    route_logger.addHandler(records)
    yield records
    route_logger.removeHandler(records)


@pytest.mark.parametrize("failure", ["provider", "derive"])
def test_policy_failure_checks_out_no_connection(settings, world, monkeypatch, api_log, failure):
    fake = RecordingEmbedder()
    provider = UnevidencedProvider() if failure == "provider" else None
    if failure == "derive":

        def broken(decision):
            raise RuntimeError("policy failure with secret detail")

        monkeypatch.setattr(filtered_api, "derive_filters", broken)
    with make_client(settings, world["engine"], fake, provider) as client:
        engine = client.app.state.engine
        fake.engine = engine
        checkouts: list = []
        event.listen(engine.pool, "checkout", lambda *args: checkouts.append(True))
        got = client.get("/search/filtered", params={"q": "hp laptop"})
        posted = client.post("/search/filtered", json={"query": "hp laptop"})
        assert engine.pool.checkedout() == 0
    for response in (got, posted):
        assert response.status_code == 503
        assert response.json() == {"detail": "search unavailable"}
    assert checkouts == []  # no engine checkout at all
    assert fake.events == [] and fake.load_calls == 0
    assert api_log.messages == ["filtered search unavailable: filter_policy_failed"] * 2
    assert not any("secret" in message or "hp laptop" in message for message in api_log.messages)


@pytest.mark.parametrize("failing", ["lexical", "dense"])
def test_database_failure_rolls_back_releases_and_never_retries(
    settings, world, monkeypatch, api_log, failing
):
    def fail(session):
        session.execute(text("SELECT * FROM no_such_table"))

    real_lexical, real_dense = filtered_lexical_search, filtered_dense_search

    def lexical_read(session, query, filters, limit):
        result = real_lexical(session, query, filters, limit)
        if failing == "lexical":
            fail(session)
        return result

    def dense_read(session, query_vector, spec, filters, limit):
        if failing == "dense":
            fail(session)
        return real_dense(session, query_vector, spec, filters, limit)

    monkeypatch.setattr(filtered_search, "filtered_lexical_search", lexical_read)
    monkeypatch.setattr(filtered_search, "filtered_dense_search", dense_read)
    with make_client(settings, world["engine"], FakeEmbedder()) as client:
        engine = client.app.state.engine
        assert client.get("/health").status_code == 200  # warm-up: dialect initialization
        reads = spy_reads(monkeypatch)
        log = record_statements(engine)
        pool_events: list = []
        event.listen(engine.pool, "checkout", lambda *args: pool_events.append("out"))
        event.listen(engine.pool, "checkin", lambda *args: pool_events.append("in"))
        response = client.get("/search/filtered", params={"q": "hp laptop"})
        assert engine.pool.checkedout() == 0
    assert response.status_code == 503 and response.json() == {"detail": "search unavailable"}
    assert "no_such_table" not in response.text
    assert len(reads) == 1  # no unfiltered or second attempt
    assert log[0] == "BEGIN" and log[-1] == "ROLLBACK"
    assert "COMMIT" not in log and log.count("BEGIN") == 1
    assert pool_events == ["out", "in"]  # one checkout, released
    assert len(api_log.messages) == 1
    assert api_log.messages[0].startswith("filtered search unavailable: table_missing")
    assert "SELECT" not in api_log.messages[0] and "hp laptop" not in api_log.messages[0]


@pytest.mark.parametrize("query", ["nike laptop", "phone ssd", "from 20k ke andar"])
def test_zero_matches_run_exactly_one_filtered_transaction(settings, world, monkeypatch, query):
    with make_client(settings, world["engine"], FakeEmbedder()) as client:
        engine = client.app.state.engine
        assert client.get("/health").status_code == 200  # warm-up: dialect initialization
        reads = spy_reads(monkeypatch)
        log = record_statements(engine)
        response = client.get("/search/filtered", params={"q": query})
    body = response.json()
    assert response.status_code == 200 and body["results"] == []
    assert body["applied_filters"] and body["lexical_hit_count"] == body["dense_hit_count"] == 0
    assert len(reads) == 1
    # One transaction: snapshot, tsquery, filtered lexical, filtered dense; nothing after.
    assert log[0] == "BEGIN" and log[-1] == "COMMIT" and len(log) == 6
    assert log[1].startswith("SET TRANSACTION ISOLATION LEVEL")
    assert "fallback" not in response.text.lower()
