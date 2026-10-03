"""`GET/POST /search/filtered` (Milestone 7, V2) with a fake embedder, an instrumented fake
session and fake filtered source reads (no DB, no model).

The fake source reads emulate the SQL over a small in-test catalog: they return only products
that satisfy the `FilterSpec` they receive (an independent Python predicate written from the fp-1
rules). The API is never allowed to see the catalog: any narrowing must come from the spec it
passes to `read_filtered_sources`.

Expected applied / ignored filters below are written from the fp-1 rules (`filtering/policy.py`
docstring); they were cross-checked against the committed policy before this file was written.
"""

import hashlib
import json
import logging
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, ProgrammingError

from ecommerce_search import filtering as filtering_package
from ecommerce_search.api import dense as dense_api
from ecommerce_search.api import filtered as filtered_api
from ecommerce_search.api import search as search_api
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dense import router as dense_router
from ecommerce_search.api.dependencies import get_db_session, get_decision_provider, get_embedder
from ecommerce_search.api.health import router as health_router
from ecommerce_search.api.hybrid import router as hybrid_router
from ecommerce_search.api.schemas import FilteredSearchResponse
from ecommerce_search.api.search import router as search_router
from ecommerce_search.catalog.taxonomy import Category, StorageInterface, StorageType
from ecommerce_search.decision import DecisionResult, DeterministicDecisionProvider
from ecommerce_search.decision import provider as decision_module
from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.filtering import (
    AppliedFilter,
    FilterDerivation,
    FilterSpec,
    IgnoredConstraint,
)
from ecommerce_search.filtering import policy as policy_module
from ecommerce_search.query_understanding import (
    Ambiguity,
    AmbiguityReason,
    Conflict,
    Intent,
    MatchedTerm,
    MatchRule,
    QueryAttributes,
    QueryUnderstanding,
    SourceSpan,
    UnderstandingField,
    parse_normalized_query,
)
from ecommerce_search.search import dense as dense_module
from ecommerce_search.search import filtered as filtered_search
from ecommerce_search.search import hybrid as hybrid_search
from ecommerce_search.search import lexical as lexical_module
from ecommerce_search.search.dense import DenseHit, DenseResult
from ecommerce_search.search.documents import FTS_CONFIG
from ecommerce_search.search.lexical import LexicalHit, LexicalResult
from fake_embedder import FakeEmbedder

SENSITIVE = "host=internal-db.example password=hunter2 SELECT * FROM products D:\\secret\\path"
FIXED_503 = {"detail": "search unavailable"}
SNAPSHOT = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
SECRETS = ("hunter2", "internal-db", "SELECT", "secret\\path", "Traceback")

# ---- in-test catalog and the fake filtered "SQL" -----------------------------------------------


def _product(pid, category, brand, price, **spec):
    return {"id": pid, "category": category, "brand": brand, "price": Decimal(price), "spec": spec}


CATALOG = (
    _product(
        "L1",
        "laptop",
        "HP",
        "33999.00",
        ram_gb=8,
        storage_gb=256,
        storage_type="SSD",
        storage_interface="SATA",
    ),
    _product(
        "L2",
        "laptop",
        "Dell",
        "55999.00",
        ram_gb=16,
        storage_gb=512,
        storage_type="SSD",
        storage_interface="NVME",
    ),
    _product("L3", "laptop", "HP", "28999.00", ram_gb=8, storage_gb=1024, storage_type="HDD"),
    _product(
        "L4",
        "laptop",
        "HP",
        "38999.00",
        ram_gb=8,
        storage_gb=256,
        storage_type="SSD",
        storage_interface="NVME",
    ),
    _product("P1", "phone", "Samsung", "26999.00", ram_gb=8, storage_gb=256),
    _product("P2", "phone", "Apple", "79999.00", ram_gb=6, storage_gb=128),
    _product("S1", "shoes", "Nike", "4299.00"),
    _product("H1", "headphones", "Sony", "1999.00"),
)
NOT_LEXICAL = {"H1"}  # the fake lexical source never matches it, so overlap < union


def _plain(value):
    return value.value if hasattr(value, "value") else value


def satisfies(product: dict, filters: FilterSpec) -> bool:
    """fp-1 eligibility over the in-test catalog (exact equality, inclusive prices; spec fields
    only on a spec row, missing values never match). Independent of the production SQL."""
    spec = product["spec"]
    checks = {
        "category": lambda v: product["category"] == v,
        "brand": lambda v: product["brand"] == v,
        "ram_gb": lambda v: spec.get("ram_gb") == v,
        "storage_gb": lambda v: spec.get("storage_gb") == v,
        "storage_type": lambda v: spec.get("storage_type") == v,
        "storage_interface": lambda v: spec.get("storage_interface") == v,
        "min_price": lambda v: product["price"] >= v,
        "max_price": lambda v: product["price"] <= v,
    }
    return all(
        check(_plain(getattr(filters, name)))
        for name, check in checks.items()
        if getattr(filters, name) is not None
    )


def _display(product: dict) -> dict:
    return {
        "product_id": product["id"],
        "title": f"{product['brand']} {product['category']} {product['id']}",
        "brand": product["brand"],
        "category": product["category"],
        "subcategory": None,
        "description": None,
        "price": product["price"],
        "currency": "INR",
        "rating": Decimal("4.2"),
        "review_count": 3,
        "availability": "in_stock",
    }


def lexical_hits_for(filters: FilterSpec, limit: int) -> list[LexicalHit]:
    eligible = [p for p in CATALOG if satisfies(p, filters) and p["id"] not in NOT_LEXICAL]
    return [
        LexicalHit(rank=r, lexical_score=1.0 / r, **_display(p))
        for r, p in enumerate(eligible[:limit], start=1)
    ]


def dense_hits_for(filters: FilterSpec, limit: int) -> list[DenseHit]:
    eligible = [p for p in reversed(CATALOG) if satisfies(p, filters)]
    return [
        DenseHit(rank=r, dense_score=1.0 - r / 100, **_display(p))
        for r, p in enumerate(eligible[:limit], start=1)
    ]


def eligible_ids(filters: FilterSpec) -> set[str]:
    return {p["id"] for p in CATALOG if satisfies(p, filters)}


def rrf_oracle(lexical: list[str], dense: list[str], rrf_k: int) -> list[tuple[str, Fraction]]:
    scores: dict[str, Fraction] = {}
    for ranking in (lexical, dense):
        for position, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, Fraction(0)) + Fraction(1, rrf_k + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


# ---- hand-written fp-1 expectations ------------------------------------------------------------

LAPTOP, PHONE, SHOES, HEADPHONES = (
    Category.LAPTOP,
    Category.PHONE,
    Category.SHOES,
    Category.HEADPHONES,
)
SSD, HDD, NVME = StorageType.SSD, StorageType.HDD, StorageInterface.NVME


def A(field, value, operator="eq"):  # noqa: N802 - table shorthand
    return {"field": field, "operator": operator, "value": value}


def I(field, reason):  # noqa: N802, E743 - table shorthand
    return {"field": field, "reason": reason}


# query -> (expected FilterSpec, applied_filters JSON, ignored_constraints JSON)
CASES = {
    "no-filters": ("comfortable gift", FilterSpec(), [], []),
    "category": (
        "wireless headphones",
        FilterSpec(category=HEADPHONES),
        [A("category", "headphones")],
        [],
    ),
    "brand": ("nike", FilterSpec(brand="Nike"), [A("brand", "Nike")], []),
    "ram": ("8gb ram", FilterSpec(ram_gb=8), [A("ram_gb", "8")], []),
    "storage": ("256gb storage", FilterSpec(storage_gb=256), [A("storage_gb", "256")], []),
    "storage-type": ("ssd", FilterSpec(storage_type=SSD), [A("storage_type", "SSD")], []),
    "storage-interface": (
        "nvme",
        FilterSpec(storage_type=SSD, storage_interface=NVME),
        [A("storage_type", "SSD"), A("storage_interface", "NVME")],
        [],
    ),
    "min-price": (
        "above 30k",
        FilterSpec(min_price=Decimal("30000.00")),
        [A("min_price", "30000.00", "gte")],
        [],
    ),
    "max-price": (
        "under 40k",
        FilterSpec(max_price=Decimal("40000.00")),
        [A("max_price", "40000.00", "lte")],
        [],
    ),
    "combined": (
        "hp laptop 8gb ram 256gb ssd under 40k",
        FilterSpec(
            category=LAPTOP,
            brand="HP",
            ram_gb=8,
            storage_gb=256,
            storage_type=SSD,
            max_price=Decimal("40000.00"),
        ),
        [
            A("category", "laptop"),
            A("brand", "HP"),
            A("ram_gb", "8"),
            A("storage_gb", "256"),
            A("storage_type", "SSD"),
            A("max_price", "40000.00", "lte"),
        ],
        [],
    ),
    "conflict-storage": (
        "nvme hdd",
        FilterSpec(),
        [],
        [I("storage_type", "conflict"), I("storage_interface", "conflict")],
    ),
    "conflict-brand": (
        "hp dell laptop",
        FilterSpec(category=LAPTOP),
        [A("category", "laptop")],
        [I("brand", "conflict")],
    ),
    "conflict-price": (
        "above 30k under 20k",
        FilterSpec(),
        [],
        [I("min_price", "conflict"), I("max_price", "conflict")],
    ),
    "ambiguity-ram": (
        "laptop 8gb 16gb ram",
        FilterSpec(category=LAPTOP),
        [A("category", "laptop")],
        [I("ram_gb", "ambiguous_family")],
    ),
    "ambiguity-range": (
        "8gb ram 512gb ssd laptop 30k-40k",
        FilterSpec(category=LAPTOP, storage_type=SSD),
        [A("category", "laptop"), A("storage_type", "SSD")],
        [I("ram_gb", "ambiguous_family"), I("storage_gb", "ambiguous_family")],
    ),
    "informational-intent": (
        "coding ke liye laptop",
        FilterSpec(category=LAPTOP),
        [A("category", "laptop")],
        [I("semantic_intent", "informational_only")],
    ),
    "informational-sasta": (
        "sasta phone",
        FilterSpec(category=PHONE),
        [A("category", "phone")],
        [I("price_preference", "informational_only")],
    ),
    "impossible": (
        "nike laptop",
        FilterSpec(category=LAPTOP, brand="Nike"),
        [A("category", "laptop"), A("brand", "Nike")],
        [],
    ),
    "punctuation": ("!!!", FilterSpec(), [], []),
}
CASE_PARAMS = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())

# ---- fakes ---------------------------------------------------------------------------------------


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
        raise AssertionError("the filtered route never asks the session for a connection")

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _Result:
    def __init__(self, one=None, rows=()):
        self._one, self._rows = one, list(rows)

    def one(self):
        return self._one

    def all(self):
        return self._rows


class SqlRecordingSession(FakeSession):
    """Runs the real filtered source functions: records every statement and its parameters."""

    def __init__(self, events: list) -> None:
        super().__init__(events)
        self.statements: list[tuple[object, dict]] = []

    def execute(self, statement, params=None):
        assert self._open, "statement outside the transaction"
        self.statements.append((statement, dict(params or {})))
        if statement is lexical_module.TSQUERY_SQL:
            return _Result(one=("'x'", 1))
        return _Result(rows=())


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
        lexical_error=None,
        dense_error=None,
        execute_error=None,
        provider=None,
        session_class=FakeSession,
        real_sources=False,
        **settings,
    ):
        sessions: list = []
        fake = RecordingEmbedder(events, sessions) if embedder == "recording" else embedder

        def fake_lexical(session, query, filters, limit):
            assert session.in_transaction() and type(filters) is FilterSpec
            events.append(("lexical", query, filters, limit))
            if lexical_error is not None:
                raise lexical_error
            return LexicalResult(
                tsquery="'x'", hits=lexical_hits_for(filters, limit), lexical_ms=1.5
            )

        def fake_dense(session, vector, spec, filters, limit):
            assert session.in_transaction() and type(filters) is FilterSpec
            events.append(("dense", len(vector), filters, limit))
            if dense_error is not None:
                raise dense_error
            return DenseResult(hits=dense_hits_for(filters, limit), vector_ms=2.5)

        if not real_sources:
            monkeypatch.setattr(filtered_search, "filtered_lexical_search", fake_lexical)
            monkeypatch.setattr(filtered_search, "filtered_dense_search", fake_dense)

        def session_dependency():
            session = (
                session_class(events, execute_error)
                if session_class is FakeSession
                else session_class(events)
            )
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


def names(events: list) -> list[str]:
    return [e if isinstance(e, str) else e[0] for e in events]


def without_latency(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("latency_ms")
    return payload


def call(client, method: str, query: str, top_k: int | None = None):
    if method == "get":
        params = {"q": query} | ({} if top_k is None else {"top_k": top_k})
        return client.get("/search/filtered", params=params)
    body = {"query": query} | ({} if top_k is None else {"top_k": top_k})
    return client.post("/search/filtered", json=body)


SPIED = (
    "normalize_query",
    "understand_normalized_query",
    "derive_filters",
    "encode_query",
    "read_filtered_sources",
    "fuse_rrf",
)
RETRIEVAL_SIDE = ("encode_query", "read_filtered_sources", "fuse_rrf")


def install_spies(monkeypatch, events: list, targets=SPIED) -> list:
    """Wrap entry points used by the V2 route; record `call:<name>` in `events` on entry and
    (name, args, kwargs, result) in the returned list on success."""
    records: list = []
    for name in targets:
        original = getattr(filtered_api, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            events.append(f"call:{_name}")
            result = _original(*args, **kwargs)
            records.append((_name, args, kwargs, result))
            return result

        monkeypatch.setattr(filtered_api, name, spy)
    return records


def by_name(records: list, name: str) -> list:
    return [record for record in records if record[0] == name]


# ---- 1. GET/POST parity ------------------------------------------------------------------------


@CASE_PARAMS
def test_get_and_post_are_equivalent_and_report_the_expected_filters(client_for, case):
    query, _spec, applied, ignored = case
    with client_for() as client:
        got = call(client, "get", query, 5)
        posted = call(client, "post", query, 5)
    assert got.status_code == posted.status_code == 200
    assert without_latency(got.json()) == without_latency(posted.json())
    body = got.json()
    assert body["applied_filters"] == applied
    assert body["ignored_constraints"] == ignored
    assert body["filter_policy"] == {"version": "fp-1"}
    FilteredSearchResponse.model_validate(body)


# ---- 2. call order and cardinality ---------------------------------------------------------------


def test_exact_call_order_and_cardinality(client_for, events, monkeypatch):
    with client_for() as client:
        install_spies(monkeypatch, events)
        assert call(client, "get", "hp laptop 8gb ram").status_code == 200
    assert names(events) == [
        "call:normalize_query",
        "call:understand_normalized_query",
        "call:derive_filters",
        "call:encode_query",
        "load",
        "load",  # count_tokens
        "embed",
        "load",  # inside embed_query
        "call:read_filtered_sources",
        "begin",
        "execute",
        "lexical",
        "dense",
        "commit",
        "call:fuse_rrf",
        "close",
    ]
    assert all(e[1] is False for e in events if e[0] in ("load", "embed"))
    assert ("execute", SNAPSHOT) in events


@pytest.mark.parametrize("method", ["get", "post"])
def test_each_stage_runs_exactly_once_per_request(client_for, events, monkeypatch, method):
    stub = DeterministicDecisionProvider()
    calls: list = []
    original = stub.understand
    stub.understand = lambda q, d: calls.append(q) or original(q, d)
    with client_for(provider=stub) as client:
        records = install_spies(monkeypatch, events)
        assert call(client, method, "  sasta   phone 50k ke andar ").status_code == 200
    counts = {name: len(by_name(records, name)) for name in SPIED}
    assert counts == dict.fromkeys(SPIED, 1)
    assert calls == ["sasta phone 50k ke andar"]
    assert by_name(records, "normalize_query")[0][1][0] == "  sasta   phone 50k ke andar "


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("query", ["!!!", "-", "...  ???", "\u2014"])
def test_punctuation_skips_model_database_and_fusion(
    client_for, events, monkeypatch, method, query
):
    with client_for() as client:
        records = install_spies(monkeypatch, events)
        response = call(client, method, query)
    assert response.status_code == 200
    assert names(events) == [
        "call:normalize_query",
        "call:understand_normalized_query",
        "call:derive_filters",
        "close",
    ]
    assert {name for name, *_ in records} == {
        "normalize_query",
        "understand_normalized_query",
        "derive_filters",
    }
    assert client.fake.load_calls == 0
    body = response.json()
    assert body["results"] == [] and body["result_count"] == 0 and body["tsquery"] is None
    assert body["dense_status"] == "skipped_no_searchable_text"
    counts = ("lexical_hit_count", "dense_hit_count", "overlap_count", "fused_count")
    assert [body[key] for key in counts] == [0, 0, 0, 0] and body["candidate_count"] == 0
    assert body["applied_filters"] == [] and body["ignored_constraints"] == []
    block = body["query_understanding"]
    assert (
        block["usage"] == "filter_source" and block["understanding"]["raw_query"] == body["query"]
    )
    latency = body["latency_ms"]
    executed = ("total_ms", "query_understanding_ms", "filter_translation_ms")
    assert all(value is None for key, value in latency.items() if key not in executed)
    assert latency["total_ms"] >= latency["query_understanding_ms"] >= 0
    assert latency["filter_translation_ms"] >= 0


# ---- 3. retrieval input isolation --------------------------------------------------------------

FORBIDDEN_TYPES = (
    QueryUnderstanding,
    DecisionResult,
    MatchedTerm,
    SourceSpan,
    Ambiguity,
    Conflict,
    QueryAttributes,
    Intent,
    FilterDerivation,
    AppliedFilter,
    IgnoredConstraint,
)


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _walk(item)


@CASE_PARAMS
def test_retrieval_receives_only_the_query_vector_and_filter_spec(
    client_for, events, monkeypatch, make_settings, case
):
    raw, expected_spec, _applied, _ignored = case
    settings = make_settings()
    with client_for() as client:
        records = install_spies(monkeypatch, events)
        assert call(client, "get", f"  {raw}  ").status_code == 200
    query = " ".join(raw.split())
    (derive,) = by_name(records, "derive_filters")
    derivation = derive[3]
    assert derivation.spec == expected_spec
    if query == "!!!":
        assert not any(by_name(records, name) for name in RETRIEVAL_SIDE)
        return
    (encode,) = by_name(records, "encode_query")
    (read,) = by_name(records, "read_filtered_sources")
    (fuse,) = by_name(records, "fuse_rrf")
    # The encoder sees exactly the unchanged normalized query.
    assert encode[1][2] == query and encode[2] == {}
    assert client.fake.encoded == [query]
    # read_filtered_sources: (session, query, vector, embedding spec, FilterSpec, lexical_k,
    # dense_k) and nothing else; the FilterSpec is the policy's own object.
    args = read[1]
    assert read[2] == {} and len(args) == 7
    assert args[0] is client.sessions[0] and args[1] == query
    assert args[2] == FakeEmbedder().embed_query(query)
    assert args[3] == settings.embedding_spec()
    assert args[4] is derivation.spec and type(args[4]) is FilterSpec
    assert args[5:] == (settings.search_lexical_k, settings.search_dense_k)
    for name, record_args, record_kwargs, _result in (encode, read, fuse):
        for argument in list(record_args) + list(record_kwargs.values()):
            for nested in _walk(argument):
                assert not isinstance(nested, FORBIDDEN_TYPES), (name, type(nested))
    # Fusion receives exactly the two returned source lists and the configured depths.
    lexical_result, dense_result = read[3]
    assert fuse[1][0] is lexical_result.hits and fuse[1][1] is dense_result.hits
    assert fuse[1][2:] == (settings.search_rrf_k, settings.search_candidate_k) and fuse[2] == {}
    # Both fake sources received the same spec object.
    source_calls = [e for e in events if isinstance(e, tuple) and e[0] in ("lexical", "dense")]
    assert [e[2] for e in source_calls] == [derivation.spec, derivation.spec]
    assert all(e[2] is derivation.spec for e in source_calls)


def _filter_params(spec: FilterSpec) -> dict:
    """Bind values expected for a spec, written out from the fp-1 field list."""
    values = {
        "f_category": _plain(spec.category),
        "f_brand": spec.brand,
        "f_ram_gb": spec.ram_gb,
        "f_storage_gb": spec.storage_gb,
        "f_storage_type": _plain(spec.storage_type),
        "f_storage_interface": _plain(spec.storage_interface),
        "f_min_price": spec.min_price,
        "f_max_price": spec.max_price,
    }
    return {key: value for key, value in values.items() if value is not None}


@CASE_PARAMS
def test_sql_parameters_carry_only_the_query_vector_and_filter_spec(
    client_for, events, make_settings, case
):
    raw, expected_spec, _applied, _ignored = case
    settings = make_settings()
    with client_for(session_class=SqlRecordingSession, real_sources=True) as client:
        response = call(client, "post", raw)
    assert response.status_code == 200
    query = " ".join(raw.split())
    if query == "!!!":
        assert client.sessions[0].statements == []
        return
    statements = client.sessions[0].statements
    assert [str(s) for s, _ in statements][0] == SNAPSHOT
    vector = FakeEmbedder().embed_query(query)
    fields = filtered_search.active_fields(expected_spec)
    expected = [
        (hybrid_search.SNAPSHOT_SQL, {}),
        (lexical_module.TSQUERY_SQL, {"cfg": FTS_CONFIG, "q": query}),
        (
            filtered_search.lexical_statement(fields),
            lexical_module.retrieval_params(query, settings.search_lexical_k)
            | _filter_params(expected_spec),
        ),
        (
            filtered_search.dense_statement(fields),
            dense_module.retrieval_params(
                vector, settings.embedding_spec(), settings.search_dense_k
            )
            | _filter_params(expected_spec),
        ),
    ]
    assert [(str(s), p) for s, p in statements] == [(str(s), p) for s, p in expected]
    for _statement, params in statements:
        for key, value in params.items():
            assert type(value) is not bool or key == "normalized"
            if key.startswith("f_") and key.endswith("_price"):
                assert type(value) is Decimal
            if key in ("f_ram_gb", "f_storage_gb"):
                assert type(value) is int
    body = response.json()
    assert body["results"] == [] and body["applied_filters"] == case[2]


def _augmented(understanding: QueryUnderstanding) -> QueryUnderstanding:
    """The same parse plus non-filter content: an intent, a price preference, an extra
    unresolved span and a numeric ambiguity that suppresses no field present in the parse."""
    span = SourceSpan(start=0, end=2, text=understanding.raw_query[:2])
    return understanding.model_copy(
        update={
            "semantic_intent": Intent.GAMING,
            "attributes": understanding.attributes.model_copy(update={"price_preference": "low"}),
            "unresolved": (*understanding.unresolved, span),
            "ambiguities": (
                *understanding.ambiguities,
                Ambiguity(reason=AmbiguityReason.BARE_CAPACITY, span=span, value="hp"),
            ),
            "matched_terms": (
                *understanding.matched_terms,
                MatchedTerm(
                    rule=MatchRule.INTENT_TERM,
                    field=UnderstandingField.SEMANTIC_INTENT,
                    value="gaming",
                    span=span,
                ),
            ),
        }
    )


class AugmentingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        return DecisionResult(understanding=_augmented(deterministic_result))


def test_non_filter_understanding_never_reaches_retrieval(client_for, events, monkeypatch):
    views, bodies = {}, {}
    for name, provider in (
        ("plain", DeterministicDecisionProvider()),
        ("aug", AugmentingProvider()),
    ):
        events.clear()
        with client_for(provider=provider) as client:
            records = install_spies(monkeypatch, events, RETRIEVAL_SIDE)
            bodies[name] = call(client, "get", "hp laptop").json()
        monkeypatch.undo()
        (read,) = by_name(records, "read_filtered_sources")
        (fuse,) = by_name(records, "fuse_rrf")
        views[name] = (read[1][1:], read[3][0].hits, read[3][1].hits, fuse[1], fuse[3])
    assert views["plain"] == views["aug"]  # identical retrieval inputs, source lists and fusion
    plain, aug = bodies["plain"], bodies["aug"]
    assert (
        plain["applied_filters"]
        == aug["applied_filters"]
        == [
            A("category", "laptop"),
            A("brand", "HP"),
        ]
    )
    assert plain["ignored_constraints"] == []
    assert aug["ignored_constraints"] == [
        I("semantic_intent", "informational_only"),
        I("price_preference", "informational_only"),
    ]
    for payload in (plain, aug):
        for key in ("query_understanding", "ignored_constraints", "latency_ms"):
            payload.pop(key)
    assert plain == aug


# ---- 4. failures ---------------------------------------------------------------------------------


class RaisingProvider:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error or RuntimeError(SENSITIVE)
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        raise self.error


class InvalidProvider:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        return self.result


def _parser_raises(monkeypatch):
    def broken(query):
        raise ValueError(SENSITIVE)

    monkeypatch.setattr(decision_module, "parse_normalized_query", broken)


def _unevidenced_decision(query):
    # A well-formed DecisionResult whose brand has no matched-term evidence: fp-1 refuses it.
    return DecisionResult(understanding=QueryUnderstanding(raw_query=query, brand="HP"))


def _derive_raises(monkeypatch):
    def broken(decision):
        raise RuntimeError(SENSITIVE)

    monkeypatch.setattr(filtered_api, "derive_filters", broken)


def _derive_returns(value):
    def install(monkeypatch):
        monkeypatch.setattr(filtered_api, "derive_filters", lambda decision: value)

    return install


POLICY_QUERY = "hunter2 hp laptop"
UNDERSTANDING_FAILURES = {
    "raising provider": (lambda mp: RaisingProvider(), None),
    "parser raises": (lambda mp: None, _parser_raises),
    "non-DecisionResult": (lambda mp: InvalidProvider({"provider": "deterministic"}), None),
    "result for another query": (
        lambda mp: InvalidProvider(
            DecisionResult(understanding=QueryUnderstanding(raw_query="something else"))
        ),
        None,
    ),
}
POLICY_FAILURES = {
    "policy rejects the decision": (
        lambda mp: InvalidProvider(_unevidenced_decision(POLICY_QUERY)),
        None,
    ),
    "derive_filters raises": (lambda mp: None, _derive_raises),
    "derive_filters returns a dict": (lambda mp: None, _derive_returns({"spec": {}})),
    "derive_filters returns None": (lambda mp: None, _derive_returns(None)),
    "derive_filters returns a FilterSpec": (lambda mp: None, _derive_returns(FilterSpec())),
}


def _app_records(caplog):
    return [record for record in caplog.records if record.name.startswith("ecommerce_")]


def _assert_sanitized(response, caplog, expected_error: str):
    assert response.status_code == 503 and response.json() == FIXED_503
    app_records = _app_records(caplog)
    errors = [r for r in app_records if r.levelno >= logging.WARNING]
    assert [r.getMessage() for r in errors] == [expected_error]
    assert [r.name for r in errors] == [filtered_api.__name__]
    assert all(r.exc_info is None and r.exc_text is None for r in app_records)
    app_log = "\n".join(r.getMessage() for r in app_records)
    for secret in (*SECRETS, "RuntimeError", "ValueError", "something else", "hp laptop"):
        assert secret not in response.text
        assert secret not in app_log


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize(
    ("failure", "reason"),
    [(f, "query_understanding_failed") for f in UNDERSTANDING_FAILURES.values()]
    + [(f, "filter_policy_failed") for f in POLICY_FAILURES.values()],
    ids=[*UNDERSTANDING_FAILURES, *POLICY_FAILURES],
)
def test_understanding_and_policy_failures_are_a_fixed_503_before_model_or_database_work(
    client_for, events, caplog, monkeypatch, method, failure, reason
):
    make_provider, install = failure
    provider = make_provider(monkeypatch)
    if install is not None:
        install(monkeypatch)
    with caplog.at_level(logging.DEBUG), client_for(provider=provider) as client:
        records = install_spies(monkeypatch, events, RETRIEVAL_SIDE)
        response = call(client, method, POLICY_QUERY)
    _assert_sanitized(response, caplog, f"filtered search unavailable: {reason}")
    assert records == [] and client.fake.load_calls == 0
    assert events == ["close"]  # no begin, statement, source read, model work or fusion
    assert not client.sessions[0].in_transaction() and client.sessions[0].closed
    if provider is not None:
        assert provider.calls == 1


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize(
    "reason",
    ["snapshot_missing", "load_failed", "dimension_mismatch", "config_mismatch", "encode_failed"],
)
def test_model_failures_are_a_fixed_503_after_the_policy_and_before_any_transaction(
    client_for, events, caplog, monkeypatch, method, reason
):
    fake = FakeEmbedder()
    fake.fail_after_batches = 0
    fake.failure = EmbedderUnavailable(reason)
    with caplog.at_level(logging.ERROR), client_for(embedder=fake) as client:
        install_spies(monkeypatch, events, ("derive_filters", "read_filtered_sources"))
        response = call(client, method, "hp laptop")
    assert response.status_code == 503 and response.json() == FIXED_503
    errors = [r.getMessage() for r in _app_records(caplog) if r.levelno >= logging.WARNING]
    assert len(errors) == 1 and errors[0].startswith(f"filtered search unavailable: {reason}")
    assert events == ["call:derive_filters", "close"]


@pytest.mark.parametrize("method", ["get", "post"])
def test_no_models_directory_is_a_fixed_503(client_for, events, caplog, method):
    with caplog.at_level(logging.ERROR), client_for(embedder=None) as client:
        response = call(client, method, "hp laptop")
    assert response.status_code == 503 and response.json() == FIXED_503
    assert "snapshot_missing" in caplog.text and "model-fetch" in caplog.text
    assert events == ["close"]


def _operational():
    return OperationalError(SENSITIVE, {"p": "hunter2"}, Exception(SENSITIVE))


def _missing_table():
    return ProgrammingError(SENSITIVE, {}, type("E", (Exception,), {"sqlstate": "42P01"})())


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize(
    ("failure", "logged", "tail"),
    [
        ({"execute_error": _operational()}, "database_error", []),
        ({"lexical_error": _operational()}, "database_error", ["lexical"]),
        ({"lexical_error": _missing_table()}, "table_missing", ["lexical"]),
        ({"dense_error": _operational()}, "database_error", ["lexical", "dense"]),
        ({"dense_error": _missing_table()}, "table_missing", ["lexical", "dense"]),
    ],
)
def test_database_failures_roll_back_release_and_never_retry_unfiltered(
    client_for, events, caplog, monkeypatch, method, failure, logged, tail
):
    with caplog.at_level(logging.DEBUG), client_for(**failure) as client:
        records = install_spies(monkeypatch, events, RETRIEVAL_SIDE)
        response = call(client, method, "hunter2 hp laptop")
    assert response.status_code == 503 and response.json() == FIXED_503
    app_log = "\n".join(r.getMessage() for r in _app_records(caplog))
    for secret in SECRETS:
        assert secret not in response.text and secret not in app_log
    errors = [r.getMessage() for r in _app_records(caplog) if r.levelno >= logging.WARNING]
    assert len(errors) == 1 and errors[0].startswith(f"filtered search unavailable: {logged}")
    assert names(events) == [
        "call:encode_query",
        "load",
        "load",
        "embed",
        "load",
        "call:read_filtered_sources",
        "begin",
        "execute",
        *tail,
        "rollback",
        "close",
    ]
    assert [name for name, *_ in records] == ["encode_query"]  # one read attempt, no fusion
    assert not client.sessions[0].in_transaction() and client.sessions[0].closed


class Boom(BaseException):
    pass


@pytest.mark.parametrize("stage", ["provider", "derive", "encode", "read", "fuse"])
def test_base_exceptions_are_not_swallowed(make_settings, events, monkeypatch, stage):
    session = FakeSession(events)
    provider = DeterministicDecisionProvider()
    embedder = FakeEmbedder()

    def boom(*args, **kwargs):
        raise Boom

    if stage == "provider":
        provider = RaisingProvider(Boom())
    elif stage == "derive":
        monkeypatch.setattr(filtered_api, "derive_filters", boom)
    elif stage == "encode":
        embedder.fail_after_batches = 0
        embedder.failure = Boom()
    elif stage == "read":
        monkeypatch.setattr(filtered_search, "filtered_lexical_search", boom)
    else:
        monkeypatch.setattr(
            filtered_search,
            "filtered_lexical_search",
            lambda *a: LexicalResult(tsquery="'x'", hits=[], lexical_ms=0.0),
        )
        monkeypatch.setattr(
            filtered_search, "filtered_dense_search", lambda *a: DenseResult(hits=[], vector_ms=0.0)
        )
        monkeypatch.setattr(filtered_api, "fuse_rrf", boom)
    with pytest.raises(Boom):
        filtered_api._filtered_search(
            session,
            make_settings(),
            embedder,
            provider,
            "hp laptop",
            None,
            ("query", "q"),
            ("query", "top_k"),
        )
    assert not session.in_transaction()
    if stage == "read":
        assert events[-1] == "rollback"


def _assert_generic_500(response, caplog, *secrets: str) -> None:
    # A programming error must not be disguised as the service-unavailable 503 (same boundary
    # as /search/hybrid), and neither the body nor the app log may leak details.
    assert response.status_code == 500 and response.text == "Internal Server Error"
    app_records = _app_records(caplog)
    assert not [r for r in app_records if r.levelno >= logging.WARNING]
    app_log = "\n".join(r.getMessage() for r in app_records)
    for secret in (*SECRETS, *secrets):
        assert secret not in response.text and secret not in app_log


@pytest.mark.parametrize("method", ["get", "post"])
def test_unexpected_read_error_is_a_generic_500_after_rollback_and_release(
    client_for, events, caplog, monkeypatch, method
):
    with caplog.at_level(logging.DEBUG), client_for(lexical_error=RuntimeError(SENSITIVE)) as c:
        records = install_spies(monkeypatch, events, RETRIEVAL_SIDE)
        response = call(TestClient(c.app, raise_server_exceptions=False), method, "hp laptop")
    _assert_generic_500(response, caplog, "RuntimeError")
    assert names(events)[names(events).index("call:read_filtered_sources") :] == [
        "call:read_filtered_sources",
        "begin",
        "execute",
        "lexical",
        "rollback",
        "close",
    ]
    assert [name for name, *_ in records] == ["encode_query"]  # one read attempt, no fusion
    assert not c.sessions[0].in_transaction() and c.sessions[0].closed


@pytest.mark.parametrize("method", ["get", "post"])
def test_malformed_source_list_is_a_generic_500_raised_after_the_transaction_ended(
    client_for, events, caplog, monkeypatch, method
):
    def duplicated_dense(session, vector, spec, filters, limit):
        events.append(("dense", len(vector), filters, limit))
        hit = dense_hits_for(filters, 1)[0]
        return DenseResult(hits=[hit, DenseHit(**{**hit.__dict__, "rank": 2})], vector_ms=0.0)

    with caplog.at_level(logging.DEBUG), client_for() as c:
        monkeypatch.setattr(filtered_search, "filtered_dense_search", duplicated_dense)
        install_spies(monkeypatch, events, RETRIEVAL_SIDE)
        response = call(TestClient(c.app, raise_server_exceptions=False), method, "hunter2 laptop")
    _assert_generic_500(response, caplog, "unique", "ValueError")
    # The unchanged fuse_rrf rejects the list only after the read-only transaction committed.
    assert names(events)[names(events).index("begin") :] == [
        "begin",
        "execute",
        "lexical",
        "dense",
        "commit",
        "call:fuse_rrf",
        "close",
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "   "},
        {"q": "shoes\x00"},
        {"q": "a\u200bb"},
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
        got = client.get("/search/filtered", params=params)
        posted = client.post("/search/filtered", json=body)
    assert got.status_code == posted.status_code == 422
    location = "top_k" if "top_k" in params else "q"
    assert got.json()["detail"][0]["loc"][-1] == location
    assert posted.json()["detail"][0]["loc"][-1] == ("top_k" if location == "top_k" else "query")
    assert provider.calls == 0 and "begin" not in events and client.fake.load_calls == 0
    assert "search unavailable" not in got.text + posted.text
    assert "shoes\x00" not in got.text and "x" * 201 not in got.text


@pytest.mark.parametrize(
    "body",
    [
        {"query": "\ud800"},
        {"query": "laptop", "extra": 1},
        {"query": "laptop", "filters": {"brand": "HP"}},  # no per-request filters
        {"query": "laptop", "applied_filters": []},
        {"query": "laptop", "rrf_k": 10},
        {"top_k": 5},
        {"query": 5},
    ],
)
def test_post_request_is_closed(client_for, events, body):
    provider = RaisingProvider()
    with client_for(provider=provider) as client:
        response = client.post(
            "/search/filtered",
            content=json.dumps(body, ensure_ascii=True),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422 and provider.calls == 0 and "begin" not in events


def test_get_ignores_unknown_filter_parameters(client_for):
    with client_for() as client:
        body = client.get(
            "/search/filtered", params={"q": "laptop", "brand": "Dell", "max_price": "1"}
        ).json()
    assert body["applied_filters"] == [A("category", "laptop")]


def test_query_over_the_model_token_limit_is_422_before_any_transaction(client_for, events):
    with client_for() as client:
        client.fake.token_overrides["toolong"] = 257
        response = call(client, "get", "toolong laptop")
    assert response.status_code == 422
    assert "toolong" not in response.text and "begin" not in events


# ---- 5. response contract ------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 3, 10])
def test_counts_results_and_fusion_are_consistent_with_the_filtered_sources(client_for, top_k):
    query, spec, applied, _ignored = CASES["ram"]
    with client_for() as client:
        body = call(client, "get", query, top_k).json()
    lexical = [hit.product_id for hit in lexical_hits_for(spec, 50)]
    dense = [hit.product_id for hit in dense_hits_for(spec, 50)]
    assert set(lexical) | set(dense) == eligible_ids(spec) == {"L1", "L3", "L4", "P1"}
    assert body["search_version"] == "v2_filtered"
    assert (body["lexical_hit_count"], body["dense_hit_count"]) == (len(lexical), len(dense))
    assert body["overlap_count"] == len(set(lexical) & set(dense))
    assert body["fused_count"] == len(set(lexical) | set(dense)) == body["candidate_count"]
    assert body["result_count"] == len(body["results"]) == min(top_k, body["candidate_count"])
    expected = rrf_oracle(lexical, dense, body["fusion"]["rrf_k"])[:top_k]
    assert [(r["product_id"], r["rrf_score"]) for r in body["results"]] == [
        (pid, float(score)) for pid, score in expected
    ]
    for position, result in enumerate(body["results"], start=1):
        product = next(p for p in CATALOG if p["id"] == result["product_id"])
        assert result["rank"] == position and satisfies(product, spec)
        assert result["lexical_rank"] == (
            lexical.index(product["id"]) + 1 if product["id"] in lexical else None
        )
        assert result["dense_rank"] == dense.index(product["id"]) + 1
        assert result["price"] == f"{product['price']}"  # a canonical decimal string
    assert body["applied_filters"] == applied


def test_candidate_k_and_top_k_truncate_as_in_v1(client_for):
    with client_for(search_candidate_k=3, search_default_top_k=2) as client:
        body = call(client, "get", "laptop").json()
        too_many = call(client, "get", "laptop", 4)
    assert body["fused_count"] == 4 and body["candidate_count"] == 3
    assert body["top_k"] == 2 and body["result_count"] == 2
    assert too_many.status_code == 422


def test_source_depths_come_from_settings(client_for, events):
    with client_for(
        search_lexical_k=7, search_dense_k=9, search_candidate_k=16, search_default_top_k=2
    ) as client:
        body = call(client, "get", "laptop").json()
    sources = {e[0]: e for e in events if isinstance(e, tuple) and e[0] in ("lexical", "dense")}
    assert sources["lexical"][3] == 7 and sources["dense"][3] == 9
    assert (body["fusion"]["lexical_k"], body["fusion"]["dense_k"]) == (7, 9)


@pytest.mark.parametrize("method", ["get", "post"])
def test_zero_matches_is_a_200_with_filters_preserved_and_no_retry(
    client_for, events, monkeypatch, method
):
    query, spec, applied, ignored = CASES["impossible"]
    assert eligible_ids(spec) == set()
    with client_for() as client:
        records = install_spies(monkeypatch, events, RETRIEVAL_SIDE)
        response = call(client, method, query)
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["result_count"] == 0 and body["candidate_count"] == 0
    assert body["lexical_hit_count"] == body["dense_hit_count"] == 0
    assert body["applied_filters"] == applied and body["ignored_constraints"] == ignored
    assert body["dense_status"] == "used"
    assert [name for name, *_ in records] == list(RETRIEVAL_SIDE)  # one read, never retried
    assert sum(1 for e in events if isinstance(e, tuple) and e[0] == "lexical") == 1
    assert "fallback" not in response.text.lower()


def test_response_contract_and_stable_serialization(client_for):
    query, _spec, applied, ignored = CASES["ambiguity-range"]
    with client_for() as client:
        first = call(client, "get", query)
        second = call(client, "post", query)
    body = first.json()
    assert list(body) == list(FilteredSearchResponse.model_fields)
    assert body["search_version"] == "v2_filtered"
    assert body["filter_policy"] == {"version": "fp-1"}
    assert body["applied_filters"] == applied and body["ignored_constraints"] == ignored
    assert all(isinstance(item, dict) for item in body["applied_filters"])  # not list[str]
    block = body["query_understanding"]
    assert set(block) == {"usage", "provider", "provider_version", "understanding"}
    assert (block["usage"], block["provider"], block["provider_version"]) == (
        "filter_source",
        "deterministic",
        "qu-1",
    )
    assert block["understanding"] == parse_normalized_query(query).model_dump(mode="json")
    latency = body["latency_ms"]
    assert set(latency) == {
        "lexical_ms",
        "model_load_ms",
        "query_embedding_ms",
        "vector_ms",
        "rrf_ms",
        "query_understanding_ms",
        "filter_translation_ms",
        "total_ms",
    }
    assert "filtering_ms" not in latency
    assert latency["query_understanding_ms"] >= 0 and latency["filter_translation_ms"] >= 0
    assert latency["total_ms"] >= latency["query_understanding_ms"]
    dumps = [
        json.dumps(without_latency(r.json()), sort_keys=False, ensure_ascii=True)
        for r in (first, second)
    ]
    assert dumps[0] == dumps[1]


def test_prices_in_filters_and_results_are_canonical_strings(client_for):
    with client_for() as client:
        body = call(client, "get", "hp laptop 8gb ram 256gb ssd under 40k").json()
    assert body["applied_filters"][-1] == A("max_price", "40000.00", "lte")
    assert [(r["product_id"], r["price"]) for r in body["results"]] == [
        ("L1", "33999.00"),
        ("L4", "38999.00"),
    ]
    assert '"price":"33999.00"' in json.dumps(body, separators=(",", ":"))


FORBIDDEN_KEYS = ("rerank", "confidence", "jev", "fallback", "pre_filter", "prefilter")


def _keys(value):
    for node in _walk(value):
        if isinstance(node, dict):
            yield from (key for key in node if isinstance(key, str))


@CASE_PARAMS
def test_no_placeholder_fields_and_both_latencies_on_every_200(client_for, case):
    with client_for() as client:
        body = call(client, "get", case[0]).json()
    for key in _keys(body):
        assert not any(word in key.lower() for word in FORBIDDEN_KEYS), key
        assert key != "filtering_ms"
    assert isinstance(body["latency_ms"]["query_understanding_ms"], float)
    assert isinstance(body["latency_ms"]["filter_translation_ms"], float)


def test_openapi_documents_the_v2_contract(client_for):
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    path = schema["paths"]["/search/filtered"]
    assert set(path) == {"get", "post"}
    assert {p["name"] for p in path["get"]["parameters"]} == {"q", "top_k"}
    assert path["post"]["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/FilteredSearchRequest"
    }
    request = components["FilteredSearchRequest"]
    assert set(request["properties"]) == {"query", "top_k"}
    assert request["additionalProperties"] is False
    for method in ("get", "post"):
        responses = path[method]["responses"]
        assert set(responses) >= {"200", "422", "503"}
        assert responses["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/FilteredSearchResponse"
        }
        assert responses["503"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/SearchUnavailable"
        }
    response = components["FilteredSearchResponse"]
    assert set(response["required"]) == set(response["properties"])
    assert response["properties"]["search_version"]["const"] == "v2_filtered"
    assert response["properties"]["filter_policy"] == {"$ref": "#/components/schemas/FilterPolicy"}
    assert response["properties"]["applied_filters"]["items"] == {
        "$ref": "#/components/schemas/AppliedFilter"
    }
    assert response["properties"]["ignored_constraints"]["items"] == {
        "$ref": "#/components/schemas/IgnoredConstraint"
    }
    assert response["properties"]["query_understanding"] == {
        "$ref": "#/components/schemas/FilteredQueryUnderstandingBlock"
    }
    assert response["properties"]["results"]["items"] == {
        "$ref": "#/components/schemas/HybridSearchResult"
    }
    assert components["FilterPolicy"]["properties"]["version"]["const"] == "fp-1"
    block = components["FilteredQueryUnderstandingBlock"]
    assert block["properties"]["usage"]["const"] == "filter_source"
    assert block["additionalProperties"] is False
    assert set(components["AppliedFilter"]["properties"]) == {"field", "operator", "value"}
    assert set(components["IgnoredConstraint"]["properties"]) == {"field", "reason"}
    latency = components["FilteredSearchLatency"]
    assert "filtering_ms" not in latency["properties"]
    for name in ("query_understanding_ms", "filter_translation_ms", "total_ms"):
        assert latency["properties"][name]["type"] == "number" and name in latency["required"]
    v2_components = (
        "FilteredSearchResponse",
        "FilteredSearchLatency",
        "FilteredQueryUnderstandingBlock",
        "FilterPolicy",
        "AppliedFilter",
        "IgnoredConstraint",
        "FilteredSearchRequest",
    )
    for name in v2_components:
        for key in _keys(components[name]["properties"]):
            assert not any(word in key.lower() for word in FORBIDDEN_KEYS), (name, key)


# ---- 6. frozen baseline ------------------------------------------------------------------------

FROZEN_PATHS = ("/search", "/search/dense", "/search/hybrid")
FROZEN_OPENAPI_PATHS = ("/health", *FROZEN_PATHS)
# sha256 of the canonical JSON of the four frozen operations plus every component they
# reference (transitively), computed from an export of HEAD a5cc374 (before any Milestone 7
# Session E change), independently of the code under test.
HEAD_FROZEN_DIGEST = "eaae8fcb0b2b49360cd444fd9b611eefc08520b4cf689f432916f4ddd91b92a6"


def _refs(node, out: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                out.add(value.rsplit("/", 1)[-1])
            else:
                _refs(value, out)
    elif isinstance(node, list):
        for value in node:
            _refs(value, out)


def frozen_openapi(schema: dict) -> dict:
    paths = {path: schema["paths"][path] for path in FROZEN_OPENAPI_PATHS}
    components = schema["components"]["schemas"]
    seen: set = set()
    todo: set = set()
    _refs(paths, todo)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        found: set = set()
        _refs(components[name], found)
        todo |= found - seen
    return {"paths": paths, "components": {name: components[name] for name in sorted(seen)}}


def _digest(value: dict) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_frozen_operations_and_components_are_unchanged_from_head(make_settings):
    schema = create_app(make_settings()).openapi()
    frozen = frozen_openapi(schema)
    assert _digest(frozen) == HEAD_FROZEN_DIGEST
    assert set(schema["paths"]) == {*FROZEN_OPENAPI_PATHS, "/search/filtered"}
    # The same app without the V2 router publishes byte-identical frozen operations.
    legacy = FastAPI(title="Intelligent E-commerce Search", version=schema["info"]["version"])
    for router in (health_router, search_router, dense_router, hybrid_router):
        legacy.include_router(router)
    assert frozen_openapi(legacy.openapi()) == frozen
    v2_only = {
        "FilteredSearchRequest",
        "FilteredSearchResponse",
        "FilteredSearchLatency",
        "FilteredQueryUnderstandingBlock",
        "FilterPolicy",
        "AppliedFilter",
        "IgnoredConstraint",
        "FilterField",
        "FilterOperator",
        "IgnoreReason",
    }
    assert not v2_only & set(frozen["components"])
    for name in ("SearchResponse", "DenseSearchResponse", "HybridSearchResponse"):
        applied = frozen["components"][name]["properties"]["applied_filters"]
        assert applied["type"] == "array" and applied["items"] == {"type": "string"}


LEGACY_KEYS = {
    "/search": {
        "query",
        "search_version",
        "document_version",
        "top_k",
        "result_count",
        "results",
        "tsquery",
        "applied_filters",
    },
    "/search/dense": {
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
    },
    "/search/hybrid": {
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
        "query_understanding",
    },
}
UNFILTERED = FilterSpec()


def _install_legacy_sources(monkeypatch, events: list) -> None:
    def lexical(session, query, limit):
        events.append(("v1-lexical", query, limit))
        return LexicalResult(
            tsquery="'x'", hits=lexical_hits_for(UNFILTERED, limit), lexical_ms=1.5
        )

    def dense(session, vector, spec, limit):
        events.append(("v1-dense", len(vector), limit))
        return DenseResult(hits=dense_hits_for(UNFILTERED, limit), vector_ms=2.5)

    monkeypatch.setattr(search_api, "lexical_search", lexical)
    monkeypatch.setattr(dense_api, "dense_search", dense)
    monkeypatch.setattr(hybrid_search, "lexical_search", lexical)
    monkeypatch.setattr(hybrid_search, "dense_search", dense)


def _legacy_responses(client) -> dict:
    out = {}
    for path in FROZEN_PATHS:
        for method in ("get", "post"):
            if method == "get":
                response = client.get(path, params={"q": "hp laptop 8gb ram", "top_k": 5})
            else:
                response = client.post(path, json={"query": "hp laptop 8gb ram", "top_k": 5})
            assert response.status_code == 200, (path, method)
            body = response.json()
            assert set(body) - {"latency_ms"} == LEGACY_KEYS[path], path
            assert body["applied_filters"] == []
            out[path, method] = without_latency(body)
    return out


def _raising_v2_stubs(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("V2 code must not run for V0/dense/V1 requests")

    monkeypatch.setattr(filtered_api, "derive_filters", broken)
    monkeypatch.setattr(filtered_api, "read_filtered_sources", broken)
    monkeypatch.setattr(policy_module, "derive_filters", broken)
    monkeypatch.setattr(filtering_package, "derive_filters", broken)
    monkeypatch.setattr(filtered_search, "read_filtered_sources", broken)
    monkeypatch.setattr(filtered_search, "filtered_lexical_search", broken)
    monkeypatch.setattr(filtered_search, "filtered_dense_search", broken)


def test_legacy_endpoints_are_unaffected_by_raising_v2_code(client_for, events, monkeypatch):
    with client_for() as client:
        _install_legacy_sources(monkeypatch, events)
        baseline = _legacy_responses(client)
        _raising_v2_stubs(monkeypatch)
        stubbed = _legacy_responses(client)
        v2 = call(client, "get", "hp laptop")
    assert stubbed == baseline
    assert baseline["/search/hybrid", "get"]["query_understanding"]["usage"] == "informational"
    assert v2.status_code == 503  # the stubs really are installed for the V2 route


V2_ONLY = ("search_version", "filter_policy", "applied_filters", "ignored_constraints")


@pytest.mark.parametrize("query", ["comfortable gift", "zzqxv"])
def test_v2_with_empty_filters_equals_v1(client_for, events, monkeypatch, query):
    with client_for() as client:
        _install_legacy_sources(monkeypatch, events)
        v1 = client.get("/search/hybrid", params={"q": query, "top_k": 50}).json()
        v2 = client.get("/search/filtered", params={"q": query, "top_k": 50}).json()
    assert v2["applied_filters"] == [] and v2["ignored_constraints"] == []
    assert (v1["search_version"], v2["search_version"]) == ("v1_hybrid", "v2_filtered")
    assert v1["query_understanding"]["understanding"] == v2["query_understanding"]["understanding"]
    assert v2["results"] and v2["fused_count"] == len(CATALOG)
    for payload in (v1, v2):
        for key in (*V2_ONLY, "query_understanding", "latency_ms"):
            payload.pop(key, None)
    assert v1 == v2  # same source results, ranks, scores, fusion order and counts
    assert [e[0] for e in events if isinstance(e, tuple)].count("lexical") == 1
