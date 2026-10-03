"""`/search/hybrid` end to end against real PostgreSQL scratch databases (fake model)."""

import logging
from decimal import Decimal
from fractions import Fraction

import pytest
from dense_support import change_product, embed_fake
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from ecommerce_search.api import hybrid as hybrid_api
from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_decision_provider, get_embedder
from ecommerce_search.api.schemas import HybridSearchResponse
from ecommerce_search.catalog.taxonomy import Category
from ecommerce_search.decision import DecisionResult
from ecommerce_search.query_understanding import Intent, QueryUnderstanding
from ecommerce_search.search.dense import dense_search
from ecommerce_search.search.lexical import lexical_search
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

PID = "SYN-SHO-0001"
QUERIES = ["hp laptop", "wireless headphones", "running shoes", "16gb ram", "zzqxv"]


class RecordingEmbedder(FakeEmbedder):
    """Records the pool's checked-out connections while the model loads and encodes."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = None
        self.events: list = []

    def load(self):
        self.events.append(("load", self.engine.pool.checkedout()))
        return super().load()

    def embed_query(self, text):
        self.events.append(("embed", self.engine.pool.checkedout()))
        return super().embed_query(text)


def make_client(settings, engine, fake, provider=None):
    scratch = settings.model_copy(update={"postgres_db": engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    app = create_app(scratch)
    app.dependency_overrides[get_embedder] = lambda: fake
    if provider is not None:
        app.dependency_overrides[get_decision_provider] = lambda: provider
    return TestClient(app)


@pytest.fixture
def catalog(migrated_engine):
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    ingest_file(migrated_engine, SEED, PROVENANCE)
    return migrated_engine


def oracle(lexical: list[str], dense: list[str], rrf_k: int) -> list[str]:
    scores: dict[str, Fraction] = {}
    for ranking in (lexical, dense):
        for position, product_id in enumerate(ranking, start=1):
            scores[product_id] = scores.get(product_id, Fraction(0)) + Fraction(1, rrf_k + position)
    return [pid for pid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def test_sources_match_search_and_dense_and_fusion_matches_an_oracle(settings, catalog):
    embed_fake(catalog)
    with make_client(settings, catalog, FakeEmbedder()) as client:
        for query in QUERIES:
            lexical = client.get("/search", params={"q": query, "top_k": 50}).json()["results"]
            dense = client.get("/search/dense", params={"q": query, "top_k": 50}).json()["results"]
            response = client.get("/search/hybrid", params={"q": query, "top_k": 50})
            assert response.status_code == 200
            body = HybridSearchResponse.model_validate(response.json())
            assert body.lexical_hit_count == len(lexical)
            assert body.dense_hit_count == len(dense) == 50
            by_lexical = {r["product_id"]: r for r in lexical}
            by_dense = {r["product_id"]: r for r in dense}
            assert body.fusion.rrf_k == settings.search_rrf_k
            expected = oracle(list(by_lexical), list(by_dense), body.fusion.rrf_k)[:50]
            assert [r.product_id for r in body.results] == expected
            assert body.overlap_count == len(by_lexical.keys() & by_dense.keys())
            assert body.fused_count == len(by_lexical.keys() | by_dense.keys())
            for result in body.results:
                lex = by_lexical.get(result.product_id)
                den = by_dense.get(result.product_id)
                assert result.lexical_rank == (lex and lex["rank"])
                assert result.lexical_score == (lex and lex["lexical_score"])
                assert result.dense_rank == (den and den["rank"])
                assert result.dense_score == (den and den["dense_score"])
                if lex and den:  # both reads share one snapshot: display fields agree
                    fields = set(lex) - {"rank", "lexical_score"}
                    assert {f: lex[f] for f in fields} == {f: den[f] for f in fields}
                source = lex or den
                assert result.title == source["title"] and str(result.price) == source["price"]
            assert len({r.product_id for r in body.results}) == len(body.results)


def test_model_work_precedes_the_transaction_and_no_connection_stays_checked_out(settings, catalog):
    embed_fake(catalog)
    fake = RecordingEmbedder()
    with make_client(settings, catalog, fake) as client:
        engine = client.app.state.engine
        fake.engine = engine
        begins: list = []
        event.listen(engine, "begin", lambda conn: begins.append(len(fake.events)))
        first = client.get("/search/hybrid", params={"q": "running shoes"})
        assert engine.pool.checkedout() == 0
        events_after_first = len(fake.events)
        second = client.post("/search/hybrid", json={"query": "running shoes"})
        assert engine.pool.checkedout() == 0
    assert first.status_code == second.status_code == 200
    assert fake.events and all(checked_out == 0 for _, checked_out in fake.events)
    # One transaction per request, each begun only after that request's model work finished.
    assert begins == [events_after_first, len(fake.events)]
    a, b = first.json(), second.json()
    a.pop("latency_ms"), b.pop("latency_ms")
    assert a == b  # repeated requests are identical


def test_stale_product_appears_lexically_with_no_dense_rank(settings, catalog):
    embed_fake(catalog)
    change_product(catalog, PID, "Nike Strato Running Shoes for Men - Blue Model SH101")
    with make_client(settings, catalog, FakeEmbedder()) as client:
        body = client.get("/search/hybrid", params={"q": "Strato running shoes", "top_k": 50})
    result = next(r for r in body.json()["results"] if r["product_id"] == PID)
    assert result["lexical_rank"] is not None
    assert result["dense_rank"] is None and result["dense_score"] is None


def test_zero_current_embeddings_returns_fewer_candidates_not_an_error(settings, catalog):
    with make_client(settings, catalog, FakeEmbedder()) as client:
        lexical = client.get("/search", params={"q": "hp laptop", "top_k": 50}).json()
        response = client.get("/search/hybrid", params={"q": "hp laptop", "top_k": 50})
    body = response.json()
    assert response.status_code == 200 and body["dense_status"] == "used"
    assert body["dense_hit_count"] == 0 and body["overlap_count"] == 0
    assert body["candidate_count"] == body["lexical_hit_count"] == len(lexical["results"])
    assert [r["product_id"] for r in body["results"]] == [
        r["product_id"] for r in lexical["results"]
    ]
    assert "fallback" not in response.text.lower()


def test_no_searchable_text_opens_no_transaction(settings, catalog):
    embed_fake(catalog)
    fake = RecordingEmbedder()
    with make_client(settings, catalog, fake) as client:
        engine = client.app.state.engine
        fake.engine = engine
        begins: list = []
        event.listen(engine, "begin", lambda conn: begins.append(True))
        response = client.get("/search/hybrid", params={"q": "!!!"})
    assert response.status_code == 200 and response.json()["results"] == []
    assert begins == [] and fake.events == [] and fake.load_calls == 0


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_database_below_0004_is_a_fixed_503(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database, "0003")
    # Alembic's fileConfig replaces the root handlers, so the hybrid logger gets its own.
    records = _Records()
    hybrid_logger = logging.getLogger("ecommerce_search.api.hybrid")
    hybrid_logger.addHandler(records)
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        with make_client(settings, engine, FakeEmbedder()) as client:
            response = client.get("/search/hybrid", params={"q": "laptop"})
            assert client.app.state.engine.pool.checkedout() == 0
            lexical = client.get("/search", params={"q": "laptop"})
            health = client.get("/health")
    finally:
        hybrid_logger.removeHandler(records)
        engine.dispose()
    assert response.status_code == 503 and response.json() == {"detail": "search unavailable"}
    assert any("table_missing" in message for message in records.messages)
    assert "product_embeddings" not in response.text
    assert lexical.status_code == 200 and health.status_code == 200


def test_missing_model_snapshot_is_a_fixed_503(settings, catalog, tmp_path):
    scratch = settings.model_copy(
        update={"postgres_db": catalog.url.database, "embedding_models_dir": tmp_path}
    )
    with TestClient(create_app(scratch)) as client:  # the real provider, empty models dir
        hybrid = client.get("/search/hybrid", params={"q": "laptop"})
        lexical = client.get("/search", params={"q": "laptop"})
    assert hybrid.status_code == 503 and hybrid.json() == {"detail": "search unavailable"}
    assert str(tmp_path) not in hybrid.text
    assert lexical.status_code == 200


# ---- query understanding (Milestone 6): informational only -------------------------------------

UNDERSTANDING_QUERIES = {
    "english": "hp gaming laptop",
    "hinglish": "sasta phone 50k ke andar",
    "numeric": "hp laptop 8gb 256gb ssd under 40k",
    "punctuation": "!!!",
}


class StubProvider:
    """A valid decision that differs between variants in every value it sets."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        a = self.variant == "A"
        return DecisionResult(
            understanding=QueryUnderstanding(
                raw_query=query,
                category=Category.LAPTOP if a else Category.SHOES,
                brand="StubBrandA" if a else "StubBrandB",
                ram_gb=8 if a else 64,
                min_price=Decimal("10.00") if a else Decimal("20.00"),
                max_price=Decimal("99.99") if a else Decimal("999.99"),
                semantic_intent=Intent.GAMING if a else Intent.OFFICE,
            )
        )


class RaisingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def understand(self, query, deterministic_result):
        self.calls += 1
        raise RuntimeError("provider failure with secret detail")


def _capture_sources(monkeypatch) -> list:
    captured: list = []
    for name in ("read_sources", "fuse_rrf"):
        original = getattr(hybrid_api, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if _name == "read_sources":
                lexical, dense = result
                captured.append((_name, args[1], lexical.tsquery, lexical.hits, dense.hits))
            else:
                captured.append((_name, result))
            return result

        monkeypatch.setattr(hybrid_api, name, spy)
    return captured


def test_providers_a_and_b_give_identical_retrieval_and_fusion(settings, catalog, monkeypatch):
    embed_fake(catalog)
    captured: dict = {}
    responses: dict = {}
    for variant in ("A", "B"):
        provider = StubProvider(variant)
        with make_client(settings, catalog, FakeEmbedder(), provider) as client:
            captured[variant] = _capture_sources(monkeypatch)
            for name, query in UNDERSTANDING_QUERIES.items():
                response = client.get("/search/hybrid", params={"q": query, "top_k": 50})
                assert response.status_code == 200, name
                responses[variant, name] = response.json()
            monkeypatch.undo()
        assert provider.calls == len(UNDERSTANDING_QUERIES)
    # Three searchable queries each read both sources once and fuse once; "!!!" reads nothing.
    assert [entry[0] for entry in captured["A"]] == ["read_sources", "fuse_rrf"] * 3
    assert captured["A"] == captured["B"]  # source lists, fused IDs, ranks and exact scores
    for name, query in UNDERSTANDING_QUERIES.items():
        a, b = responses["A", name], responses["B", name]
        assert a["applied_filters"] == b["applied_filters"] == []
        assert a["query_understanding"]["understanding"]["brand"] == "StubBrandA"
        assert b["query_understanding"]["understanding"]["brand"] == "StubBrandB"
        assert a["latency_ms"]["query_understanding_ms"] is not None
        assert b["latency_ms"]["query_understanding_ms"] is not None
        for payload in (a, b):
            payload.pop("query_understanding"), payload.pop("latency_ms")
        assert a == b, name
        if query == "!!!":
            assert a["results"] == [] and a["dense_status"] == "skipped_no_searchable_text"
            continue
        # Independent oracle: the two sources run separately, then fused with exact fractions.
        spec = settings.embedding_spec()
        with Session(catalog) as session:
            lexical = lexical_search(session, query, settings.search_lexical_k).hits
            dense = dense_search(
                session, FakeEmbedder().embed_query(query), spec, settings.search_dense_k
            ).hits
        rrf_k = a["fusion"]["rrf_k"]
        assert rrf_k == settings.search_rrf_k
        expected = oracle(
            [hit.product_id for hit in lexical], [hit.product_id for hit in dense], rrf_k
        )[: settings.search_candidate_k][:50]
        assert [r["product_id"] for r in a["results"]] == expected, name
        lexical_rank = {hit.product_id: hit.rank for hit in lexical}
        dense_rank = {hit.product_id: hit.rank for hit in dense}
        for position, result in enumerate(a["results"], start=1):
            pid = result["product_id"]
            assert result["rank"] == position
            assert result["lexical_rank"] == lexical_rank.get(pid)
            assert result["dense_rank"] == dense_rank.get(pid)
            score = sum(
                Fraction(1, rrf_k + rank)
                for rank in (lexical_rank.get(pid), dense_rank.get(pid))
                if rank is not None
            )
            assert result["rrf_score"] == float(score)
        assert a["lexical_hit_count"] == len(lexical) and a["dense_hit_count"] == len(dense)


def test_provider_failure_is_a_fixed_503_without_a_connection_checkout(settings, catalog):
    embed_fake(catalog)
    fake = RecordingEmbedder()
    provider = RaisingProvider()
    with make_client(settings, catalog, fake, provider) as client:
        engine = client.app.state.engine
        fake.engine = engine
        checkouts: list = []
        event.listen(engine.pool, "checkout", lambda *args: checkouts.append(True))
        got = client.get("/search/hybrid", params={"q": "hp laptop"})
        posted = client.post("/search/hybrid", json={"query": "hp laptop"})
        assert engine.pool.checkedout() == 0
    for response in (got, posted):
        assert response.status_code == 503
        assert response.json() == {"detail": "search unavailable"}
        assert "secret detail" not in response.text
    assert provider.calls == 2
    assert checkouts == []  # no connection was ever checked out
    assert fake.events == [] and fake.load_calls == 0
