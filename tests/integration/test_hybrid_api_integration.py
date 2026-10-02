"""`/search/hybrid` end to end against real PostgreSQL scratch databases (fake model)."""

import logging
from fractions import Fraction

import pytest
from dense_support import change_product, embed_fake
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event

from ecommerce_search.api.app import create_app
from ecommerce_search.api.dependencies import get_embedder
from ecommerce_search.api.schemas import HybridSearchResponse
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


def make_client(settings, engine, fake):
    scratch = settings.model_copy(update={"postgres_db": engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    app = create_app(scratch)
    app.dependency_overrides[get_embedder] = lambda: fake
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
