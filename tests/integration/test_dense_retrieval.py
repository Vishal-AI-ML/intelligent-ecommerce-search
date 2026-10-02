"""Exact cosine retrieval over pgvector on scratch databases (handcrafted vectors)."""

import json
import math

import pytest
from dense_support import SPEC, embed_fake, sql
from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.search.dense import RETRIEVAL_SQL, dense_search, retrieval_params

pytestmark = pytest.mark.integration


def axis(i: int, *, scale: float = 1.0, also: tuple[int, float] | None = None) -> list[float]:
    vector = [0.0] * 384
    vector[i] = scale
    if also is not None:
        vector[also[0]] = also[1]
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def set_vector(engine, product_id: str, vector: list[float]) -> None:
    sql(
        engine,
        "UPDATE product_embeddings SET embedding = CAST(:v AS vector) WHERE product_id = :p",
        v="[" + ",".join(repr(v) for v in vector) + "]",
        p=product_id,
    )


@pytest.fixture(scope="module")
def handcrafted(writable_seeded_engine):
    engine = writable_seeded_engine
    embed_fake(engine)
    # Every product points at axis 300; a few chosen products get controlled vectors.
    sql(
        engine,
        "UPDATE product_embeddings SET embedding = CAST(:v AS vector)",
        v="[" + ",".join(repr(v) for v in axis(300)) + "]",
    )
    set_vector(engine, "SYN-PHN-0001", axis(0))  # identical to the query
    set_vector(engine, "SYN-LAP-0002", axis(0, also=(1, 0.5)))  # close
    set_vector(engine, "SYN-LAP-0001", axis(0, also=(1, 0.5)))  # tie with LAP-0002
    set_vector(engine, "SYN-HDP-0001", axis(0, also=(1, 2.0)))  # further
    set_vector(engine, "SYN-SHO-0001", axis(0, scale=-1.0))  # opposite
    return engine


def search(engine, vector, limit=10, spec=SPEC):
    with Session(engine) as session:
        return dense_search(session, vector, spec, limit)


def test_results_are_ordered_by_cosine_distance_then_product_id(handcrafted):
    hits = search(handcrafted, axis(0), limit=5).hits
    assert [h.product_id for h in hits] == [
        "SYN-PHN-0001",
        "SYN-LAP-0001",  # tie with LAP-0002 broken by product_id ascending
        "SYN-LAP-0002",
        "SYN-HDP-0001",
        hits[4].product_id,
    ]
    assert hits[0].dense_score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].dense_score == hits[2].dense_score == pytest.approx(2 / math.sqrt(5), abs=1e-6)
    assert hits[3].dense_score == pytest.approx(1 / math.sqrt(5), abs=1e-6)
    assert hits[4].dense_score == pytest.approx(0.0, abs=1e-6)  # orthogonal axis-300 products
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_scores_are_cosine_similarity_including_negative_values(handcrafted):
    hits = search(handcrafted, axis(0), limit=240).hits
    assert len(hits) == 240
    assert hits[-1].product_id == "SYN-SHO-0001"
    assert hits[-1].dense_score == pytest.approx(-1.0, abs=1e-6)
    scores = [h.dense_score for h in hits]
    assert scores == sorted(scores, reverse=True) and all(
        -1 - 1e-6 <= s <= 1 + 1e-6 for s in scores
    )
    ties = [h for h in hits if h.dense_score == pytest.approx(0.0, abs=1e-9)]
    assert [h.product_id for h in ties] == sorted(h.product_id for h in ties)


def test_top_k_is_respected_and_results_are_deterministic(handcrafted):
    first = [h.product_id for h in search(handcrafted, axis(0), limit=7).hits]
    assert len(first) == 7
    for _ in range(3):
        assert [h.product_id for h in search(handcrafted, axis(0), limit=7).hits] == first


def test_rows_for_another_model_or_text_version_are_excluded(handcrafted):
    try:
        sql(
            handcrafted,
            "UPDATE product_embeddings SET model_revision = :r WHERE product_id = :p",
            r="f" * 40,
            p="SYN-PHN-0001",
        )
        sql(
            handcrafted,
            "UPDATE product_embeddings SET embedding_text_version = '0' "
            "WHERE product_id = 'SYN-LAP-0001'",
        )
        sql(
            handcrafted,
            "UPDATE product_embeddings SET embedding_config_sha256 = :c "
            "WHERE product_id = 'SYN-LAP-0002'",
            c="0" * 64,
        )
        ids = [h.product_id for h in search(handcrafted, axis(0), limit=240).hits]
        assert len(ids) == 237
        assert not {"SYN-PHN-0001", "SYN-LAP-0001", "SYN-LAP-0002"} & set(ids)
        assert ids[0] == "SYN-HDP-0001"
    finally:
        sql(
            handcrafted,
            "UPDATE product_embeddings SET model_revision = :r, "
            "embedding_text_version = '1', embedding_config_sha256 = :c",
            r=SPEC.revision,
            c=SPEC.config_sha256(),
        )


def test_stale_content_rows_are_excluded(handcrafted):
    try:
        sql(
            handcrafted,
            "UPDATE products SET content_sha256 = :s WHERE product_id = 'SYN-PHN-0001'",
            s="7" * 64,
        )
        ids = [h.product_id for h in search(handcrafted, axis(0), limit=3).hits]
        assert "SYN-PHN-0001" not in ids and ids[0] == "SYN-LAP-0001"
    finally:
        sql(
            handcrafted,
            "UPDATE products p SET content_sha256 = e.source_content_sha256 "
            "FROM product_embeddings e WHERE e.product_id = p.product_id "
            "AND p.product_id = 'SYN-PHN-0001'",
        )
    assert search(handcrafted, axis(0), limit=1).hits[0].product_id == "SYN-PHN-0001"


def test_query_vector_is_a_bound_parameter(handcrafted):
    params = retrieval_params(axis(0), SPEC, 5)
    assert params["qvec"].startswith("[") and ":qvec" in str(RETRIEVAL_SQL)
    assert "'" not in str(RETRIEVAL_SQL).split("WHERE")[1]  # no literal values in the predicate


def test_exact_scan_plan_uses_no_vector_index(handcrafted):
    with handcrafted.connect() as conn:
        conn.execute(text("ANALYZE product_embeddings"))
        plan = conn.execute(
            text("EXPLAIN (FORMAT JSON) " + str(RETRIEVAL_SQL)),
            retrieval_params(axis(0), SPEC, 10),
        ).scalar_one()
    rendered = json.dumps(plan)
    assert "hnsw" not in rendered.lower() and "ivfflat" not in rendered.lower()
    assert '"Relation Name": "product_embeddings"' in rendered


def test_an_empty_index_returns_no_results(migrated_engine):
    assert search(migrated_engine, axis(0)).hits == []
