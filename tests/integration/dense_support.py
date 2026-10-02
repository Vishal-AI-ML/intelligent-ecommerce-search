"""Helpers for the Milestone 4 dense integration tests (scratch databases only)."""

import hashlib

from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.search.dense import dense_search
from ecommerce_search.search.dense_indexing import embed, embedding_status
from fake_embedder import FakeEmbedder

SPEC = ALL_MINILM_L6_V2
TABLE = "product_embeddings"


def sql(engine, statement: str, **params):
    with engine.begin() as conn:
        return conn.execute(text(statement), params)


def embeddings(engine) -> dict[str, dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT product_id, model_id, model_revision, embedding_config_sha256, "
                "source_content_sha256, embedding_text_sha256, embedded_at, "
                "embedding::text AS vector FROM product_embeddings"
            )
        ).mappings()
        return {row["product_id"]: dict(row) for row in rows}


def status(engine, spec=SPEC, embedder=None):
    with Session(engine) as session:
        return embedding_status(session, spec, embedder)


def embed_fake(engine, fake: FakeEmbedder | None = None, **kwargs):
    return embed(engine, fake or FakeEmbedder(), **kwargs)


def dense_ids(engine, query: str, limit: int = 240, spec=SPEC) -> list[str]:
    vector = FakeEmbedder(spec).embed_query(query)
    with Session(engine) as session:
        return [hit.product_id for hit in dense_search(session, vector, spec, limit).hits]


def change_product(engine, product_id: str, title: str) -> str:
    """Simulate a committed catalog change: new title and a new content hash."""
    new_sha = hashlib.sha256(f"{product_id}:{title}".encode()).hexdigest()
    sql(
        engine,
        "UPDATE products SET title = :t, content_sha256 = :s WHERE product_id = :p",
        t=title,
        s=new_sha,
        p=product_id,
    )
    return new_sha
