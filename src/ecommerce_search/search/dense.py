"""Dense retrieval: exact cosine-distance search over `product_embeddings` (pgvector).

Only embeddings that are current for the active configuration take part: same model id,
immutable revision, configuration hash, embedding-text version, dimension and normalization,
AND built from the product's current content (`source_content_sha256 =
products.content_sha256`). An updated product therefore never ranks with its old vector; it is
absent from dense results until `embed` heals it, and results may be incomplete while
embeddings are missing or stale (`embed-status` reports this; search never claims completeness).

Ordering is cosine distance ascending, then `product_id` ascending, so ties are deterministic.
`dense_score = 1 - cosine distance` (cosine similarity, -1..1): uncalibrated, not a
probability, comparable only within one query and one model. There is no similarity threshold:
every query returns up to `limit` nearest products, even a nonsense query.

No vector index exists (ADR-006): at the current catalog size PostgreSQL scans exactly.
"""

import time
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.search.dense_indexing import vector_literal

DISTANCE_METRIC = "cosine"
DENSE_SEARCH_VERSION = "dense_only"

# Also used by tests/benchmarks for EXPLAIN. Every value is a bound parameter.
RETRIEVAL_SQL = text(
    """
    SELECT p.product_id, p.title, p.brand, p.category, p.subcategory, p.description,
           p.price, p.currency, p.rating, p.review_count, p.availability,
           e.embedding <=> CAST(:qvec AS vector) AS distance
    FROM product_embeddings e
    JOIN products p ON p.product_id = e.product_id
    WHERE e.model_id = :model_id
      AND e.model_revision = :model_revision
      AND e.embedding_config_sha256 = :config_sha
      AND e.embedding_text_version = :text_version
      AND e.dimension = :dimension
      AND e.normalized = :normalized
      AND e.source_content_sha256 = p.content_sha256
    ORDER BY distance ASC, p.product_id ASC
    LIMIT :limit
    """
)


@dataclass(frozen=True)
class DenseHit:
    rank: int
    dense_score: float
    product_id: str
    title: str
    brand: str
    category: str
    subcategory: str | None
    description: str | None
    price: Decimal
    currency: str
    rating: Decimal | None
    review_count: int | None
    availability: str


@dataclass(frozen=True)
class DenseResult:
    hits: list[DenseHit]
    vector_ms: float


def retrieval_params(query_vector, spec: EmbeddingModelSpec, limit: int) -> dict:
    return {
        "qvec": vector_literal(query_vector),
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "config_sha": spec.config_sha256(),
        "text_version": EMBEDDING_TEXT_VERSION,
        "dimension": spec.dimension,
        "normalized": spec.normalize,
        "limit": limit,
    }


def dense_search(
    session: Session, query_vector, spec: EmbeddingModelSpec, limit: int
) -> DenseResult:
    """Up to `limit` nearest current products. Database errors propagate."""
    start = time.perf_counter()
    rows = session.execute(RETRIEVAL_SQL, retrieval_params(query_vector, spec, limit)).all()
    hits = [
        DenseHit(
            rank=position,
            dense_score=1.0 - float(row.distance),
            product_id=row.product_id,
            title=row.title,
            brand=row.brand,
            category=row.category,
            subcategory=row.subcategory,
            description=row.description,
            price=row.price,
            currency=row.currency,
            rating=row.rating,
            review_count=row.review_count,
            availability=row.availability,
        )
        for position, row in enumerate(rows, start=1)
    ]
    return DenseResult(hits=hits, vector_ms=round((time.perf_counter() - start) * 1000, 3))
