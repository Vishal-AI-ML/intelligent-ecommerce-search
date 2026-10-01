"""V0 lexical retrieval: PostgreSQL full-text search with strict AND semantics.

`plainto_tsquery('simple', q)` ANDs every term and never raises on user text. Results are
ordered by `ts_rank` (descending) and then `product_id` (ascending), so ties are deterministic.
Scores are query-relative and uncalibrated: they are not probabilities and are comparable only
within one query.

The index is a derived table. This service does not claim it is complete: if documents are
missing or stale, results may be incomplete until `search reindex` succeeds.
"""

import time
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.search.documents import FTS_CONFIG
from ecommerce_search.search.query import RANK_NORMALIZATION, RANK_WEIGHT_ARRAY

TSQUERY_SQL = text(
    """
    SELECT CAST(plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text)) AS text),
           numnode(plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text)))
    """
)
# Also used by the benchmark for EXPLAIN. `:limit` is capped by the API layer, not here.
RETRIEVAL_SQL = text(
    """
    SELECT p.product_id, p.title, p.brand, p.category, p.subcategory, p.description,
           p.price, p.currency, p.rating, p.review_count, p.availability,
           ts_rank(CAST(:weights AS real[]), d.search_vector,
                   plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text)),
                   CAST(:normalization AS integer)) AS lexical_score
    FROM product_search_documents d
    JOIN products p ON p.product_id = d.product_id
    WHERE d.search_vector @@ plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text))
    ORDER BY lexical_score DESC, p.product_id ASC
    LIMIT :limit
    """
)


@dataclass(frozen=True)
class LexicalHit:
    rank: int
    lexical_score: float
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
class LexicalResult:
    tsquery: str
    hits: list[LexicalHit]
    lexical_ms: float


def retrieval_params(query: str, limit: int) -> dict:
    return {
        "cfg": FTS_CONFIG,
        "q": query,
        "weights": RANK_WEIGHT_ARRAY,
        "normalization": RANK_NORMALIZATION,
        "limit": limit,
    }


def lexical_search(session: Session, query: str, limit: int) -> LexicalResult:
    """Return up to `limit` hits for an already-normalized query. Database errors propagate."""
    start = time.perf_counter()
    tsquery, node_count = session.execute(TSQUERY_SQL, {"cfg": FTS_CONFIG, "q": query}).one()
    hits: list[LexicalHit] = []
    if node_count > 0:  # punctuation-only text yields an empty query: no rows, no error
        rows = session.execute(RETRIEVAL_SQL, retrieval_params(query, limit)).all()
        hits = [
            LexicalHit(
                rank=position,
                lexical_score=float(row.lexical_score),
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
    elapsed_ms = (time.perf_counter() - start) * 1000
    return LexicalResult(tsquery=tsquery, hits=hits, lexical_ms=round(elapsed_ms, 3))
