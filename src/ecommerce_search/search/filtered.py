"""V2 filtered source retrieval (Milestone 7): fp-1 hard filters inside the lexical and dense SQL.

Each active `FilterSpec` field adds one constant fragment from `FILTER_FRAGMENTS` to the WHERE
clause of a statement that is otherwise the existing V0 lexical / dense statement, so filtering
happens before ORDER BY and LIMIT. Filter values are always named bind parameters: the statement
text depends only on *which* fields are set, never on their values. Prices are bound as the
`Decimal` values of the spec.

Semantics (fp-1): category and brand are exact equality; RAM and storage are exact equality on a
laptop or phone spec row (correlated EXISTS, so a product is never duplicated); storage type and
interface exist only on laptop spec rows; prices are inclusive bounds. A missing spec row or a
NULL spec value never satisfies a spec filter.

Nothing is relaxed: there is no Python post-filter, no retry without filters and no fallback. An
empty list is a result. Ordering, tie-breaks and scores are those of `lexical_search` /
`dense_search`, and ranks are positional within the filtered list. Dense retrieval keeps the
current-embedding predicates unchanged, so stale or missing embeddings only shorten the list.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from types import MappingProxyType

from sqlalchemy import TextClause, text
from sqlalchemy.orm import Session

from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.filtering import FilterField, FilterSpec
from ecommerce_search.search import dense, lexical
from ecommerce_search.search.dense import DenseHit, DenseResult
from ecommerce_search.search.documents import FTS_CONFIG
from ecommerce_search.search.hybrid import SNAPSHOT_SQL
from ecommerce_search.search.lexical import LexicalHit, LexicalResult


@dataclass(frozen=True)
class FilterFragment:
    sql: str  # constant SQL over the products alias `p`; never contains a value
    param: str  # the named bind parameter the fragment uses


_LAPTOP = "EXISTS (SELECT 1 FROM laptop_specs s WHERE s.product_id = p.product_id AND "
_PHONE = "EXISTS (SELECT 1 FROM phone_specs s WHERE s.product_id = p.product_id AND "

# The one closed table shared by lexical and dense statements, in fp-1 field order. Every
# fragment is a constant string over the products alias `p` with one named bind parameter.
FILTER_FRAGMENTS: Mapping[FilterField, FilterFragment] = MappingProxyType(
    {
        FilterField.CATEGORY: FilterFragment("p.category = :f_category", "f_category"),
        FilterField.BRAND: FilterFragment("p.brand = :f_brand", "f_brand"),
        FilterField.RAM_GB: FilterFragment(
            "(" + _LAPTOP + "s.ram_gb = :f_ram_gb) OR " + _PHONE + "s.ram_gb = :f_ram_gb))",
            "f_ram_gb",
        ),
        FilterField.STORAGE_GB: FilterFragment(
            "("
            + _LAPTOP
            + "s.storage_gb = :f_storage_gb) OR "
            + _PHONE
            + "s.storage_gb = :f_storage_gb))",
            "f_storage_gb",
        ),
        FilterField.STORAGE_TYPE: FilterFragment(
            _LAPTOP + "s.storage_type = :f_storage_type)", "f_storage_type"
        ),
        FilterField.STORAGE_INTERFACE: FilterFragment(
            _LAPTOP + "s.storage_interface = :f_storage_interface)", "f_storage_interface"
        ),
        FilterField.MIN_PRICE: FilterFragment("p.price >= :f_min_price", "f_min_price"),
        FilterField.MAX_PRICE: FilterFragment("p.price <= :f_max_price", "f_max_price"),
    }
)

# The existing statements (`lexical.RETRIEVAL_SQL`, `dense.RETRIEVAL_SQL`) split at the end of
# their WHERE clause; a unit test keeps head + tail equal to them.
_LEXICAL_HEAD = """
    SELECT p.product_id, p.title, p.brand, p.category, p.subcategory, p.description,
           p.price, p.currency, p.rating, p.review_count, p.availability,
           ts_rank(CAST(:weights AS real[]), d.search_vector,
                   plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text)),
                   CAST(:normalization AS integer)) AS lexical_score
    FROM product_search_documents d
    JOIN products p ON p.product_id = d.product_id
    WHERE d.search_vector @@ plainto_tsquery(CAST(:cfg AS regconfig), CAST(:q AS text))"""
_LEXICAL_TAIL = """
    ORDER BY lexical_score DESC, p.product_id ASC
    LIMIT :limit
    """
_DENSE_HEAD = """
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
      AND e.source_content_sha256 = p.content_sha256"""
_DENSE_TAIL = """
    ORDER BY distance ASC, p.product_id ASC
    LIMIT :limit
    """


def active_fields(filters: FilterSpec) -> tuple[FilterField, ...]:
    """The set fields of `filters`, in fp-1 order."""
    if not isinstance(filters, FilterSpec):
        raise TypeError("filters must be a FilterSpec")
    return tuple(field for field in FilterField if getattr(filters, field.value) is not None)


def filter_params(filters: FilterSpec) -> dict:
    """Bind values for the active fragments, in fp-1 order.

    Prices stay `Decimal` and capacities `int`. Enum members are bound as their plain string
    value (a driver enum adapter could otherwise send the member name)."""
    params = {}
    for field in active_fields(filters):
        value = getattr(filters, field.value)
        params[FILTER_FRAGMENTS[field].param] = value.value if isinstance(value, StrEnum) else value
    return params


def filter_clause(fields: Sequence[FilterField]) -> str:
    return "".join(f"\n      AND {FILTER_FRAGMENTS[field].sql}" for field in fields)


@cache
def lexical_statement(fields: tuple[FilterField, ...]) -> TextClause:
    return text(_LEXICAL_HEAD + filter_clause(fields) + _LEXICAL_TAIL)


@cache
def dense_statement(fields: tuple[FilterField, ...]) -> TextClause:
    return text(_DENSE_HEAD + filter_clause(fields) + _DENSE_TAIL)


def filtered_lexical_search(
    session: Session, query: str, filters: FilterSpec, limit: int
) -> LexicalResult:
    """Up to `limit` lexical hits that satisfy `filters`. Database errors propagate."""
    start = time.perf_counter()
    fields = active_fields(filters)
    tsquery, node_count = session.execute(
        lexical.TSQUERY_SQL, {"cfg": FTS_CONFIG, "q": query}
    ).one()
    hits: list[LexicalHit] = []
    if node_count > 0:  # punctuation-only text yields an empty query: no rows, no error
        params = lexical.retrieval_params(query, limit) | filter_params(filters)
        rows = session.execute(lexical_statement(fields), params).all()
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


def filtered_dense_search(
    session: Session,
    query_vector: Sequence[float],
    spec: EmbeddingModelSpec,
    filters: FilterSpec,
    limit: int,
) -> DenseResult:
    """Up to `limit` nearest current products that satisfy `filters`. Errors propagate."""
    start = time.perf_counter()
    fields = active_fields(filters)
    params = dense.retrieval_params(query_vector, spec, limit) | filter_params(filters)
    rows = session.execute(dense_statement(fields), params).all()
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


def read_filtered_sources(
    session: Session,
    query: str,
    query_vector: Sequence[float],
    spec: EmbeddingModelSpec,
    filters: FilterSpec,
    lexical_k: int,
    dense_k: int,
) -> tuple[LexicalResult, DenseResult]:
    """Run both filtered reads in one REPEATABLE READ READ ONLY transaction (as `read_sources`).

    The transaction commits on success and rolls back on any error; either way the session
    releases its connection before this returns. Database errors propagate."""
    if session.in_transaction():
        raise RuntimeError("read_filtered_sources needs a session without an open transaction")
    active_fields(filters)  # reject a non-FilterSpec before any database work
    with session.begin():
        session.execute(SNAPSHOT_SQL)  # must be the first statement of the transaction
        lexical_result = filtered_lexical_search(session, query, filters, lexical_k)
        dense_result = filtered_dense_search(session, query_vector, spec, filters, dense_k)
    return lexical_result, dense_result
