"""`GET /search/dense` and `POST /search/dense` (Milestone 4). One shared validation path.

Failures are translated here to the same fixed generic 503 body as `/search`. The server log
records only a fixed reason code and an operator hint, never paths, SQL, hosts, credentials or
exception text. `/search` and `/health` are untouched by this module.
"""

import logging
import time
import unicodedata
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.api.dependencies import get_db_session, get_embedder, get_settings_dep
from ecommerce_search.api.schemas import (
    DenseSearchLatency,
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    SearchUnavailable,
)
from ecommerce_search.config import Settings
from ecommerce_search.embeddings.provider import Embedder, EmbedderUnavailable
from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.search.dense import DENSE_SEARCH_VERSION, DISTANCE_METRIC, dense_search
from ecommerce_search.search.query import QueryError, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE_DETAIL = "search unavailable"
# Fixed operator hints, logged server-side only (never in a response).
HINTS = {
    "snapshot_missing": "the embedding model snapshot is not present: run "
    "`python -m ecommerce_search.search model-fetch` (see ADR-006)",
    "table_missing": "the embedding table is missing: run `uv run alembic upgrade head`, then "
    "`python -m ecommerce_search.search embed`",
}
RESPONSES = {
    422: {"description": "Invalid query text or top_k (standard validation error body)."},
    503: {
        "model": SearchUnavailable,
        "description": "The embedding model or the search database is unavailable. The body is "
        "fixed and carries no internal details.",
    },
}
DESCRIPTION = (
    "Dense-only semantic retrieval (Milestone 4): the query is embedded with the local, pinned "
    "embedding model and compared with product embeddings by exact cosine distance in pgvector. "
    "Results are ordered by `dense_score` (cosine similarity) descending, then `product_id` "
    "ascending. The score is uncalibrated and not a probability. There is no similarity "
    "threshold: every query returns up to `top_k` nearest products, even a nonsense query. Only "
    "embeddings built from a product's current content with the active model are used, so "
    "results may be incomplete until `python -m ecommerce_search.search embed` succeeds. "
    "Queries without any letter or digit return no results without running the model."
)


def _invalid(location: tuple[str, str], message: str) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": location, "msg": message}])


def _unavailable(reason: str) -> HTTPException:
    hint = HINTS.get(reason)
    if hint:
        logger.error("dense search unavailable: %s (%s)", reason, hint)
    else:
        logger.error("dense search unavailable: %s", reason)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE_DETAIL)


def has_searchable_text(query: str) -> bool:
    """True when the query contains at least one letter or number (any script)."""
    return any(unicodedata.category(char)[0] in "LN" for char in query)


def encode_query(
    embedder: Embedder,
    spec: EmbeddingModelSpec,
    query: str,
    query_location: tuple[str, str],
) -> tuple[list[float], float | None, float]:
    """Load the model, check the query token limit and encode: `(vector, load_ms, embed_ms)`.

    Raises a 422 validation error for a query over the token limit and lets
    `EmbedderUnavailable` propagate. Touches no database session (shared with /search/hybrid)."""
    load_ms = embedder.load()
    embed_started = time.perf_counter()
    (tokens,) = embedder.count_tokens([query], "query")
    if tokens > spec.max_seq_length:
        raise _invalid(query_location, "query is too long for the embedding model")
    vector = embedder.embed_query(query)
    embed_ms = round((time.perf_counter() - embed_started) * 1000, 3)
    return vector, load_ms, embed_ms


def _dense_search(
    session: Session,
    settings: Settings,
    embedder: Embedder | None,
    raw_query: str,
    top_k: int | None,
    query_location: tuple[str, str],
    top_k_location: tuple[str, str],
) -> DenseSearchResponse:
    started = time.perf_counter()
    try:
        query = normalize_query(raw_query, settings.search_max_query_length)
    except QueryError as exc:
        raise _invalid(query_location, str(exc)) from None
    limit = settings.search_default_top_k if top_k is None else top_k
    if not 1 <= limit <= settings.search_dense_k:
        raise _invalid(top_k_location, f"top_k must be between 1 and {settings.search_dense_k}")
    spec = settings.embedding_spec()
    load_ms = embed_ms = vector_ms = None
    results: list[DenseSearchResult] = []
    if has_searchable_text(query):
        if embedder is None:
            raise _unavailable("snapshot_missing")
        try:
            vector, load_ms, embed_ms = encode_query(embedder, spec, query, query_location)
        except EmbedderUnavailable as exc:
            raise _unavailable(exc.reason) from None
        try:
            result = dense_search(session, vector, spec, limit)
        except SQLAlchemyError as exc:
            missing = getattr(getattr(exc, "orig", None), "sqlstate", None) == "42P01"
            raise _unavailable("table_missing" if missing else "database_error") from None
        vector_ms = result.vector_ms
        results = [
            DenseSearchResult(
                rank=hit.rank,
                dense_score=hit.dense_score,
                product_id=hit.product_id,
                title=hit.title,
                brand=hit.brand,
                category=hit.category,
                subcategory=hit.subcategory,
                description=hit.description,
                price=hit.price,
                currency=hit.currency,
                rating=None if hit.rating is None else float(hit.rating),
                review_count=hit.review_count,
                availability=hit.availability,
            )
            for hit in result.hits
        ]
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    return DenseSearchResponse(
        query=query,
        search_version=DENSE_SEARCH_VERSION,
        embedding_model_id=spec.model_id,
        embedding_model_revision=spec.revision,
        embedding_dimension=spec.dimension,
        embedding_text_version=EMBEDDING_TEXT_VERSION,
        distance_metric=DISTANCE_METRIC,
        top_k=limit,
        result_count=len(results),
        results=results,
        applied_filters=[],
        latency_ms=DenseSearchLatency(
            model_load_ms=load_ms,
            query_embedding_ms=embed_ms,
            vector_ms=vector_ms,
            total_ms=total_ms,
        ),
    )


@router.get(
    "/search/dense",
    response_model=DenseSearchResponse,
    responses=RESPONSES,
    summary="Dense semantic search (Milestone 4)",
    description=DESCRIPTION,
)
def dense_search_get(
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    q: Annotated[str, Query(description="Search text.")],
    top_k: Annotated[
        int | None,
        Query(ge=1, description="Maximum results. Default and upper bound come from settings."),
    ] = None,
) -> DenseSearchResponse:
    return _dense_search(session, settings, embedder, q, top_k, ("query", "q"), ("query", "top_k"))


@router.post(
    "/search/dense",
    response_model=DenseSearchResponse,
    responses=RESPONSES,
    summary="Dense semantic search (Milestone 4)",
    description=DESCRIPTION,
)
def dense_search_post(
    body: DenseSearchRequest,
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
) -> DenseSearchResponse:
    return _dense_search(
        session, settings, embedder, body.query, body.top_k, ("body", "query"), ("body", "top_k")
    )
