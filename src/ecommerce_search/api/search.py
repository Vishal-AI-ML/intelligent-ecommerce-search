"""`GET /search` and `POST /search` (V0 lexical). Both share one validation and retrieval path.

Database failures are translated here, in the route, to a fixed generic 503. No application-wide
exception handler is installed, so `/health` keeps its own 200/503 contract.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.api.dependencies import get_db_session, get_settings_dep
from ecommerce_search.api.schemas import (
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchUnavailable,
)
from ecommerce_search.config import Settings
from ecommerce_search.search.documents import DOCUMENT_VERSION
from ecommerce_search.search.lexical import lexical_search
from ecommerce_search.search.query import SEARCH_VERSION, QueryError, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE_DETAIL = "search unavailable"
RESPONSES = {
    422: {"description": "Invalid query text or top_k (standard validation error body)."},
    503: {
        "model": SearchUnavailable,
        "description": "The search database is unavailable or the search index is not built. "
        "The body is fixed and carries no internal details.",
    },
}
DESCRIPTION = (
    "V0 lexical search: PostgreSQL full-text search (`simple` configuration), every term must "
    "match (strict AND). Results are ordered by `lexical_score` descending, then `product_id` "
    "ascending. The score is query-relative and uncalibrated. Queries are not interpreted: "
    "RAM, price, brand and storage are plain words, and filler words can produce zero results. "
    "The index is a derived table: if it is missing or stale, results may be incomplete until "
    "`python -m ecommerce_search.search reindex` succeeds."
)


def _invalid(location: tuple[str, str], message: str) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": location, "msg": message}])


def _search(
    session: Session,
    settings: Settings,
    raw_query: str,
    top_k: int | None,
    query_location: tuple[str, str],
    top_k_location: tuple[str, str],
) -> SearchResponse:
    started = time.perf_counter()
    try:
        query = normalize_query(raw_query, settings.search_max_query_length)
    except QueryError as exc:
        raise _invalid(query_location, str(exc)) from None
    limit = settings.search_default_top_k if top_k is None else top_k
    if not 1 <= limit <= settings.search_lexical_k:
        raise _invalid(top_k_location, f"top_k must be between 1 and {settings.search_lexical_k}")
    try:
        result = lexical_search(session, query, limit)
    except SQLAlchemyError as exc:
        # Log the exception type only (never its message or SQL). A missing table gets an
        # operator hint in the server log; the response body stays fixed and generic.
        if getattr(getattr(exc, "orig", None), "sqlstate", None) == "42P01":
            logger.error(
                "search failed: %s (the search index table is missing: run "
                "`uv run alembic upgrade head`, then `python -m ecommerce_search.search reindex`)",
                type(exc).__name__,
            )
        else:
            logger.error("search failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE_DETAIL) from None
    results = [
        SearchResult(
            rank=hit.rank,
            lexical_score=hit.lexical_score,
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
    return SearchResponse(
        query=query,
        search_version=SEARCH_VERSION,
        document_version=DOCUMENT_VERSION,
        top_k=limit,
        result_count=len(results),
        results=results,
        tsquery=result.tsquery,
        applied_filters=[],
        latency_ms=SearchLatency(lexical_ms=result.lexical_ms, total_ms=total_ms),
    )


@router.get(
    "/search",
    response_model=SearchResponse,
    responses=RESPONSES,
    summary="Lexical search (V0)",
    description=DESCRIPTION,
)
def search_get(
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    q: Annotated[str, Query(description="Search text.")],
    top_k: Annotated[
        int | None,
        Query(ge=1, description="Maximum results. Default and upper bound come from settings."),
    ] = None,
) -> SearchResponse:
    return _search(session, settings, q, top_k, ("query", "q"), ("query", "top_k"))


@router.post(
    "/search",
    response_model=SearchResponse,
    responses=RESPONSES,
    summary="Lexical search (V0)",
    description=DESCRIPTION,
)
def search_post(
    body: SearchRequest,
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> SearchResponse:
    return _search(session, settings, body.query, body.top_k, ("body", "query"), ("body", "top_k"))
