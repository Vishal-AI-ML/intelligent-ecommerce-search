"""`GET /search/hybrid` and `POST /search/hybrid` (Milestone 5, V1). One shared validation path.

Order of work: validate, then load the model and encode the query (no database connection is
used), then read both sources in one REPEATABLE READ READ ONLY transaction, which ends before
fusion and response construction. Failures map to the same fixed generic 503 body as
`/search/dense`; the server log records only a fixed reason code and an operator hint. `/search`,
`/search/dense` and `/health` are untouched by this module.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.api.dense import HINTS, encode_query, has_searchable_text
from ecommerce_search.api.dependencies import get_db_session, get_embedder, get_settings_dep
from ecommerce_search.api.schemas import (
    HybridFusion,
    HybridSearchLatency,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    SearchUnavailable,
)
from ecommerce_search.config import Settings
from ecommerce_search.embeddings.provider import Embedder, EmbedderUnavailable
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.search.dense import DISTANCE_METRIC
from ecommerce_search.search.documents import DOCUMENT_VERSION
from ecommerce_search.search.hybrid import (
    FUSION_METHOD,
    HYBRID_SEARCH_VERSION,
    RRF_K_STATUS,
    fuse_rrf,
    read_sources,
)
from ecommerce_search.search.query import QueryError, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE_DETAIL = "search unavailable"
RESPONSES = {
    422: {"description": "Invalid query text or top_k (standard validation error body)."},
    503: {
        "model": SearchUnavailable,
        "description": "The embedding model, the search index or the search database is "
        "unavailable. The body is fixed and carries no internal details.",
    },
}
DESCRIPTION = (
    "V1 hybrid retrieval (Milestone 5): lexical full-text candidates (as `/search`) and dense "
    "embedding candidates (as `/search/dense`) are fused with Reciprocal Rank Fusion: each "
    "source adds 1 / (rrf_k + source rank), with equal weights. Results are ordered by "
    "`rrf_score` descending, then `product_id` ascending, and keep both source ranks and scores "
    "(null when a product is absent from that source). `rrf_k` is a candidate value pending "
    "selection (see `fusion.rrf_k_status`). Scores are uncalibrated and not probabilities. Dense "
    "retrieval has no similarity threshold, so even a nonsense query can return results. Only "
    "embeddings current for the active model and product content are used, so dense candidates "
    "may be fewer while embeddings are missing or stale (`dense_hit_count`). Queries without any "
    "letter or digit return no results without running the model or the database."
)


def _invalid(location: tuple[str, str], message: str) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": location, "msg": message}])


def _unavailable(reason: str) -> HTTPException:
    hint = HINTS.get(reason)
    if hint:
        logger.error("hybrid search unavailable: %s (%s)", reason, hint)
    else:
        logger.error("hybrid search unavailable: %s", reason)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE_DETAIL)


def _hybrid_search(
    session: Session,
    settings: Settings,
    embedder: Embedder | None,
    raw_query: str,
    top_k: int | None,
    query_location: tuple[str, str],
    top_k_location: tuple[str, str],
) -> HybridSearchResponse:
    started = time.perf_counter()
    try:
        query = normalize_query(raw_query, settings.search_max_query_length)
    except QueryError as exc:
        raise _invalid(query_location, str(exc)) from None
    limit = settings.search_default_top_k if top_k is None else top_k
    if not 1 <= limit <= settings.search_candidate_k:
        raise _invalid(top_k_location, f"top_k must be between 1 and {settings.search_candidate_k}")
    spec = settings.embedding_spec()
    searchable = has_searchable_text(query)
    tsquery = lexical_ms = load_ms = embed_ms = vector_ms = rrf_ms = None
    lexical_hits: list = []
    dense_hits: list = []
    fused: list = []
    if searchable:
        if embedder is None:
            raise _unavailable("snapshot_missing")
        # Model work first: no database connection is checked out until it is done.
        try:
            vector, load_ms, embed_ms = encode_query(embedder, spec, query, query_location)
        except EmbedderUnavailable as exc:
            raise _unavailable(exc.reason) from None
        try:
            lexical, dense = read_sources(
                session, query, vector, spec, settings.search_lexical_k, settings.search_dense_k
            )
        except SQLAlchemyError as exc:
            missing = getattr(getattr(exc, "orig", None), "sqlstate", None) == "42P01"
            raise _unavailable("table_missing" if missing else "database_error") from None
        tsquery, lexical_ms, vector_ms = lexical.tsquery, lexical.lexical_ms, dense.vector_ms
        lexical_hits, dense_hits = lexical.hits, dense.hits
        rrf_started = time.perf_counter()
        fused = fuse_rrf(
            lexical_hits, dense_hits, settings.search_rrf_k, settings.search_candidate_k
        )
        rrf_ms = round((time.perf_counter() - rrf_started) * 1000, 3)
    rows = {hit.product_id: hit for hit in dense_hits} | {
        hit.product_id: hit for hit in lexical_hits
    }
    results = []
    for position, candidate in enumerate(fused[:limit], start=1):
        row = rows[candidate.product_id]  # lexical display fields win; same snapshot either way
        results.append(
            HybridSearchResult(
                rank=position,
                rrf_score=float(candidate.rrf_score),
                lexical_rank=candidate.lexical_rank,
                lexical_score=candidate.lexical_score,
                dense_rank=candidate.dense_rank,
                dense_score=candidate.dense_score,
                product_id=row.product_id,
                title=row.title,
                brand=row.brand,
                category=row.category,
                subcategory=row.subcategory,
                description=row.description,
                price=row.price,
                currency=row.currency,
                rating=None if row.rating is None else float(row.rating),
                review_count=row.review_count,
                availability=row.availability,
            )
        )
    lexical_ids = {hit.product_id for hit in lexical_hits}
    dense_ids = {hit.product_id for hit in dense_hits}
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    return HybridSearchResponse(
        query=query,
        search_version=HYBRID_SEARCH_VERSION,
        document_version=DOCUMENT_VERSION,
        tsquery=tsquery,
        embedding_model_id=spec.model_id,
        embedding_model_revision=spec.revision,
        embedding_dimension=spec.dimension,
        embedding_text_version=EMBEDDING_TEXT_VERSION,
        distance_metric=DISTANCE_METRIC,
        fusion=HybridFusion(
            method=FUSION_METHOD,
            rrf_k=settings.search_rrf_k,
            rrf_k_status=RRF_K_STATUS,
            lexical_k=settings.search_lexical_k,
            dense_k=settings.search_dense_k,
            candidate_k=settings.search_candidate_k,
        ),
        dense_status="used" if searchable else "skipped_no_searchable_text",
        lexical_hit_count=len(lexical_hits),
        dense_hit_count=len(dense_hits),
        overlap_count=len(lexical_ids & dense_ids),
        fused_count=len(lexical_ids | dense_ids),
        candidate_count=len(fused),
        top_k=limit,
        result_count=len(results),
        results=results,
        applied_filters=[],
        latency_ms=HybridSearchLatency(
            lexical_ms=lexical_ms,
            model_load_ms=load_ms,
            query_embedding_ms=embed_ms,
            vector_ms=vector_ms,
            rrf_ms=rrf_ms,
            total_ms=total_ms,
        ),
    )


@router.get(
    "/search/hybrid",
    response_model=HybridSearchResponse,
    responses=RESPONSES,
    summary="Hybrid search: lexical + dense + RRF (V1, Milestone 5)",
    description=DESCRIPTION,
)
def hybrid_search_get(
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    q: Annotated[str, Query(description="Search text.")],
    top_k: Annotated[
        int | None,
        Query(ge=1, description="Maximum results. Default and upper bound come from settings."),
    ] = None,
) -> HybridSearchResponse:
    return _hybrid_search(session, settings, embedder, q, top_k, ("query", "q"), ("query", "top_k"))


@router.post(
    "/search/hybrid",
    response_model=HybridSearchResponse,
    responses=RESPONSES,
    summary="Hybrid search: lexical + dense + RRF (V1, Milestone 5)",
    description=DESCRIPTION,
)
def hybrid_search_post(
    body: HybridSearchRequest,
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
) -> HybridSearchResponse:
    return _hybrid_search(
        session, settings, embedder, body.query, body.top_k, ("body", "query"), ("body", "top_k")
    )
