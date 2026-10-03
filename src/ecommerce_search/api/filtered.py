"""`GET /search/filtered` and `POST /search/filtered` (Milestone 7, V2). One shared path.

V2 = V1 + structured filtering. Order of work: validate; run deterministic query understanding
once; translate the decision into fp-1 filters once (pure, no I/O); then, only for a query with a
letter or digit, load the model and encode the unchanged normalized query (no database
connection is used), then read both filtered sources in one REPEATABLE READ READ ONLY
transaction, which ends before fusion and response construction. Fusion is the unchanged V1
Reciprocal Rank Fusion.

The parse and the filter derivation reach retrieval only as the `FilterSpec` passed to
`read_filtered_sources`. The query text is never rewritten or stripped, results are never
post-filtered, and a search is never retried without filters: zero matching products is a 200
with no results and the applied filters reported.

Failures map to the same fixed generic 503 body as `/search/hybrid`; the server log records only
a fixed reason code (plus a fixed operator hint where one exists), never query text, SQL, paths,
credentials, exception text or a traceback. Existing routes are untouched by this module.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.api.dense import HINTS, encode_query, has_searchable_text
from ecommerce_search.api.dependencies import (
    get_db_session,
    get_decision_provider,
    get_embedder,
    get_settings_dep,
)
from ecommerce_search.api.schemas import (
    FilteredQueryUnderstandingBlock,
    FilteredSearchLatency,
    FilteredSearchRequest,
    FilteredSearchResponse,
    FilterPolicy,
    HybridFusion,
    HybridSearchResult,
    SearchUnavailable,
)
from ecommerce_search.config import Settings
from ecommerce_search.decision import DecisionProvider, understand_normalized_query
from ecommerce_search.embeddings.provider import Embedder, EmbedderUnavailable
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.filtering import FILTER_POLICY_VERSION, FilterDerivation, derive_filters
from ecommerce_search.search.dense import DISTANCE_METRIC
from ecommerce_search.search.documents import DOCUMENT_VERSION
from ecommerce_search.search.filtered import read_filtered_sources
from ecommerce_search.search.hybrid import FUSION_METHOD, RRF_K_STATUS, fuse_rrf
from ecommerce_search.search.query import QueryError, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter()

FILTERED_SEARCH_VERSION = "v2_filtered"
UNAVAILABLE_DETAIL = "search unavailable"
RESPONSES = {
    422: {"description": "Invalid query text or top_k (standard validation error body)."},
    503: {
        "model": SearchUnavailable,
        "description": "Query understanding or filter translation failed, or the embedding "
        "model, the search index or the search database is unavailable. The body is fixed and "
        "carries no internal details.",
    },
}
DESCRIPTION = (
    "V2 filtered hybrid retrieval (Milestone 7): V1 hybrid retrieval with structured filters. "
    "The query is parsed deterministically (`query_understanding`) and the filter policy "
    "(`filter_policy.version`) turns only explicit, unambiguous, non-conflicting category, "
    "brand, RAM, storage, storage type/interface and price constraints into hard filters "
    "(`applied_filters`); other parsed constraints are reported in `ignored_constraints` with a "
    "reason. Filters run inside both the lexical and the dense source query, before each "
    "source's limit; the query text itself is sent unchanged. Filtered candidates are fused "
    "with Reciprocal Rank Fusion exactly as in `/search/hybrid`. Filters are never relaxed and "
    "the search is never retried without them: when no product satisfies them, the response is "
    "a 200 with no results. Scores are uncalibrated and not probabilities. Queries without any "
    "letter or digit return no results without running the model or the database."
)


def _invalid(location: tuple[str, str], message: str) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": location, "msg": message}])


def _unavailable(reason: str) -> HTTPException:
    hint = HINTS.get(reason)
    if hint:
        logger.error("filtered search unavailable: %s (%s)", reason, hint)
    else:
        logger.error("filtered search unavailable: %s", reason)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE_DETAIL)


def _filtered_search(
    session: Session,
    settings: Settings,
    embedder: Embedder | None,
    provider: DecisionProvider,
    raw_query: str,
    top_k: int | None,
    query_location: tuple[str, str],
    top_k_location: tuple[str, str],
) -> FilteredSearchResponse:
    started = time.perf_counter()
    try:
        query = normalize_query(raw_query, settings.search_max_query_length)
    except QueryError as exc:
        raise _invalid(query_location, str(exc)) from None
    limit = settings.search_default_top_k if top_k is None else top_k
    if not 1 <= limit <= settings.search_candidate_k:
        raise _invalid(top_k_location, f"top_k must be between 1 and {settings.search_candidate_k}")
    understanding_started = time.perf_counter()
    try:
        decision = understand_normalized_query(query, provider)
    except Exception:
        raise _unavailable("query_understanding_failed") from None
    understanding_ms = round((time.perf_counter() - understanding_started) * 1000, 3)
    translation_started = time.perf_counter()
    try:
        derivation = derive_filters(decision)
        if not isinstance(derivation, FilterDerivation):
            raise TypeError
    except Exception:
        raise _unavailable("filter_policy_failed") from None
    translation_ms = round((time.perf_counter() - translation_started) * 1000, 3)
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
            lexical, dense = read_filtered_sources(
                session,
                query,
                vector,
                spec,
                derivation.spec,
                settings.search_lexical_k,
                settings.search_dense_k,
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
    return FilteredSearchResponse(
        query=query,
        search_version=FILTERED_SEARCH_VERSION,
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
        filter_policy=FilterPolicy(version=FILTER_POLICY_VERSION),
        applied_filters=list(derivation.applied_filters),
        ignored_constraints=list(derivation.ignored_constraints),
        query_understanding=FilteredQueryUnderstandingBlock(
            usage="filter_source",
            provider=decision.provider,
            provider_version=decision.provider_version,
            understanding=decision.understanding,
        ),
        latency_ms=FilteredSearchLatency(
            lexical_ms=lexical_ms,
            model_load_ms=load_ms,
            query_embedding_ms=embed_ms,
            vector_ms=vector_ms,
            rrf_ms=rrf_ms,
            query_understanding_ms=understanding_ms,
            filter_translation_ms=translation_ms,
            total_ms=total_ms,
        ),
    )


@router.get(
    "/search/filtered",
    response_model=FilteredSearchResponse,
    responses=RESPONSES,
    summary="Filtered hybrid search: fp-1 filters + lexical + dense + RRF (V2, Milestone 7)",
    description=DESCRIPTION,
)
def filtered_search_get(
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    provider: Annotated[DecisionProvider, Depends(get_decision_provider)],
    q: Annotated[str, Query(description="Search text.")],
    top_k: Annotated[
        int | None,
        Query(ge=1, description="Maximum results. Default and upper bound come from settings."),
    ] = None,
) -> FilteredSearchResponse:
    return _filtered_search(
        session, settings, embedder, provider, q, top_k, ("query", "q"), ("query", "top_k")
    )


@router.post(
    "/search/filtered",
    response_model=FilteredSearchResponse,
    responses=RESPONSES,
    summary="Filtered hybrid search: fp-1 filters + lexical + dense + RRF (V2, Milestone 7)",
    description=DESCRIPTION,
)
def filtered_search_post(
    body: FilteredSearchRequest,
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    provider: Annotated[DecisionProvider, Depends(get_decision_provider)],
) -> FilteredSearchResponse:
    return _filtered_search(
        session,
        settings,
        embedder,
        provider,
        body.query,
        body.top_k,
        ("body", "query"),
        ("body", "top_k"),
    )
