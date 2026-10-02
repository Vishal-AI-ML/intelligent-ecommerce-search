from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ComponentCheck(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float | None = None
    version: str | None = None
    detail: str | None = None


class HealthChecks(BaseModel):
    database: ComponentCheck
    pgvector: ComponentCheck


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    service: str
    version: str
    environment: str
    checks: HealthChecks


class SearchRequest(BaseModel):
    """Body of `POST /search`. Same validation as `GET /search`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search text. Whitespace is collapsed; see /docs for limits.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum results to return. Default and upper bound come from settings.",
    )


class SearchResult(BaseModel):
    rank: int = Field(description="1-based position in this response.")
    lexical_score: float = Field(
        description="PostgreSQL ts_rank value. Query-relative and uncalibrated: not a "
        "probability, and comparable only within one query."
    )
    product_id: str
    title: str
    brand: str
    category: str
    subcategory: str | None
    description: str | None
    price: Decimal = Field(description="Serialized as a decimal string (no float rounding).")
    currency: str
    rating: float | None
    review_count: int | None
    availability: str


class SearchLatency(BaseModel):
    lexical_ms: float = Field(description="Time in the SQL queries and row fetch.")
    total_ms: float = Field(
        description="Application processing from handler entry through construction of the "
        "result models. Excludes request validation and framework response serialization."
    )


class SearchResponse(BaseModel):
    query: str = Field(description="The query after whitespace normalization.")
    search_version: Literal["v0_lexical"]
    document_version: str
    top_k: int = Field(description="The requested (or default) maximum number of results.")
    result_count: int = Field(
        description="Number of results returned in this response (not the total number of "
        "database matches)."
    )
    results: list[SearchResult]
    tsquery: str = Field(description="PostgreSQL tsquery text generated from the query.")
    applied_filters: list[str] = Field(description="Always empty in V0: no filters exist yet.")
    latency_ms: SearchLatency


class SearchUnavailable(BaseModel):
    detail: Literal["search unavailable"]


# ---- dense search (Milestone 4) ----------------------------------------------------------------


class DenseSearchRequest(BaseModel):
    """Body of `POST /search/dense`. Same validation as `GET /search/dense`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search text. Whitespace is collapsed; see /docs for limits.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum results to return. Default and upper bound come from settings.",
    )


class DenseSearchResult(BaseModel):
    rank: int = Field(description="1-based position in this response.")
    dense_score: float = Field(
        description="Cosine similarity between the query and product embeddings (1 - cosine "
        "distance, range -1..1). Uncalibrated: not a probability, and comparable only within one "
        "query and one model."
    )
    product_id: str
    title: str
    brand: str
    category: str
    subcategory: str | None
    description: str | None
    price: Decimal = Field(description="Serialized as a decimal string (no float rounding).")
    currency: str
    rating: float | None
    review_count: int | None
    availability: str


class DenseSearchLatency(BaseModel):
    model_load_ms: float | None = Field(
        description="Time to load the embedding model during this request; null when it was "
        "already loaded or not needed."
    )
    query_embedding_ms: float | None = Field(
        description="Time to check the query's token length and encode it; null when no "
        "encoding ran."
    )
    vector_ms: float | None = Field(
        description="Time in the vector SQL query and row fetch; null when it did not run."
    )
    total_ms: float = Field(
        description="Application processing from handler entry through construction of the "
        "result models. Excludes request validation and framework response serialization."
    )


class DenseSearchResponse(BaseModel):
    query: str = Field(description="The query after whitespace normalization.")
    search_version: Literal["dense_only"] = Field(
        description="Dense-only retrieval (Milestone 4). Not a V-numbered search version."
    )
    embedding_model_id: str
    embedding_model_revision: str = Field(description="Immutable model commit hash.")
    embedding_dimension: int
    embedding_text_version: str
    distance_metric: Literal["cosine"]
    top_k: int = Field(description="The requested (or default) maximum number of results.")
    result_count: int = Field(
        description="Number of results returned in this response (not the number of products "
        "with embeddings)."
    )
    results: list[DenseSearchResult]
    applied_filters: list[str] = Field(description="Always empty: no filters exist yet.")
    latency_ms: DenseSearchLatency


# ---- hybrid search (Milestone 5) ---------------------------------------------------------------


class HybridSearchRequest(BaseModel):
    """Body of `POST /search/hybrid`. Same validation as `GET /search/hybrid`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search text. Whitespace is collapsed; see /docs for limits.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum results to return. Default and upper bound come from settings.",
    )


class HybridFusion(BaseModel):
    method: Literal["rrf"] = Field(description="Reciprocal Rank Fusion with equal source weights.")
    rrf_k: int = Field(description="RRF constant: each source adds 1 / (rrf_k + source rank).")
    rrf_k_status: Literal["candidate_pending_selection", "provisional"] = Field(
        description="`candidate_pending_selection`: the value is not yet selected by any "
        "evidence. `provisional`: selected on non-authoritative smoke queries, to be re-decided "
        "on the Golden Dataset."
    )
    lexical_k: int = Field(description="Lexical candidates retrieved before fusion.")
    dense_k: int = Field(description="Dense candidates retrieved before fusion.")
    candidate_k: int = Field(description="Maximum fused candidates kept after fusion.")


class HybridSearchResult(BaseModel):
    rank: int = Field(description="1-based position in this response.")
    rrf_score: float = Field(
        description="Reciprocal Rank Fusion score (sum of 1 / (rrf_k + source rank)). Ordering "
        "uses the exact value; ties are broken by product_id ascending. A rank-fusion value, not "
        "a relevance probability."
    )
    lexical_rank: int | None = Field(
        description="1-based position in the lexical candidate list; null when not a lexical "
        "candidate."
    )
    lexical_score: float | None = Field(
        description="PostgreSQL ts_rank value (query-relative, uncalibrated); null when absent."
    )
    dense_rank: int | None = Field(
        description="1-based position in the dense candidate list; null when not a dense candidate."
    )
    dense_score: float | None = Field(
        description="Cosine similarity (uncalibrated, not a probability); null when absent."
    )
    product_id: str
    title: str
    brand: str
    category: str
    subcategory: str | None
    description: str | None
    price: Decimal = Field(description="Serialized as a decimal string (no float rounding).")
    currency: str
    rating: float | None
    review_count: int | None
    availability: str


class HybridSearchLatency(BaseModel):
    lexical_ms: float | None = Field(
        description="Time in the lexical SQL queries and row fetch; null when they did not run."
    )
    model_load_ms: float | None = Field(
        description="Time to load the embedding model during this request; null when it was "
        "already loaded or not needed."
    )
    query_embedding_ms: float | None = Field(
        description="Time to check the query's token length and encode it; null when no "
        "encoding ran."
    )
    vector_ms: float | None = Field(
        description="Time in the vector SQL query and row fetch; null when it did not run."
    )
    rrf_ms: float | None = Field(
        description="Time to fuse the candidate lists; null when fusion did not run."
    )
    total_ms: float = Field(
        description="Application processing from handler entry through construction of the "
        "result models. Excludes request validation and framework response serialization."
    )


class HybridSearchResponse(BaseModel):
    query: str = Field(description="The query after whitespace normalization.")
    search_version: Literal["v1_hybrid"]
    document_version: str
    tsquery: str | None = Field(
        description="PostgreSQL tsquery text generated from the query; null when no SQL ran "
        "(the query has no letter or digit)."
    )
    embedding_model_id: str
    embedding_model_revision: str = Field(description="Immutable model commit hash.")
    embedding_dimension: int
    embedding_text_version: str
    distance_metric: Literal["cosine"]
    fusion: HybridFusion
    dense_status: Literal["used", "skipped_no_searchable_text"]
    lexical_hit_count: int = Field(description="Lexical candidates retrieved (at most lexical_k).")
    dense_hit_count: int = Field(
        description="Dense candidates retrieved (at most dense_k). Only embeddings current for "
        "the active model and the product's current content take part, so this can be lower "
        "while embeddings are missing or stale."
    )
    overlap_count: int = Field(description="Products present in both candidate lists.")
    fused_count: int = Field(description="Unique products in the union of both lists.")
    candidate_count: int = Field(description="Fused candidates kept (at most candidate_k).")
    top_k: int = Field(description="The requested (or default) maximum number of results.")
    result_count: int = Field(description="Number of results returned in this response.")
    results: list[HybridSearchResult]
    applied_filters: list[str] = Field(description="Always empty: no filters exist yet.")
    latency_ms: HybridSearchLatency
