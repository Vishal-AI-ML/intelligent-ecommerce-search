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
