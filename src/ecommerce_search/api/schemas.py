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
