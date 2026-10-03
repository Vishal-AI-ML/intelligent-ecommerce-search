"""Typed, frozen query-understanding models (Milestone 6).

Every model is frozen and forbids extra fields; every collection is a tuple with a fixed,
documented ordering, so the same input and versions always give byte-identical JSON.

Spans are zero-based Unicode code-point offsets into `QueryUnderstanding.raw_query` with an
exclusive end, and `span.text == raw_query[start:end]` always holds (checked here).

Numeric bounds are the authoritative catalog bounds (`docs/data-quality.md` §3.1, products.price
`Numeric(12, 2)` with `price > 0`): price `(0, 9999999999.99]` with 2 decimals, RAM 1–4096 GB,
storage 1–65536 GB.
"""

from decimal import Context, Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from ecommerce_search.catalog.taxonomy import Category, StorageType

MIN_CAPACITY_GB = 1
MAX_RAM_GB = 4096
MAX_STORAGE_GB = 65536
MAX_PRICE = Decimal("9999999999.99")

# A fixed local context: results never depend on the caller's thread-local decimal context.
DECIMAL_CONTEXT = Context(prec=28)
CENT = Decimal("0.01")


class UnderstandingField(StrEnum):
    CATEGORY = "category"
    BRAND = "brand"
    RAM_GB = "ram_gb"
    STORAGE_GB = "storage_gb"
    STORAGE_TYPE = "storage_type"
    STORAGE_INTERFACE = "storage_interface"
    MIN_PRICE = "min_price"
    MAX_PRICE = "max_price"
    SEMANTIC_INTENT = "semantic_intent"
    PRICE_PREFERENCE = "price_preference"


class MatchRule(StrEnum):
    CATEGORY_TERM = "category_term"
    BRAND_TERM = "brand_term"
    INTENT_TERM = "intent_term"
    STORAGE_TYPE_TERM = "storage_type_term"
    STORAGE_INTERFACE_TERM = "storage_interface_term"
    STORAGE_TYPE_IMPLIED_BY_INTERFACE = "storage_type_implied_by_interface"
    PRICE_PREFERENCE_TERM = "price_preference_term"
    RAM_CAPACITY = "ram_capacity"
    RAM_CAPACITY_PAIRED = "ram_capacity_paired"
    STORAGE_CAPACITY = "storage_capacity"
    STORAGE_CAPACITY_IMPLICIT_GB = "storage_capacity_implicit_gb"
    PRICE_UPPER_BOUND = "price_upper_bound"
    PRICE_LOWER_BOUND = "price_lower_bound"


class AmbiguityReason(StrEnum):
    BARE_CAPACITY = "bare_capacity"
    AMBIGUOUS_MEMORY_CAPACITY = "ambiguous_memory_capacity"
    CAPACITY_MULTIPLE_ROLES = "capacity_multiple_roles"
    PRICE_WITHOUT_BOUND = "price_without_bound"
    UNSUPPORTED_GROUPED_NUMBER = "unsupported_grouped_number"
    UNSUPPORTED_RANGE = "unsupported_range"
    UNSUPPORTED_NUMERIC_COMPOUND = "unsupported_numeric_compound"
    OUT_OF_RANGE_VALUE = "out_of_range_value"


class ConflictReason(StrEnum):
    REPEATED_DIFFERENT_VALUES = "repeated_different_values"
    MIN_PRICE_EXCEEDS_MAX_PRICE = "min_price_exceeds_max_price"
    DEPENDENT_ON_CONFLICTED_STORAGE_TYPE = "dependent_on_conflicted_storage_type"


class Intent(StrEnum):
    GAMING = "gaming"
    CODING = "coding"
    STUDENT = "student"
    LIGHTWEIGHT = "lightweight"
    PREMIUM = "premium"
    COMFORT = "comfort"
    TRAVEL = "travel"
    OFFICE = "office"


def _quantize_price(value: Decimal) -> Decimal:
    return value.quantize(CENT, context=DECIMAL_CONTEXT)


Price = Annotated[
    Decimal,
    Field(gt=0, le=MAX_PRICE, max_digits=12, decimal_places=2, allow_inf_nan=False),
    AfterValidator(_quantize_price),
]
RamGb = Annotated[int, Field(ge=MIN_CAPACITY_GB, le=MAX_RAM_GB)]
StorageGb = Annotated[int, Field(ge=MIN_CAPACITY_GB, le=MAX_STORAGE_GB)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceSpan(_Frozen):
    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if not 0 <= self.start < self.end:
            raise ValueError("span must satisfy 0 <= start < end")
        return self


class MatchedTerm(_Frozen):
    rule: MatchRule
    field: UnderstandingField
    value: str
    span: SourceSpan


class Ambiguity(_Frozen):
    reason: AmbiguityReason
    span: SourceSpan
    value: str | None = None


class Conflict(_Frozen):
    field: UnderstandingField
    reason: ConflictReason
    candidates: tuple[MatchedTerm, ...] = Field(min_length=1)


class QueryAttributes(_Frozen):
    storage_interface: Literal["NVME"] | None = None
    price_preference: Literal["low"] | None = None


class QueryUnderstanding(_Frozen):
    raw_query: str = Field(min_length=1)
    category: Category | None = None
    brand: str | None = None
    ram_gb: RamGb | None = None
    storage_gb: StorageGb | None = None
    storage_type: StorageType | None = None
    min_price: Price | None = None
    max_price: Price | None = None
    semantic_intent: Intent | None = None
    retrieval_strategy: None = None
    attributes: QueryAttributes = QueryAttributes()
    parser_version: Literal["qu-1"] = "qu-1"
    lexicon_version: Literal["1"] = "1"
    matched_terms: tuple[MatchedTerm, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    unresolved: tuple[SourceSpan, ...] = ()

    @model_validator(mode="after")
    def _check_span_texts(self) -> Self:
        spans = [term.span for term in self.matched_terms]
        spans += [ambiguity.span for ambiguity in self.ambiguities]
        spans += [term.span for conflict in self.conflicts for term in conflict.candidates]
        spans += list(self.unresolved)
        for span in spans:
            if span.end > len(self.raw_query) or self.raw_query[span.start : span.end] != span.text:
                raise ValueError("span text must equal raw_query[start:end]")
        return self
