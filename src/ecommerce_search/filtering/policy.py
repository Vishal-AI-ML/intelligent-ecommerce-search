"""Filter policy `fp-1` (Milestone 7): which parsed constraints become hard filters.

`derive_filters` is pure. It turns a deterministic `DecisionResult` into a closed, frozen
`FilterSpec` plus the filters it applies and the parsed constraints it deliberately ignores. It
uses no I/O, settings, database, model, network, clock, randomness or logging, and the same input
always gives byte-identical JSON.

Rules (`docs/spec.md` §7; approved M7 decisions B2–B4, I4, I5):

* Only the deterministic provider at `qu-1`/lexicon `1` may create filters. Model-derived
  decisions cannot become hard filters before Milestone 12.
* A field becomes a filter only when it is non-null **and** a matched term carries that field
  with the same canonical value. A non-null field without such evidence is a contract violation
  (`FilterPolicyError`), never applied silently.
* A field named by any conflict is not applied (`conflict`). A conflicted `storage_type` also
  blocks the NVME interface that depends on it.
* Family-scoped ambiguity suppression: a capacity ambiguity blocks both `ram_gb` and
  `storage_gb`; a price ambiguity blocks both `min_price` and `max_price`; the unsupported-number
  and out-of-range reasons block both families (`ambiguous_family`). Category, brand and the
  storage-type family are unaffected by numeric ambiguities.
* `semantic_intent` and `price_preference` are never filters (`informational_only`).
* RAM and storage are exact equality; prices are inclusive bounds; `ssd` filters only the medium
  (so NVMe and SATA SSDs both match), `nvme` adds `storage_interface = NVME`.
* Brand and category are applied literally even when the brand dictionary finds the pair unusual.
* Nothing is ever relaxed: an empty result is a result.

Any change to these rules must bump `FILTER_POLICY_VERSION`.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Strict,
    ValidationError,
    model_validator,
)

from ecommerce_search.catalog.taxonomy import (
    BRAND_CATEGORIES,
    Category,
    StorageInterface,
    StorageType,
)
from ecommerce_search.decision.provider import DecisionResult
from ecommerce_search.query_understanding.models import (
    CENT,
    DECIMAL_CONTEXT,
    AmbiguityReason,
    Price,
    QueryUnderstanding,
    RamGb,
    StorageGb,
    UnderstandingField,
)

FILTER_POLICY_VERSION: Literal["fp-1"] = "fp-1"

# The only decision source allowed to create hard filters under fp-1.
SUPPORTED_PROVIDER = "deterministic"
SUPPORTED_PROVIDER_VERSION = "qu-1"
SUPPORTED_PARSER_VERSION = "qu-1"
SUPPORTED_LEXICON_VERSION = "1"


class FilterField(StrEnum):
    """Filterable fields, in the fixed order used for applied filters."""

    CATEGORY = "category"
    BRAND = "brand"
    RAM_GB = "ram_gb"
    STORAGE_GB = "storage_gb"
    STORAGE_TYPE = "storage_type"
    STORAGE_INTERFACE = "storage_interface"
    MIN_PRICE = "min_price"
    MAX_PRICE = "max_price"


class FilterOperator(StrEnum):
    EQ = "eq"
    GTE = "gte"  # inclusive lower bound
    LTE = "lte"  # inclusive upper bound


class IgnoreReason(StrEnum):
    CONFLICT = "conflict"
    AMBIGUOUS_FAMILY = "ambiguous_family"
    INFORMATIONAL_ONLY = "informational_only"


OPERATORS: dict[FilterField, FilterOperator] = {
    FilterField.CATEGORY: FilterOperator.EQ,
    FilterField.BRAND: FilterOperator.EQ,
    FilterField.RAM_GB: FilterOperator.EQ,
    FilterField.STORAGE_GB: FilterOperator.EQ,
    FilterField.STORAGE_TYPE: FilterOperator.EQ,
    FilterField.STORAGE_INTERFACE: FilterOperator.EQ,
    FilterField.MIN_PRICE: FilterOperator.GTE,
    FilterField.MAX_PRICE: FilterOperator.LTE,
}

CAPACITY_FAMILY = frozenset({FilterField.RAM_GB, FilterField.STORAGE_GB})
PRICE_FAMILY = frozenset({FilterField.MIN_PRICE, FilterField.MAX_PRICE})

# Closed, explicit sets of ambiguity reasons that suppress each numeric family (approved plan §5
# rule 4). A reason in neither set suppresses nothing; a new parser reason must be classified
# explicitly (a test lists every current reason with its families).
CAPACITY_AMBIGUITIES = frozenset(
    {
        AmbiguityReason.BARE_CAPACITY,
        AmbiguityReason.AMBIGUOUS_MEMORY_CAPACITY,
        AmbiguityReason.CAPACITY_MULTIPLE_ROLES,
        AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER,
        AmbiguityReason.UNSUPPORTED_RANGE,
        AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND,
        AmbiguityReason.OUT_OF_RANGE_VALUE,
    }
)
PRICE_AMBIGUITIES = frozenset(
    {
        AmbiguityReason.PRICE_WITHOUT_BOUND,
        AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER,
        AmbiguityReason.UNSUPPORTED_RANGE,
        AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND,
        AmbiguityReason.OUT_OF_RANGE_VALUE,
    }
)


_UNDERSTANDING_ORDER = {field: index for index, field in enumerate(UnderstandingField)}


class FilterPolicyError(Exception):
    """The decision cannot be translated safely. The message never contains query text."""

    def __init__(self) -> None:
        super().__init__("filter policy failed")


def _known_brand(value: str) -> str:
    if value not in BRAND_CATEGORIES:
        raise ValueError("brand must be a canonical brand-dictionary spelling")
    return value


KnownBrand = Annotated[str, AfterValidator(_known_brand)]
# FilterSpec is the internal boundary to SQL: numbers must already be exact `int`/`Decimal` values
# (as the parser produces them), never coerced from floats, booleans or strings. JSON input
# (`model_validate_json`) still accepts the canonical decimal strings this model emits.
StrictRamGb = Annotated[RamGb, Strict()]
StrictStorageGb = Annotated[StorageGb, Strict()]
StrictPrice = Annotated[Price, Strict()]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AppliedFilter(_Frozen):
    field: FilterField
    operator: FilterOperator
    value: str  # the M6 canonical evidence string ("laptop", "HP", "256", "SSD", "40000.00")


class IgnoredConstraint(_Frozen):
    field: UnderstandingField
    reason: IgnoreReason


class FilterSpec(_Frozen):
    """Closed set of hard filters. Absent (null) fields do not filter."""

    policy_version: Literal["fp-1"] = FILTER_POLICY_VERSION
    category: Category | None = None
    brand: KnownBrand | None = None
    ram_gb: StrictRamGb | None = None
    storage_gb: StrictStorageGb | None = None
    storage_type: StorageType | None = None
    storage_interface: StorageInterface | None = None
    min_price: StrictPrice | None = None
    max_price: StrictPrice | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        if self.storage_interface is not None and self.storage_type is not StorageType.SSD:
            raise ValueError("storage_interface requires storage_type SSD")
        return self

    def is_empty(self) -> bool:
        return all(getattr(self, field.value) is None for field in FilterField)

    def applied_filters(self) -> tuple[AppliedFilter, ...]:
        return tuple(
            AppliedFilter(field=field, operator=OPERATORS[field], value=_canonical(value))
            for field in FilterField
            if (value := getattr(self, field.value)) is not None
        )


class FilterDerivation(_Frozen):
    policy_version: Literal["fp-1"] = FILTER_POLICY_VERSION
    spec: FilterSpec
    applied_filters: tuple[AppliedFilter, ...]
    ignored_constraints: tuple[IgnoredConstraint, ...]

    @model_validator(mode="after")
    def _check_applied_matches_spec(self) -> Self:
        if self.applied_filters != self.spec.applied_filters():
            raise ValueError("applied_filters must be derived from spec")
        return self


def _canonical(value: object) -> str:
    """The M6 canonical evidence string for a field value (`docs/query-understanding-m6.md` §3)."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Decimal):
        return format(value.quantize(CENT, context=DECIMAL_CONTEXT), "f")
    return str(value)  # an int capacity or a canonical brand / interface string


def _parsed_values(understanding: QueryUnderstanding) -> dict[FilterField, object]:
    return {
        FilterField.CATEGORY: understanding.category,
        FilterField.BRAND: understanding.brand,
        FilterField.RAM_GB: understanding.ram_gb,
        FilterField.STORAGE_GB: understanding.storage_gb,
        FilterField.STORAGE_TYPE: understanding.storage_type,
        FilterField.STORAGE_INTERFACE: understanding.attributes.storage_interface,
        FilterField.MIN_PRICE: understanding.min_price,
        FilterField.MAX_PRICE: understanding.max_price,
    }


def _check_source(decision: object) -> QueryUnderstanding:
    if not isinstance(decision, DecisionResult):
        raise FilterPolicyError()
    understanding = decision.understanding
    if (
        decision.provider != SUPPORTED_PROVIDER
        or decision.provider_version != SUPPORTED_PROVIDER_VERSION
        or understanding.parser_version != SUPPORTED_PARSER_VERSION
        or understanding.lexicon_version != SUPPORTED_LEXICON_VERSION
    ):
        raise FilterPolicyError()
    return understanding


def derive_filters(decision: DecisionResult) -> FilterDerivation:
    """Translate a deterministic decision into fp-1 filters. Raises `FilterPolicyError`."""
    understanding = _check_source(decision)
    parsed = _parsed_values(understanding)

    evidence = {(term.field, term.value) for term in understanding.matched_terms}
    for field, value in parsed.items():
        if value is None:
            continue
        if (UnderstandingField(field.value), _canonical(value)) not in evidence:
            raise FilterPolicyError()

    conflicted = {conflict.field.value for conflict in understanding.conflicts}
    if (
        FilterField.STORAGE_TYPE.value in conflicted
        and parsed[FilterField.STORAGE_INTERFACE] is not None
    ):
        conflicted.add(FilterField.STORAGE_INTERFACE.value)  # the interface depends on the medium
    reasons = {ambiguity.reason for ambiguity in understanding.ambiguities}
    suppressed: set[FilterField] = set()
    if reasons & CAPACITY_AMBIGUITIES:
        suppressed |= CAPACITY_FAMILY
    if reasons & PRICE_AMBIGUITIES:
        suppressed |= PRICE_FAMILY

    kept: dict[str, object] = {}
    ignored: set[tuple[UnderstandingField, IgnoreReason]] = set()
    for field in conflicted:
        ignored.add((UnderstandingField(field), IgnoreReason.CONFLICT))
    for field, value in parsed.items():
        if value is None or field.value in conflicted:
            continue
        if field in suppressed:
            ignored.add((UnderstandingField(field.value), IgnoreReason.AMBIGUOUS_FAMILY))
            continue
        kept[field.value] = value
    if understanding.semantic_intent is not None:
        ignored.add((UnderstandingField.SEMANTIC_INTENT, IgnoreReason.INFORMATIONAL_ONLY))
    if understanding.attributes.price_preference is not None:
        ignored.add((UnderstandingField.PRICE_PREFERENCE, IgnoreReason.INFORMATIONAL_ONLY))

    try:
        spec = FilterSpec(**kept)
    except ValidationError:
        raise FilterPolicyError() from None
    ordered_ignored = sorted(ignored, key=lambda item: (_UNDERSTANDING_ORDER[item[0]], item[1]))
    return FilterDerivation(
        spec=spec,
        applied_filters=spec.applied_filters(),
        ignored_constraints=tuple(
            IgnoredConstraint(field=field, reason=reason) for field, reason in ordered_ignored
        ),
    )
