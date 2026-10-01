"""Validated, normalized catalog records (Pydantic).

`parse_raw` turns one flat raw seller-style object into a `CatalogRecord`. Normalization is
deterministic and strict: unknown units, unknown taxonomy values and attributes that do not
belong to the category are errors, never silently mapped or dropped. General listing
normalization is a later milestone; only what Milestone 2 needs lives here.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.taxonomy import Category

# ---- primitive parsers ------------------------------------------------------------------

_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MEASURE_RE = re.compile(
    rf"(?P<prefix>₹|rs\.?|inr)?\s*(?P<num>{_NUMBER})\s*(?P<unit>[a-z\"]*)", re.IGNORECASE
)

CAPACITY_UNITS = {"gb": Decimal(1), "tb": Decimal(1024)}
RAM_UNITS = {"gb": Decimal(1)}
SCREEN_UNITS = {"in": Decimal(1), "inch": Decimal(1), "inches": Decimal(1), '"': Decimal(1)}
WEIGHT_UNITS = {"kg": Decimal(1)}
BATTERY_UNITS = {"mah": Decimal(1)}
HOUR_UNITS = {h: Decimal(1) for h in ("h", "hr", "hrs", "hour", "hours")}
NO_UNITS: dict[str, Decimal] = {}


def parse_decimal_measure(
    value: Any,
    *,
    what: str,
    units: Mapping[str, Decimal],
    dp: int,
    prefixes: bool = False,
) -> Decimal:
    """Parse a number or `<number><unit>` string into a Decimal with exactly `dp` places.

    A bare number has the field's default unit. An unknown unit or excess precision is an
    error.
    """
    multiplier = Decimal(1)
    if isinstance(value, bool):
        raise ValueError(f"{what}: a boolean is not a number")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(repr(value))
    elif isinstance(value, str):
        match = _MEASURE_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"{what}: cannot parse {value!r}")
        if match["prefix"] and not prefixes:
            raise ValueError(f"{what}: unexpected currency marker in {value!r}")
        number = Decimal(match["num"].replace(",", ""))
        unit = match["unit"].casefold()
        if unit:
            if unit not in units:
                raise ValueError(f"{what}: unknown unit {match['unit']!r}")
            multiplier = units[unit]
    else:
        raise ValueError(f"{what}: expected a number or string, got {type(value).__name__}")
    if not number.is_finite():
        raise ValueError(f"{what}: not a finite number")
    number *= multiplier
    try:
        quantized = number.quantize(Decimal(1).scaleb(-dp))
    except InvalidOperation as exc:
        raise ValueError(f"{what}: value out of range") from exc
    if quantized != number:
        raise ValueError(f"{what}: more than {dp} decimal place(s) in {value!r}")
    return quantized


def _bounded(
    number: Decimal,
    what: str,
    gt: Decimal | int | None,
    ge: Decimal | int | None,
    le: Decimal | int | None,
) -> None:
    if gt is not None and not number > gt:
        raise ValueError(f"{what}: must be > {gt}")
    if ge is not None and not number >= ge:
        raise ValueError(f"{what}: must be >= {ge}")
    if le is not None and not number <= le:
        raise ValueError(f"{what}: must be <= {le}")


def decimal_field(
    what: str,
    *,
    units: Mapping[str, Decimal],
    dp: int,
    gt: Decimal | int | None = None,
    ge: Decimal | int | None = None,
    le: Decimal | int | None = None,
    prefixes: bool = False,
    nullable: bool = True,
) -> BeforeValidator:
    def _validate(value: Any) -> Decimal | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(f"{what} is required")
        number = parse_decimal_measure(value, what=what, units=units, dp=dp, prefixes=prefixes)
        _bounded(number, what, gt, ge, le)
        return number

    return BeforeValidator(_validate)


def int_field(
    what: str,
    *,
    units: Mapping[str, Decimal],
    ge: int,
    le: int,
    nullable: bool = True,
) -> BeforeValidator:
    def _validate(value: Any) -> int | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(f"{what} is required")
        number = parse_decimal_measure(value, what=what, units=units, dp=0)
        _bounded(number, what, None, ge, le)
        return int(number)

    return BeforeValidator(_validate)


def clean_text(value: Any, what: str = "text") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what}: expected a string, got {type(value).__name__}")
    return " ".join(value.split()) or None


def _required_text(value: Any) -> str:
    cleaned = clean_text(value)
    if cleaned is None:
        raise ValueError("value is required and must not be blank")
    return cleaned


def _optional_text(value: Any) -> str | None:
    return clean_text(value)


def _brand(value: Any) -> str:
    return tx.canonical_brand(_required_text(value))


def _subcategory(value: Any) -> str | None:
    cleaned = clean_text(value, "subcategory")
    return cleaned.casefold() if cleaned else None


def _currency(value: Any) -> str:
    return _required_text(value).upper()


def enum_parser[E: Enum](enum_cls: type[E], *, nullable: bool) -> BeforeValidator:
    lookup = {str(m.value).casefold(): m for m in enum_cls}

    def _validate(value: Any) -> E | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(f"{enum_cls.__name__} value is required")
        if isinstance(value, enum_cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{enum_cls.__name__}: expected a string")
        member = lookup.get(" ".join(value.split()).casefold())
        if member is None:
            allowed = ", ".join(str(m.value) for m in enum_cls)
            raise ValueError(f"unknown {enum_cls.__name__} value {value!r}; allowed: {allowed}")
        return member

    return BeforeValidator(_validate)


# ---- field types ------------------------------------------------------------------------

ProductId = Annotated[
    str,
    BeforeValidator(_required_text),
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
]
ReqText = Annotated[str, BeforeValidator(_required_text)]
OptText = Annotated[str | None, BeforeValidator(_optional_text)]
Brand = Annotated[str, BeforeValidator(_brand)]
Subcategory = Annotated[str | None, BeforeValidator(_subcategory)]
Currency = Annotated[str, BeforeValidator(_currency), StringConstraints(pattern=r"^[A-Z]{3}$")]

Price = Annotated[
    Decimal,
    decimal_field(
        "price", units=NO_UNITS, dp=2, gt=0, le=qp.PRICE_MAX, prefixes=True, nullable=False
    ),
]
Rating = Annotated[
    Decimal | None,
    decimal_field("rating", units=NO_UNITS, dp=1, ge=qp.RATING_MIN, le=qp.RATING_MAX),
]
ReviewCount = Annotated[
    int | None, int_field("review_count", units=NO_UNITS, ge=0, le=qp.REVIEW_COUNT_MAX)
]
RamGb = Annotated[int | None, int_field("ram_gb", units=RAM_UNITS, ge=1, le=qp.RAM_GB_MAX)]
StorageGb = Annotated[
    int | None, int_field("storage_gb", units=CAPACITY_UNITS, ge=1, le=qp.STORAGE_GB_MAX)
]
ScreenInches = Annotated[
    Decimal | None,
    decimal_field("screen_size_inches", units=SCREEN_UNITS, dp=1, gt=0, le=qp.SCREEN_INCHES_MAX),
]
WeightKg = Annotated[
    Decimal | None,
    decimal_field("weight_kg", units=WEIGHT_UNITS, dp=2, gt=0, le=qp.WEIGHT_KG_MAX),
]
BatteryMah = Annotated[
    int | None, int_field("battery_mah", units=BATTERY_UNITS, ge=1, le=qp.BATTERY_MAH_MAX)
]
BatteryHours = Annotated[
    Decimal | None,
    decimal_field("battery_life_hours", units=HOUR_UNITS, dp=1, gt=0, le=qp.BATTERY_HOURS_MAX),
]
ShoeSize = Annotated[
    Decimal | None,
    decimal_field("size", units=NO_UNITS, dp=1, gt=0, le=qp.SHOE_SIZE_MAX),
]

CategoryField = Annotated[Category, enum_parser(Category, nullable=False)]
AvailabilityField = Annotated[tx.Availability, enum_parser(tx.Availability, nullable=False)]
SourceTypeField = Annotated[tx.SourceType, enum_parser(tx.SourceType, nullable=False)]
StorageTypeField = Annotated[tx.StorageType | None, enum_parser(tx.StorageType, nullable=True)]
StorageInterfaceField = Annotated[
    tx.StorageInterface | None, enum_parser(tx.StorageInterface, nullable=True)
]
SizeSystemField = Annotated[tx.SizeSystem | None, enum_parser(tx.SizeSystem, nullable=True)]
GenderField = Annotated[tx.Gender | None, enum_parser(tx.Gender, nullable=True)]
ConnectivityField = Annotated[tx.Connectivity | None, enum_parser(tx.Connectivity, nullable=True)]

# ---- per-category attributes ------------------------------------------------------------


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LaptopSpecData(_Spec):
    ram_gb: RamGb = None
    storage_gb: StorageGb = None
    storage_type: StorageTypeField = None
    storage_interface: StorageInterfaceField = None
    processor: OptText = None
    gpu: OptText = None
    screen_size_inches: ScreenInches = None
    operating_system: OptText = None
    weight_kg: WeightKg = None

    @model_validator(mode="after")
    def _nvme_requires_ssd(self) -> "LaptopSpecData":
        if (
            self.storage_interface is tx.StorageInterface.NVME
            and self.storage_type is not tx.StorageType.SSD
        ):
            raise ValueError("storage_interface NVME requires storage_type SSD")
        return self


class PhoneSpecData(_Spec):
    ram_gb: RamGb = None
    storage_gb: StorageGb = None
    camera: OptText = None
    battery_mah: BatteryMah = None
    screen_size_inches: ScreenInches = None
    operating_system: OptText = None


class ShoeSpecData(_Spec):
    size: ShoeSize = None
    size_system: SizeSystemField = None
    color: OptText = None
    material: OptText = None
    gender: GenderField = None

    @model_validator(mode="after")
    def _size_and_system_together(self) -> "ShoeSpecData":
        if (self.size is None) != (self.size_system is None):
            raise ValueError("size and size_system must be provided together")
        return self


class HeadphoneSpecData(_Spec):
    wireless: StrictBool | None = None
    anc: StrictBool | None = None
    battery_life_hours: BatteryHours = None
    connectivity: ConnectivityField = None

    @model_validator(mode="after")
    def _connectivity_matches_wireless(self) -> "HeadphoneSpecData":
        c = tx.Connectivity
        if self.connectivity is c.WIRED and self.wireless is True:
            raise ValueError("connectivity 'wired' contradicts wireless=true")
        if self.connectivity in (c.BLUETOOTH, c.WIRELESS_2_4GHZ) and self.wireless is False:
            raise ValueError(f"connectivity {self.connectivity.value!r} contradicts wireless=false")
        return self


SpecData = LaptopSpecData | PhoneSpecData | ShoeSpecData | HeadphoneSpecData
SPEC_MODELS: dict[Category, type[_Spec]] = {
    Category.LAPTOP: LaptopSpecData,
    Category.PHONE: PhoneSpecData,
    Category.SHOES: ShoeSpecData,
    Category.HEADPHONES: HeadphoneSpecData,
}

# ---- record -----------------------------------------------------------------------------


class CoreFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: ProductId
    seller_id: OptText = None
    title: ReqText
    description: OptText = None
    category: CategoryField
    subcategory: Subcategory = None
    brand: Brand
    price: Price
    currency: Currency
    rating: Rating = None
    review_count: ReviewCount = None
    availability: AvailabilityField
    source_type: SourceTypeField
    is_synthetic: StrictBool


CORE_KEYS = tuple(CoreFields.model_fields)


class CatalogRecord(CoreFields):
    spec: SpecData

    @model_validator(mode="after")
    def _cross_field(self) -> "CatalogRecord":
        if not isinstance(self.spec, SPEC_MODELS[self.category]):
            raise ValueError(f"attributes do not belong to category {self.category.value!r}")
        if self.subcategory is not None and self.subcategory not in tx.SUBCATEGORIES[self.category]:
            allowed = ", ".join(sorted(tx.SUBCATEGORIES[self.category]))
            raise ValueError(
                f"invalid subcategory {self.subcategory!r} for {self.category.value}; "
                f"allowed: {allowed}"
            )
        if self.is_synthetic != (self.source_type is tx.SourceType.SYNTHETIC):
            raise ValueError(
                f"is_synthetic={self.is_synthetic} is inconsistent with "
                f"source_type={self.source_type.value!r}"
            )
        return self


class RecordValidationError(ValueError):
    """Carries every (field, message) problem found in one raw record."""

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{field}: {message}" for field, message in errors))


def _flatten(exc: ValidationError, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "<record>"
        message = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            message = "attribute is not allowed for this category or is unknown"
        out.append((f"{prefix}{field}", message))
    return out


def parse_raw(raw: Any) -> CatalogRecord:
    """Validate and normalize one raw record; raise RecordValidationError listing all problems."""
    if not isinstance(raw, dict):
        raise RecordValidationError([("<record>", "record must be a JSON object")])
    errors: list[tuple[str, str]] = []
    core_input = {k: v for k, v in raw.items() if k in CORE_KEYS}
    attr_input = {k: v for k, v in raw.items() if k not in CORE_KEYS}

    core: CoreFields | None = None
    try:
        core = CoreFields.model_validate(core_input)
    except ValidationError as exc:
        errors.extend(_flatten(exc))

    # The category alone decides which attribute model applies, so attributes are checked
    # even when another core field is invalid and all problems are reported together.
    category = core.category if core is not None else _parse_category(raw.get("category"))
    spec: _Spec | None = None
    if category is not None:
        try:
            spec = SPEC_MODELS[category].model_validate(attr_input)
        except ValidationError as exc:
            errors.extend(_flatten(exc))
    elif attr_input:
        errors.append(("<attributes>", "not checked: category is invalid or missing"))

    if core is not None:
        errors.extend(_cross_field_errors(core))

    if errors or core is None or spec is None:
        raise RecordValidationError(errors)
    try:
        return CatalogRecord(**dict(core), spec=spec)
    except ValidationError as exc:
        raise RecordValidationError(_flatten(exc)) from exc


_CATEGORY_ADAPTER = TypeAdapter(CategoryField)


def _parse_category(value: Any) -> Category | None:
    try:
        return _CATEGORY_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _cross_field_errors(core: CoreFields) -> list[tuple[str, str]]:
    """Cross-field problems attributed to the field at fault (the model validator on
    CatalogRecord remains as a second line of defence)."""
    errors = []
    if core.subcategory is not None and core.subcategory not in tx.SUBCATEGORIES[core.category]:
        allowed = ", ".join(sorted(tx.SUBCATEGORIES[core.category]))
        errors.append(
            (
                "subcategory",
                f"invalid subcategory {core.subcategory!r} for {core.category.value}; "
                f"allowed: {allowed}",
            )
        )
    if core.is_synthetic != (core.source_type is tx.SourceType.SYNTHETIC):
        errors.append(
            (
                "is_synthetic",
                f"is_synthetic={core.is_synthetic} is inconsistent with "
                f"source_type={core.source_type.value!r}",
            )
        )
    return errors


def record_content_hash(record: CatalogRecord) -> str:
    """SHA-256 of the canonical normalized content (no timestamps, no provenance ids)."""
    canonical = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
