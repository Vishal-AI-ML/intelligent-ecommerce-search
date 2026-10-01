"""Conversion between validated `CatalogRecord`s and database rows."""

from enum import Enum
from typing import Any

from ecommerce_search.catalog.schemas import CORE_KEYS, CatalogRecord
from ecommerce_search.catalog.taxonomy import Category
from ecommerce_search.models.catalog import SPEC_TABLES, Product

_NON_ATTRIBUTE_COLUMNS = frozenset({"product_id", "category"})


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def core_values(record: CatalogRecord) -> dict[str, Any]:
    """Column values for `products` (excluding keys, provenance ids, hash and timestamps)."""
    return {name: _plain(getattr(record, name)) for name in CORE_KEYS if name != "product_id"}


def spec_values(record: CatalogRecord) -> dict[str, Any]:
    return {name: _plain(value) for name, value in dict(record.spec).items()}


def spec_columns(category) -> list[str]:
    table = SPEC_TABLES[category].__table__
    return [c.name for c in table.columns if c.name not in _NON_ATTRIBUTE_COLUMNS]


def flat_from_rows(product: Product, spec: Any) -> dict[str, Any]:
    """Flat dict in the same shape as a raw record, for re-validation with `parse_raw`."""
    flat = {name: getattr(product, name) for name in CORE_KEYS}
    for name in spec_columns(Category(product.category)):
        flat[name] = getattr(spec, name)
    return flat
