"""Deterministic embedding-text construction (pure: no database, no network, no model).

The text is built only from a validated, normalized `CatalogRecord`, through a closed whitelist
of fields. It never reads raw seller lines, dataset/provenance data, review outcomes, seller ids,
prices, currency, availability, ratings, review counts or shoe sizes. Missing attributes emit
nothing, and a negative boolean emits nothing (`anc=False` never mentions noise cancellation).

Model-specific prefixes (for example `passage: `) are not part of this text: they belong to the
model specification and are applied by the provider.

Changing the output of `build_embedding_text` requires bumping `EMBEDDING_TEXT_VERSION`, pinning
the reviewed corpus digest (`tests/unit/search/test_embedding_corpus_digest.py`) and
regenerating every embedding.
"""

import unicodedata
from decimal import Decimal

from ecommerce_search.catalog.schemas import CatalogRecord
from ecommerce_search.catalog.taxonomy import Category, Connectivity

EMBEDDING_TEXT_VERSION = "1"

# The only record fields the builder may read (a test checks this against the label blocklist).
INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "title",
        "brand",
        "category",
        "subcategory",
        "description",
        # laptop
        "ram_gb",
        "storage_gb",
        "storage_type",
        "storage_interface",
        "processor",
        "gpu",
        "operating_system",
        "screen_size_inches",
        "weight_kg",
        # phone
        "camera",
        "battery_mah",
        # shoes
        "color",
        "material",
        "gender",
        # headphones
        "wireless",
        "anc",
        "connectivity",
        "battery_life_hours",
    }
)

CONNECTIVITY_PHRASES: dict[Connectivity, str] = {
    Connectivity.BLUETOOTH: "Bluetooth connectivity",
    Connectivity.WIRED: "wired connection",
    Connectivity.USB: "USB connection",
    Connectivity.WIRELESS_2_4GHZ: "2.4 GHz wireless connection",
}


def number_text(value: Decimal | int) -> str:
    """`14.0` -> `14`, `1.40` -> `1.4`, `5000` -> `5000` (never scientific notation)."""
    if isinstance(value, int):
        return str(value)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def capacity_text(gb: int) -> str:
    """`512` -> `512 GB`; a whole number of TB is written as TB (`1024` -> `1 TB`)."""
    if gb >= 1024 and gb % 1024 == 0:
        return f"{gb // 1024} TB"
    return f"{gb} GB"


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _specifications(record: CatalogRecord) -> list[str]:
    spec = record.spec
    parts: list[str] = []
    match record.category:
        case Category.LAPTOP:
            if spec.ram_gb is not None:
                parts.append(f"{spec.ram_gb} GB RAM")
            if spec.storage_gb is not None:
                storage = capacity_text(spec.storage_gb)
                if spec.storage_type is not None:
                    storage += f" {spec.storage_type.value}"
                storage += " storage"
                if spec.storage_interface is not None:
                    storage += f" ({spec.storage_interface.value.replace('NVME', 'NVMe')})"
                parts.append(storage)
            elif spec.storage_type is not None:
                parts.append(f"{spec.storage_type.value} storage")
            if spec.processor:
                parts.append(f"processor {spec.processor}")
            if spec.gpu:
                parts.append(f"graphics {spec.gpu}")
            if spec.operating_system:
                parts.append(spec.operating_system)
            if spec.screen_size_inches is not None:
                parts.append(f"{number_text(spec.screen_size_inches)}-inch screen")
            if spec.weight_kg is not None:
                parts.append(f"weight {number_text(spec.weight_kg)} kg")
        case Category.PHONE:
            if spec.ram_gb is not None:
                parts.append(f"{spec.ram_gb} GB RAM")
            if spec.storage_gb is not None:
                parts.append(f"{capacity_text(spec.storage_gb)} storage")
            if spec.camera:
                parts.append(f"camera {spec.camera}")
            if spec.battery_mah is not None:
                parts.append(f"{spec.battery_mah} mAh battery")
            if spec.screen_size_inches is not None:
                parts.append(f"{number_text(spec.screen_size_inches)}-inch screen")
            if spec.operating_system:
                parts.append(spec.operating_system)
        case Category.SHOES:
            if spec.color:
                parts.append(f"color {spec.color}")
            if spec.material:
                parts.append(f"material {spec.material}")
            if spec.gender is not None:
                parts.append(f"for {spec.gender.value}")
        case Category.HEADPHONES:
            if spec.wireless is True:
                parts.append("wireless")
            if spec.connectivity is not None:
                parts.append(CONNECTIVITY_PHRASES[spec.connectivity])
            if spec.anc is True:
                parts.append("active noise cancellation (ANC)")
            if spec.battery_life_hours is not None:
                parts.append(f"up to {number_text(spec.battery_life_hours)} hours battery life")
    return [_clean(p) for p in parts if _clean(p)]


def build_embedding_text(record: CatalogRecord) -> str:
    """One product's embedding text: title, brand/category, specifications, description."""
    category = record.category.value
    if record.subcategory is not None:
        category += f" ({record.subcategory})"
    lines = [
        _clean(record.title),
        _clean(f"Brand: {record.brand}. Category: {category}."),
    ]
    specs = _specifications(record)
    if specs:
        lines.append("Specifications: " + "; ".join(specs) + ".")
    if record.description:
        lines.append(_clean(record.description))
    return "\n".join(line for line in lines if line)
