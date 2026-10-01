"""Catalog taxonomy: closed value sets, brand dictionary and versions.

Every set here is mirrored by a named CHECK constraint (TEXT + CHECK, not native ENUMs); a
test keeps the two in sync. Query-matching semantics for these values (for example how an
`ssd` query relates to NVME) are deliberately not defined here; that is decided in later
milestones.
"""

from enum import StrEnum

TAXONOMY_VERSION = "1"
TRANSFORM_VERSION = "1"


class Category(StrEnum):
    LAPTOP = "laptop"
    PHONE = "phone"
    SHOES = "shoes"
    HEADPHONES = "headphones"


class StorageType(StrEnum):
    """Storage medium."""

    SSD = "SSD"
    HDD = "HDD"


class StorageInterface(StrEnum):
    """Storage interface. NVME requires an SSD medium; it is not an alternative to SSD."""

    NVME = "NVME"
    SATA = "SATA"


class Availability(StrEnum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"


class SourceType(StrEnum):
    SYNTHETIC = "synthetic"
    PUBLIC_DATASET = "public_dataset"


class SizeSystem(StrEnum):
    UK = "UK"
    US = "US"
    EU = "EU"


class Gender(StrEnum):
    MEN = "men"
    WOMEN = "women"
    UNISEX = "unisex"
    KIDS = "kids"


class Connectivity(StrEnum):
    BLUETOOTH = "bluetooth"
    WIRED = "wired"
    USB = "usb"
    WIRELESS_2_4GHZ = "wireless_2_4ghz"


class ReviewVerdict(StrEnum):
    ACCEPT = "accept"
    NEEDS_CORRECTION = "needs_correction"
    UNSURE = "unsure"


# Subcategory values are validated by the application (not by a DB CHECK).
SUBCATEGORIES: dict[Category, frozenset[str]] = {
    Category.LAPTOP: frozenset({"everyday", "business", "ultrabook", "performance"}),
    Category.PHONE: frozenset({"smartphone"}),
    Category.SHOES: frozenset({"running", "casual", "sports", "formal"}),
    Category.HEADPHONES: frozenset({"in-ear", "on-ear", "over-ear"}),
}

# Brand dictionary: canonical brand -> categories it is plausibly sold in. Real brand names
# are used only as labels on synthetic records; no affiliation or data source is implied.
BRAND_CATEGORIES: dict[str, frozenset[Category]] = {
    "HP": frozenset({Category.LAPTOP}),
    "Dell": frozenset({Category.LAPTOP}),
    "Lenovo": frozenset({Category.LAPTOP}),
    "Asus": frozenset({Category.LAPTOP, Category.PHONE}),
    "Acer": frozenset({Category.LAPTOP}),
    "MSI": frozenset({Category.LAPTOP}),
    "Apple": frozenset({Category.LAPTOP, Category.PHONE, Category.HEADPHONES}),
    "Samsung": frozenset({Category.PHONE, Category.HEADPHONES}),
    "Xiaomi": frozenset({Category.PHONE, Category.HEADPHONES}),
    "OnePlus": frozenset({Category.PHONE, Category.HEADPHONES}),
    "Realme": frozenset({Category.PHONE, Category.HEADPHONES}),
    "Motorola": frozenset({Category.PHONE}),
    "Nike": frozenset({Category.SHOES}),
    "Adidas": frozenset({Category.SHOES}),
    "Puma": frozenset({Category.SHOES}),
    "Reebok": frozenset({Category.SHOES}),
    "Skechers": frozenset({Category.SHOES}),
    "Sony": frozenset({Category.HEADPHONES}),
    "Bose": frozenset({Category.HEADPHONES}),
    "JBL": frozenset({Category.HEADPHONES}),
    "boAt": frozenset({Category.HEADPHONES}),
    "Sennheiser": frozenset({Category.HEADPHONES}),
}

# Case-insensitive lookup; explicit aliases are kept to the minimum M2 needs.
_EXTRA_ALIASES = {"hewlett-packard": "HP", "one plus": "OnePlus"}
BRAND_LOOKUP: dict[str, str] = {b.casefold(): b for b in BRAND_CATEGORIES} | _EXTRA_ALIASES


def canonical_brand(text: str) -> str:
    """Canonical dictionary spelling for a known brand or alias; other text is kept as given."""
    return BRAND_LOOKUP.get(text.casefold(), text)


def brand_categories(brand: str) -> frozenset[Category] | None:
    """Categories a known brand is expected in, or None when the brand is not in the dictionary."""
    return BRAND_CATEGORIES.get(brand)
