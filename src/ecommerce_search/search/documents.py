"""Deterministic search-document construction (pure: no database, no network).

A document is built only from a validated, normalized `CatalogRecord`, through a closed
whitelist of fields. It never reads raw seller lines, dataset/provenance data, review outcomes,
seller ids, prices, availability or ratings. Missing attributes emit nothing, and a negative
boolean never emits a positive token (`anc=False` does not emit `anc`).

Four sections map to PostgreSQL weights A to D. Changing the builder, the section-to-weight
assignment or `FTS_CONFIG` requires bumping `DOCUMENT_VERSION` and reindexing.
"""

import re
from dataclasses import dataclass

from ecommerce_search.catalog.schemas import CatalogRecord
from ecommerce_search.catalog.taxonomy import SUBCATEGORIES, Category, Connectivity

DOCUMENT_VERSION = "1"
FTS_CONFIG = "simple"

# Section -> PostgreSQL weight label. Initial baseline assignment, not tuned.
SECTION_WEIGHTS: dict[str, str] = {
    "name": "A",
    "taxonomy": "B",
    "attributes": "C",
    "description": "D",
}

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
    }
)

# Explicit, versioned singular/plural forms of known taxonomy values. Not an inflector, and not
# synonyms: every form below was written down on purpose.
CATEGORY_FORMS: dict[Category, tuple[str, ...]] = {
    Category.LAPTOP: ("laptop", "laptops"),
    Category.PHONE: ("phone", "phones"),
    Category.SHOES: ("shoe", "shoes"),
    Category.HEADPHONES: ("headphone", "headphones"),
}
SUBCATEGORY_FORMS: dict[str, tuple[str, ...]] = {
    "everyday": ("everyday",),
    "business": ("business",),
    "ultrabook": ("ultrabook", "ultrabooks"),
    "performance": ("performance",),
    "smartphone": ("smartphone", "smartphones"),
    "running": ("running",),
    "casual": ("casual",),
    "sports": ("sports",),
    "formal": ("formal",),
    "in-ear": ("in-ear",),
    "on-ear": ("on-ear",),
    "over-ear": ("over-ear",),
}

# Connectivity values tokenize badly as stored (`wireless_2_4ghz` becomes `2` and `4ghz`), so
# the searchable form is written out explicitly.
CONNECTIVITY_TERMS: dict[Connectivity, str] = {
    Connectivity.BLUETOOTH: "bluetooth",
    Connectivity.WIRED: "wired",
    Connectivity.USB: "usb",
    Connectivity.WIRELESS_2_4GHZ: "2.4ghz wireless",
}

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
_EDGE_PUNCTUATION = ",;:()[]\"'"


@dataclass(frozen=True)
class SearchDocument:
    """The four weighted sections of one product's document (plain text for PostgreSQL)."""

    name: str
    taxonomy: str
    attributes: str
    description: str

    def sections(self) -> dict[str, str]:
        return {
            "name": self.name,
            "taxonomy": self.taxonomy,
            "attributes": self.attributes,
            "description": self.description,
        }


def capacity_terms(gb: int) -> list[str]:
    """`256` -> `256gb`; a whole number of TB also gets its TB form (`1024` -> `1024gb`, `1tb`)."""
    terms = [f"{gb}gb"]
    if gb >= 1024 and gb % 1024 == 0:
        terms.append(f"{gb // 1024}tb")
    return terms


def identifier_variants(title: str) -> list[str]:
    """Hyphen-stripped forms of model-like title tokens (`WH-1000XM5` -> `WH1000XM5`).

    A token qualifies only if it is a hyphen-joined alphanumeric word containing both a letter
    and a digit. The variant is emitted only when it differs from the token."""
    variants: list[str] = []
    for chunk in title.split():
        token = chunk.strip(_EDGE_PUNCTUATION)
        if not _IDENTIFIER_RE.fullmatch(token):
            continue
        if not (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)):
            continue
        variant = token.replace("-", "")
        if variant != token and variant not in variants:
            variants.append(variant)
    return variants


def _attribute_terms(record: CatalogRecord) -> list[str]:
    spec = record.spec
    terms: list[str] = []

    def add(value: object) -> None:
        if value is not None and str(value).strip():
            terms.append(str(value))

    match record.category:
        case Category.LAPTOP:
            if spec.ram_gb is not None:
                terms += [f"{spec.ram_gb}gb", "ram"]
            if spec.storage_gb is not None:
                terms += capacity_terms(spec.storage_gb)
            if spec.storage_type is not None:
                terms.append(spec.storage_type.value.lower())
            if spec.storage_interface is not None:
                terms.append(spec.storage_interface.value.lower())
            add(spec.processor)
            add(spec.gpu)
            add(spec.operating_system)
        case Category.PHONE:
            if spec.ram_gb is not None:
                terms += [f"{spec.ram_gb}gb", "ram"]
            if spec.storage_gb is not None:
                terms += capacity_terms(spec.storage_gb)
            add(spec.camera)
            if spec.battery_mah is not None:
                terms.append(f"{spec.battery_mah}mah")
            add(spec.operating_system)
        case Category.SHOES:
            add(spec.color)
            add(spec.material)
            if spec.gender is not None:
                terms.append(spec.gender.value)
        case Category.HEADPHONES:
            if spec.wireless is True:
                terms.append("wireless")
            if spec.connectivity is not None:
                terms.append(CONNECTIVITY_TERMS[spec.connectivity])
            if spec.anc is True:
                terms.append("anc")
    return terms


def build_document(record: CatalogRecord) -> SearchDocument:
    taxonomy = list(CATEGORY_FORMS[record.category])
    if record.subcategory is not None:
        taxonomy += SUBCATEGORY_FORMS[record.subcategory]
    taxonomy += identifier_variants(record.title)
    return SearchDocument(
        name=f"{record.title} {record.brand}",
        taxonomy=" ".join(taxonomy),
        attributes=" ".join(_attribute_terms(record)),
        description=record.description or "",
    )


def _check_taxonomy_coverage() -> None:
    known = {s for subs in SUBCATEGORIES.values() for s in subs}
    if known != set(SUBCATEGORY_FORMS) or set(CATEGORY_FORMS) != set(Category):
        raise RuntimeError("explicit search forms must cover exactly the catalog taxonomy")


_check_taxonomy_coverage()
