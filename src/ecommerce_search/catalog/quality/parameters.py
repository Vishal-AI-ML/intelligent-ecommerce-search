"""Centralised, versioned quality parameters.

These are provisional engineering guardrails, NOT measured marketplace facts.

* HARD bounds reject impossible values (schema-level errors).
* PLAUSIBILITY sets/ranges and PRICE bands only produce warnings; an uncommon value is
  never rejected for being uncommon.
"""

from decimal import Decimal

from ecommerce_search.catalog.taxonomy import Category, SizeSystem

RULES_VERSION = "1"

# ---- hard bounds (impossible values -> error) -------------------------------------------
PRICE_MAX = Decimal("9999999999.99")
RATING_MIN, RATING_MAX = Decimal(0), Decimal(5)
REVIEW_COUNT_MAX = 2_147_483_647
RAM_GB_MAX = 4096
STORAGE_GB_MAX = 65536
SCREEN_INCHES_MAX = Decimal(100)
WEIGHT_KG_MAX = Decimal(100)
BATTERY_MAH_MAX = 100_000
BATTERY_HOURS_MAX = Decimal(1000)
SHOE_SIZE_MAX = Decimal(60)

# ---- provisional plausibility (warnings) ------------------------------------------------
EXPECTED_CURRENCY = "INR"

COMMON_RAM_GB: dict[Category, frozenset[int]] = {
    Category.LAPTOP: frozenset({4, 8, 12, 16, 24, 32, 64}),
    Category.PHONE: frozenset({2, 3, 4, 6, 8, 12, 16}),
}
COMMON_STORAGE_GB: dict[Category, frozenset[int]] = {
    Category.LAPTOP: frozenset({128, 256, 512, 1024, 2048}),
    Category.PHONE: frozenset({32, 64, 128, 256, 512, 1024}),
}

# (category, field) -> inclusive plausible range
PLAUSIBLE_RANGES: dict[tuple[Category, str], tuple[Decimal, Decimal]] = {
    (Category.LAPTOP, "screen_size_inches"): (Decimal(10), Decimal(18)),
    (Category.LAPTOP, "weight_kg"): (Decimal("0.5"), Decimal(5)),
    (Category.PHONE, "screen_size_inches"): (Decimal(4), Decimal(8)),
    (Category.PHONE, "battery_mah"): (Decimal(2000), Decimal(10000)),
    (Category.HEADPHONES, "battery_life_hours"): (Decimal(1), Decimal(100)),
}
SHOE_SIZE_RANGES: dict[SizeSystem, tuple[Decimal, Decimal]] = {
    SizeSystem.UK: (Decimal(1), Decimal(15)),
    SizeSystem.US: (Decimal(1), Decimal(17)),
    SizeSystem.EU: (Decimal(16), Decimal(52)),
}

PRICE_BANDS_INR: dict[Category, tuple[Decimal, Decimal]] = {
    Category.LAPTOP: (Decimal(10_000), Decimal(500_000)),
    Category.PHONE: (Decimal(3_000), Decimal(250_000)),
    Category.SHOES: (Decimal(300), Decimal(30_000)),
    Category.HEADPHONES: (Decimal(200), Decimal(60_000)),
}

# Expected (warning if absent) vs optional (info if absent) attributes per category.
EXPECTED_FIELDS: dict[Category, tuple[str, ...]] = {
    Category.LAPTOP: (
        "ram_gb",
        "storage_gb",
        "storage_type",
        "processor",
        "screen_size_inches",
        "operating_system",
    ),
    Category.PHONE: (
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "screen_size_inches",
        "operating_system",
    ),
    Category.SHOES: ("size", "size_system", "color", "gender"),
    Category.HEADPHONES: ("wireless", "connectivity"),
}
OPTIONAL_FIELDS: dict[Category, tuple[str, ...]] = {
    Category.LAPTOP: ("storage_interface", "gpu", "weight_kg"),
    Category.PHONE: ("camera",),
    Category.SHOES: ("material",),
    Category.HEADPHONES: ("anc", "battery_life_hours"),
}
OPTIONAL_CORE_FIELDS = ("description", "subcategory", "rating", "review_count")

NEAR_DUPLICATE_JACCARD = Decimal("0.9")

# Field names that must never exist on catalog tables/models or in seed records: they belong
# to evaluation labels or to the separate human-review artifact.
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "relevance",
        "relevance_label",
        "relevant",
        "borderline",
        "not_relevant",
        "label",
        "labels",
        "golden",
        "golden_label",
        "judgment",
        "judgement",
        "qrel",
        "qrels",
        "verdict",
        "reviewer",
        "issue_fields",
        "review_notes",
        "human_review",
        "human_label",
    }
)


def _num(value: Decimal) -> str:
    return format(value.normalize(), "f")


def parameters_as_dict() -> dict:
    """Every implemented set/range/band/threshold, for embedding in quality reports."""
    return {
        "rules_version": RULES_VERSION,
        "kind": "provisional engineering guardrails, not measured marketplace facts",
        "hard_bounds": {
            "price": f"> 0 and <= {_num(PRICE_MAX)}, 2 decimal places",
            "rating": f">= {RATING_MIN} and <= {RATING_MAX}, 1 decimal place",
            "review_count": f">= 0 and <= {REVIEW_COUNT_MAX}",
            "ram_gb": f">= 1 and <= {RAM_GB_MAX}",
            "storage_gb": f">= 1 and <= {STORAGE_GB_MAX}",
            "screen_size_inches": f"> 0 and <= {SCREEN_INCHES_MAX}, 1 decimal place",
            "weight_kg": f"> 0 and <= {WEIGHT_KG_MAX}, 2 decimal places",
            "battery_mah": f">= 1 and <= {BATTERY_MAH_MAX}",
            "battery_life_hours": f"> 0 and <= {BATTERY_HOURS_MAX}, 1 decimal place",
            "size": f"> 0 and <= {SHOE_SIZE_MAX}, 1 decimal place",
        },
        "expected_currency": EXPECTED_CURRENCY,
        "common_ram_gb": {c.value: sorted(v) for c, v in COMMON_RAM_GB.items()},
        "common_storage_gb": {c.value: sorted(v) for c, v in COMMON_STORAGE_GB.items()},
        "plausible_ranges": {
            f"{c.value}.{f}": [_num(lo), _num(hi)] for (c, f), (lo, hi) in PLAUSIBLE_RANGES.items()
        },
        "shoe_size_ranges": {
            s.value: [_num(lo), _num(hi)] for s, (lo, hi) in SHOE_SIZE_RANGES.items()
        },
        "price_bands_inr": {
            c.value: [_num(lo), _num(hi)] for c, (lo, hi) in PRICE_BANDS_INR.items()
        },
        "expected_fields": {c.value: list(v) for c, v in EXPECTED_FIELDS.items()},
        "optional_fields": {c.value: list(v) for c, v in OPTIONAL_FIELDS.items()},
        "near_duplicate_jaccard_threshold": _num(NEAR_DUPLICATE_JACCARD),
        "near_duplicate_tokenisation": "casefolded [a-z0-9]+ tokens of the title, set Jaccard",
    }
