"""Milestone 7 filtered-retrieval truth table and catalog oracle (scratch databases only).

Independence rules for this module:

* It never imports `ecommerce_search.search.filtered` (no `FILTER_FRAGMENTS`, no statement
  builders) and never runs a production retrieval statement.
* Eligibility is computed from plain `SELECT`s of the catalog tables and Python predicates
  written from the fp-1 rules (exact equality; RAM/storage on a laptop or phone spec row;
  storage type/interface on a laptop spec row; inclusive prices; missing or NULL spec data
  never matches).
* Current embeddings are recomputed in Python from the raw embedding rows and the M4 freshness
  rule, not by the dense retrieval SQL.
* The truth table below was written by hand from fp-1 and the seed catalog listing
  (`data/seed/catalog_seed_v1.jsonl`) plus the deliberate MUTATIONS, before any filtered
  retrieval ran. `expect`, `include` and `exclude` are predeclared, not copied from output.
"""

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import text

from ecommerce_search.catalog.taxonomy import Category, StorageInterface, StorageType
from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.filtering import FilterSpec

LAPTOP, PHONE, SHOES, HEADPHONES = (
    Category.LAPTOP,
    Category.PHONE,
    Category.SHOES,
    Category.HEADPHONES,
)
SSD, HDD = StorageType.SSD, StorageType.HDD
NVME = StorageInterface.NVME

# Deliberate catalog damage applied after ingest + fake embedding (see `apply_mutations`).
MISSING_LAPTOP_SPEC = "SYN-LAP-0057"  # HP laptop 43999, was 8GB/256GB SSD NVME: row deleted
MISSING_PHONE_SPEC = "SYN-PHN-0049"  # Samsung phone 26999, was 8GB/256GB: row deleted
NULL_RAM_LAPTOP = "SYN-LAP-0071"  # HP laptop 51999, 512GB SSD NVME, ram_gb set NULL
NULL_STORAGE_TYPE_LAPTOP = "SYN-LAP-0036"  # HP laptop 40999, 8GB/512GB, type+interface NULL
STALE_EMBEDDING = "SYN-LAP-0001"  # HP laptop 33999, 8GB/256GB SSD SATA: embedding made stale
MISSING_EMBEDDING = "SYN-LAP-0016"  # Dell laptop 40999, 8GB/512GB SSD NVME: embedding deleted
NOT_CURRENT = frozenset({STALE_EMBEDDING, MISSING_EMBEDDING})

FULL_DEPTH = 1000  # larger than the 240-product catalog
K_VALUES = (1, 5, 50)


@dataclass(frozen=True)
class Row:
    id: str
    query: str  # the normalized query text sent to both sources
    spec: FilterSpec  # expected fp-1 spec (parsed rows) or the spec under test (direct rows)
    parsed: bool  # True: the spec must equal derive_filters(parse(query)).spec
    expect: str  # "empty" | "nonempty": the oracle-eligible catalog set
    include: frozenset[str] = field(default_factory=frozenset)
    exclude: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


def _row(id, query, *, parsed, expect, include=(), exclude=(), note="", **spec) -> Row:
    return Row(
        id=id,
        query=query,
        spec=FilterSpec(**spec),
        parsed=parsed,
        expect=expect,
        include=frozenset(include),
        exclude=frozenset(exclude),
        note=note,
    )


def P(id, query, expect, **kw) -> Row:  # noqa: N802 - table shorthand
    return _row(id, query, parsed=True, expect=expect, **kw)


def D(id, query, expect, **kw) -> Row:  # noqa: N802 - table shorthand
    return _row(id, query, parsed=False, expect=expect, **kw)


TRUTH_TABLE: tuple[Row, ...] = (
    # --- parsed queries: real qu-1 parse -> fp-1 spec (expected spec written by hand) ---------
    P(
        "cat-brand",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        include={MISSING_LAPTOP_SPEC, NULL_RAM_LAPTOP, NULL_STORAGE_TYPE_LAPTOP, STALE_EMBEDDING},
        exclude={"SYN-LAP-0002", "SYN-PHN-0008"},
        note="non-spec filters keep products with missing/NULL spec data",
    ),
    P(
        "cat-brand-ram",
        "hp laptop 8gb ram",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        ram_gb=8,
        include={NULL_STORAGE_TYPE_LAPTOP, STALE_EMBEDDING, "SYN-LAP-0064"},
        exclude={MISSING_LAPTOP_SPEC, NULL_RAM_LAPTOP, "SYN-LAP-0008", "SYN-LAP-0015"},
    ),
    P(
        "ram-only",
        "8gb ram",
        "nonempty",
        ram_gb=8,
        include={"SYN-LAP-0001", "SYN-PHN-0004", "SYN-PHN-0026"},
        exclude={MISSING_LAPTOP_SPEC, MISSING_PHONE_SPEC, "SYN-PHN-0003", "SYN-LAP-0015"},
        note="RAM spans laptop_specs OR phone_specs",
    ),
    P(
        "storage-ssd",
        "1tb ssd",
        "nonempty",
        storage_gb=1024,
        storage_type=SSD,
        include={"SYN-LAP-0003"},
        exclude={"SYN-PHN-0016", "SYN-LAP-0004", "SYN-LAP-0002"},
        note="1TB phones have no storage type; 1TB HDD laptops are not SSD",
    ),
    P(
        "ssd",
        "ssd",
        "nonempty",
        storage_type=SSD,
        include={NULL_RAM_LAPTOP, "SYN-LAP-0001"},
        exclude={NULL_STORAGE_TYPE_LAPTOP, MISSING_LAPTOP_SPEC, "SYN-LAP-0004", "SYN-PHN-0004"},
    ),
    P(
        "nvme",
        "nvme",
        "nonempty",
        storage_type=SSD,
        storage_interface=NVME,
        include={"SYN-LAP-0002"},
        exclude={"SYN-LAP-0001", "SYN-LAP-0004", NULL_STORAGE_TYPE_LAPTOP},
    ),
    P(
        "hdd",
        "hdd laptop",
        "nonempty",
        category=LAPTOP,
        storage_type=HDD,
        include={"SYN-LAP-0004", "SYN-LAP-0050"},
        exclude={"SYN-LAP-0001", NULL_STORAGE_TYPE_LAPTOP},
    ),
    P("impossible-phone-ssd", "phone ssd", "empty", category=PHONE, storage_type=SSD),
    P("impossible-brand", "nike laptop", "empty", category=LAPTOP, brand="Nike"),
    P("conflict-storage", "nvme hdd", "nonempty", note="storage conflict: no filters"),
    P("conflict-ssd-hdd", "ssd hdd", "nonempty"),
    P(
        "conflict-brand",
        "hp dell laptop",
        "nonempty",
        category=LAPTOP,
        include={"SYN-LAP-0001", "SYN-LAP-0002"},
    ),
    P("conflict-price", "above 30k under 20k", "nonempty"),
    P("informational-intent", "coding ke liye laptop", "nonempty", category=LAPTOP),
    P(
        "informational-price",
        "sasta phone",
        "nonempty",
        category=PHONE,
        include={MISSING_PHONE_SPEC},
    ),
    P("ambiguous-ram", "16gb ram laptop ₹40,000", "nonempty", category=LAPTOP),
    P("ambiguous-bare", "laptop 16gb", "nonempty", category=LAPTOP),
    P("ambiguous-capacity", "8 256 ssd", "nonempty", storage_type=SSD),
    P("ambiguous-memory", "memory 8gb", "nonempty"),
    P("ram-ignores-price-word", "laptop 8gb ram 50k", "nonempty", category=LAPTOP, ram_gb=8),
    P("ram-not-storage", "ram 8gb ssd", "nonempty", storage_type=SSD),
    P(
        "full",
        "hp laptop 8gb ram 256gb ssd under 40k",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        ram_gb=8,
        storage_gb=256,
        storage_type=SSD,
        max_price=Decimal("40000.00"),
        include={"SYN-LAP-0001"},
        exclude={MISSING_LAPTOP_SPEC, "SYN-LAP-0050", "SYN-LAP-0036"},
    ),
    P(
        "max-price",
        "laptop under 50k 16gb",
        "nonempty",
        category=LAPTOP,
        max_price=Decimal("50000.00"),
        include={"SYN-LAP-0050"},
        exclude={"SYN-LAP-0002"},
    ),
    P(
        "min-price",
        "above 30k",
        "nonempty",
        min_price=Decimal("30000.00"),
        include={"SYN-LAP-0001"},
        exclude={"SYN-LAP-0050", "SYN-HDP-0001"},
    ),
    P(
        "exact-price-none",
        "from 20k ke andar",
        "empty",
        min_price=Decimal("20000.00"),
        max_price=Decimal("20000.00"),
    ),
    P("punctuation", "!!!", "nonempty", note="empty tsquery: lexical returns nothing"),
    # --- direct specs: boundaries and spec semantics the parser cannot express -------------
    D("empty-spec", "hp laptop", "nonempty"),
    D(
        "min-at",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        min_price=Decimal("28999.00"),
        include={"SYN-LAP-0050"},
    ),
    D(
        "min-above",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        min_price=Decimal("28999.01"),
        include={"SYN-LAP-0001"},
        exclude={"SYN-LAP-0050"},
    ),
    D(
        "max-at",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        max_price=Decimal("28999.00"),
        include={"SYN-LAP-0050"},
        exclude={"SYN-LAP-0001"},
    ),
    D(
        "max-below",
        "hp laptop",
        "empty",
        category=LAPTOP,
        brand="HP",
        max_price=Decimal("28998.99"),
    ),
    D(
        "min-eq-max",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        min_price=Decimal("28999.00"),
        max_price=Decimal("28999.00"),
        include={"SYN-LAP-0050"},
        exclude={"SYN-LAP-0001"},
    ),
    D(
        "price-window",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        brand="HP",
        min_price=Decimal("28999.01"),
        max_price=Decimal("33999.00"),
        include={"SYN-LAP-0001"},
        exclude={"SYN-LAP-0050"},
    ),
    D(
        "price-window-below",
        "hp laptop",
        "empty",
        category=LAPTOP,
        brand="HP",
        min_price=Decimal("33999.01"),
        max_price=Decimal("35998.99"),
    ),
    D("storage-not-ram", "8gb ram", "empty", storage_gb=8, note="8 is a RAM size only"),
    D("ram-not-storage-256", "256gb", "empty", ram_gb=256, note="256 is a storage size only"),
    D("ram-and-storage-8", "8gb ram", "empty", ram_gb=8, storage_gb=8),
    D(
        "storage-256",
        "256gb",
        "nonempty",
        storage_gb=256,
        include={"SYN-LAP-0001", "SYN-PHN-0005"},
        exclude={MISSING_LAPTOP_SPEC, MISSING_PHONE_SPEC, "SYN-LAP-0002"},
    ),
    D(
        "phone-no-spec-row",
        "samsung phone",
        "nonempty",
        category=PHONE,
        brand="Samsung",
        include={MISSING_PHONE_SPEC},
    ),
    D(
        "phone-storage",
        "samsung phone",
        "nonempty",
        brand="Samsung",
        storage_gb=256,
        include={"SYN-PHN-0013"},
        exclude={MISSING_PHONE_SPEC, "SYN-PHN-0001"},
    ),
    D(
        "hdd-direct",
        "laptop",
        "nonempty",
        storage_type=HDD,
        include={"SYN-LAP-0004"},
        exclude={NULL_STORAGE_TYPE_LAPTOP, "SYN-LAP-0001"},
    ),
    D(
        "nvme-ram-price",
        "hp laptop",
        "nonempty",
        category=LAPTOP,
        ram_gb=16,
        storage_type=SSD,
        storage_interface=NVME,
        max_price=Decimal("60000.00"),
        include={"SYN-LAP-0008"},
        exclude={"SYN-LAP-0002", NULL_RAM_LAPTOP},
    ),
    D("category-spec-mismatch", "laptop", "empty", category=SHOES, ram_gb=8),
    D(
        "headphones-price",
        "wireless headphones",
        "nonempty",
        category=HEADPHONES,
        max_price=Decimal("2099.00"),
        include={"SYN-HDP-0001", "SYN-HDP-0013"},
        exclude={"SYN-HDP-0002"},
    ),
    D(
        "brand-across-categories",
        "apple",
        "nonempty",
        brand="Apple",
        include={"SYN-LAP-0006", "SYN-PHN-0002"},
    ),
    D(
        "ram-12-laptop",
        "8gb ram",
        "nonempty",
        category=LAPTOP,
        ram_gb=12,
        include={"SYN-LAP-0015"},
        exclude={"SYN-LAP-0001"},
    ),
)
TRUTH_TABLE_IDS = [row.id for row in TRUTH_TABLE]


# ------------------------------------------------------------------------------ catalog state


def apply_mutations(engine) -> None:
    """Damage the scratch catalog deliberately; the FTS documents are left as ingested."""
    stale_sha = hashlib.sha256(b"m7-session-c-stale-embedding").hexdigest()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM laptop_specs WHERE product_id = :p"), {"p": MISSING_LAPTOP_SPEC}
        )
        conn.execute(
            text("DELETE FROM phone_specs WHERE product_id = :p"), {"p": MISSING_PHONE_SPEC}
        )
        conn.execute(
            text("UPDATE laptop_specs SET ram_gb = NULL WHERE product_id = :p"),
            {"p": NULL_RAM_LAPTOP},
        )
        conn.execute(
            text(
                "UPDATE laptop_specs SET storage_type = NULL, storage_interface = NULL "
                "WHERE product_id = :p"
            ),
            {"p": NULL_STORAGE_TYPE_LAPTOP},
        )
        conn.execute(
            text("UPDATE product_embeddings SET source_content_sha256 = :s WHERE product_id = :p"),
            {"s": stale_sha, "p": STALE_EMBEDDING},
        )
        conn.execute(
            text("DELETE FROM product_embeddings WHERE product_id = :p"), {"p": MISSING_EMBEDDING}
        )


def catalog_facts(engine) -> dict[str, dict]:
    """product_id -> core columns plus the laptop / phone spec row (or None)."""
    with engine.connect() as conn:
        facts = {
            row["product_id"]: {**dict(row), "laptop": None, "phone": None}
            for row in conn.execute(
                text("SELECT product_id, category, brand, price FROM products")
            ).mappings()
        }
        for row in conn.execute(
            text(
                "SELECT product_id, ram_gb, storage_gb, storage_type, storage_interface "
                "FROM laptop_specs"
            )
        ).mappings():
            facts[row["product_id"]]["laptop"] = dict(row)
        for row in conn.execute(
            text("SELECT product_id, ram_gb, storage_gb FROM phone_specs")
        ).mappings():
            facts[row["product_id"]]["phone"] = dict(row)
    return facts


def _plain(value):
    return value.value if hasattr(value, "value") else value


def satisfies(fact: dict, spec: FilterSpec) -> bool:
    """fp-1 eligibility of one product, written independently of the production SQL."""
    capacity_rows = [row for row in (fact["laptop"], fact["phone"]) if row is not None]
    checks = []
    if spec.category is not None:
        checks.append(fact["category"] == _plain(spec.category))
    if spec.brand is not None:
        checks.append(fact["brand"] == spec.brand)
    if spec.ram_gb is not None:
        checks.append(any(row["ram_gb"] == spec.ram_gb for row in capacity_rows))
    if spec.storage_gb is not None:
        checks.append(any(row["storage_gb"] == spec.storage_gb for row in capacity_rows))
    if spec.storage_type is not None:
        laptop = fact["laptop"]
        checks.append(laptop is not None and laptop["storage_type"] == _plain(spec.storage_type))
    if spec.storage_interface is not None:
        laptop = fact["laptop"]
        checks.append(
            laptop is not None and laptop["storage_interface"] == _plain(spec.storage_interface)
        )
    if spec.min_price is not None:
        checks.append(fact["price"] >= spec.min_price)
    if spec.max_price is not None:
        checks.append(fact["price"] <= spec.max_price)
    return all(checks)


def oracle_eligible(facts: dict[str, dict], spec: FilterSpec) -> set[str]:
    return {pid for pid, fact in facts.items() if satisfies(fact, spec)}


def current_embedding_ids(engine, spec: EmbeddingModelSpec) -> set[str]:
    """Products whose stored embedding is current for `spec` (M4 rule, checked in Python)."""
    with engine.connect() as conn:
        content = dict(conn.execute(text("SELECT product_id, content_sha256 FROM products")).all())
        rows = conn.execute(
            text(
                "SELECT product_id, model_id, model_revision, embedding_config_sha256, "
                "embedding_text_version, dimension, normalized, source_content_sha256 "
                "FROM product_embeddings"
            )
        ).mappings()
        wanted = (
            spec.model_id,
            spec.revision,
            spec.config_sha256(),
            EMBEDDING_TEXT_VERSION,
            spec.dimension,
            spec.normalize,
        )
        return {
            row["product_id"]
            for row in rows
            if (
                row["model_id"],
                row["model_revision"],
                row["embedding_config_sha256"],
                row["embedding_text_version"],
                row["dimension"],
                row["normalized"],
            )
            == wanted
            and row["source_content_sha256"] == content[row["product_id"]]
        }
