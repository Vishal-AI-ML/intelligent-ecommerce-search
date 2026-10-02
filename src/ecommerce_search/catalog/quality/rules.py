"""Catalog quality rules.

Each check is a pure function over validated `CatalogRecord`s. The same code runs on records
loaded from a file (before persistence) and on records rebuilt from the database (audit).
Checks that need raw text or database structure are computed by the loaders and passed in as
pre-computed outcomes. A check that evaluated nothing is reported `not_applicable`, never
`pass`.
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.findings import CheckOutcome, Finding, Severity
from ecommerce_search.catalog.schemas import CatalogRecord
from ecommerce_search.catalog.taxonomy import Category

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO  # noqa: E741


@dataclass(frozen=True)
class CheckSpec:
    name: str
    severity: Severity
    description: str
    # "record": computed by run_checks from records; "external": supplied by a loader;
    # "static": not applicable unless a database audit supplies an outcome.
    kind: str
    fn: Callable[["Context"], CheckOutcome] | None = None
    # Note used when a "static" check is not applicable (file reports).
    not_applicable_note: str | None = None


@dataclass(frozen=True)
class Context:
    records: Sequence[CatalogRecord]
    dataset_synthetic: Mapping[str, bool]  # product_id -> provenance is_synthetic


def _f(check: str, sev: Severity, record: CatalogRecord | None, message: str, **kw) -> Finding:
    return Finding(
        check=check,
        severity=sev,
        message=message,
        product_id=record.product_id if record else None,
        **kw,
    )


def _title_key(title: str) -> str:
    return " ".join(title.casefold().split())


def _tokens(title: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", title.casefold()))


# ---- error-level record checks ----------------------------------------------------------


def unique_product_id(ctx: Context) -> CheckOutcome:
    counts = Counter(r.product_id for r in ctx.records)
    findings = [
        _f("unique_product_id", E, r, f"product_id appears {counts[r.product_id]} times")
        for r in ctx.records
        if counts[r.product_id] > 1
    ]
    return CheckOutcome(len(ctx.records), tuple(findings))


def provenance_flags(ctx: Context) -> CheckOutcome:
    findings = []
    for r in ctx.records:
        expected = ctx.dataset_synthetic.get(r.product_id)
        if expected is not None and r.is_synthetic != expected:
            findings.append(
                _f(
                    "provenance_flags",
                    E,
                    r,
                    f"record is_synthetic={r.is_synthetic} but its dataset is_synthetic={expected}",
                    field="is_synthetic",
                )
            )
    return CheckOutcome(len(ctx.records), tuple(findings))


def _content_key(record: CatalogRecord) -> str:
    """Canonical normalized content with the identity (product_id) removed."""
    content = record.model_dump(mode="json")
    content.pop("product_id")
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


def exact_duplicate(ctx: Context) -> CheckOutcome:
    """Error: two product ids whose normalized content is identical in every other field,
    seller included. The identity is contradictory because nothing distinguishes the listings.
    A shared title alone is not an error (see same_title_listing)."""
    groups: dict[str, list[CatalogRecord]] = defaultdict(list)
    for r in ctx.records:
        groups[_content_key(r)].append(r)
    findings = []
    for members in groups.values():
        ids = sorted({m.product_id for m in members})
        if len(ids) > 1:
            findings.extend(
                _f(
                    "exact_duplicate",
                    E,
                    m,
                    "identical content (every field except product_id) as "
                    + ", ".join(i for i in ids if i != m.product_id),
                )
                for m in members
            )
    return CheckOutcome(len(ctx.records), tuple(findings))


def _title_groups(records: Sequence[CatalogRecord]) -> list[list[CatalogRecord]]:
    """Groups of records with the same category, brand and normalized title (distinct ids)."""
    groups: dict[tuple, list[CatalogRecord]] = defaultdict(list)
    for r in records:
        groups[(r.category, r.brand, _title_key(r.title))].append(r)
    return [g for g in groups.values() if len({m.product_id for m in g}) > 1]


def same_title_listing(ctx: Context) -> CheckOutcome:
    """Warning: same category, brand and normalized title but different meaningful content
    (seller, price, ...). Legitimate multi-seller listings look like this; different brands are
    never grouped. Groups whose members are all exact-content duplicates are reported as errors
    by exact_duplicate instead."""
    findings = []
    for group in _title_groups(ctx.records):
        if len({_content_key(m) for m in group}) == 1:
            continue
        ids = sorted(m.product_id for m in group)
        findings.extend(
            _f(
                "same_title_listing",
                W,
                m,
                "same category, brand and normalized title as "
                + ", ".join(i for i in ids if i != m.product_id)
                + " but different content",
                field="title",
            )
            for m in group
        )
    return CheckOutcome(len(ctx.records), tuple(findings))


def schema_no_label_fields(_ctx: Context) -> CheckOutcome:
    """Structural check: catalog models/tables carry no evaluation-label or review fields."""
    from ecommerce_search.catalog.schemas import SPEC_MODELS, CoreFields
    from ecommerce_search.models.catalog import SPEC_TABLES, Product

    inspected = 0
    findings = []

    def scan(owner: str, names) -> None:
        nonlocal inspected
        for name in names:
            inspected += 1
            if name.casefold() in qp.FORBIDDEN_LABEL_FIELDS:
                findings.append(
                    Finding(
                        check="schema_no_label_fields",
                        severity=E,
                        message=f"{owner} has forbidden label/review field {name!r}",
                        field=name,
                    )
                )

    scan("CoreFields", CoreFields.model_fields)
    for category, model in SPEC_MODELS.items():
        scan(f"{category.value} spec model", model.model_fields)
    scan("products table", (c.name for c in Product.__table__.columns))
    for category, table in SPEC_TABLES.items():
        scan(f"{category.value} spec table", (c.name for c in table.__table__.columns))
    return CheckOutcome(inspected, tuple(findings))


# ---- warning-level record checks --------------------------------------------------------


def plausibility_common_values(ctx: Context) -> CheckOutcome:
    findings, evaluated = [], 0
    for r in ctx.records:
        for field_name, table in (
            ("ram_gb", qp.COMMON_RAM_GB),
            ("storage_gb", qp.COMMON_STORAGE_GB),
        ):
            common = table.get(r.category)
            value = getattr(r.spec, field_name, None)
            if common is None or value is None:
                continue
            evaluated += 1
            if value not in common:
                findings.append(
                    _f(
                        "plausibility_common_values",
                        W,
                        r,
                        f"{field_name}={value} is outside the common {r.category.value} values "
                        f"{sorted(common)} (unusual, not necessarily invalid)",
                        field=field_name,
                    )
                )
    return CheckOutcome(evaluated, tuple(findings))


def plausibility_ranges(ctx: Context) -> CheckOutcome:
    findings, evaluated = [], 0
    for r in ctx.records:
        ranges: list[tuple[str, Decimal, Decimal]] = [
            (f, lo, hi) for (c, f), (lo, hi) in qp.PLAUSIBLE_RANGES.items() if c is r.category
        ]
        if r.category is Category.SHOES:
            system = getattr(r.spec, "size_system", None)
            if system is not None:
                lo, hi = qp.SHOE_SIZE_RANGES[system]
                ranges.append(("size", lo, hi))
        for field_name, lo, hi in ranges:
            value = getattr(r.spec, field_name, None)
            if value is None:
                continue
            evaluated += 1
            if not lo <= Decimal(value) <= hi:
                findings.append(
                    _f(
                        "plausibility_ranges",
                        W,
                        r,
                        f"{field_name}={value} is outside the plausible range [{lo}, {hi}]",
                        field=field_name,
                    )
                )
    return CheckOutcome(evaluated, tuple(findings))


def price_band(ctx: Context) -> CheckOutcome:
    findings = []
    for r in ctx.records:
        if r.currency != qp.EXPECTED_CURRENCY:
            findings.append(
                _f(
                    "price_band",
                    W,
                    r,
                    f"currency {r.currency} differs from {qp.EXPECTED_CURRENCY}; band not assessed",
                    field="currency",
                )
            )
            continue
        lo, hi = qp.PRICE_BANDS_INR[r.category]
        if not lo <= r.price <= hi:
            findings.append(
                _f(
                    "price_band",
                    W,
                    r,
                    f"price {r.price} is outside the {r.category.value} sanity band [{lo}, {hi}]",
                    field="price",
                )
            )
    return CheckOutcome(len(ctx.records), tuple(findings))


def brand_category(ctx: Context) -> CheckOutcome:
    findings, evaluated = [], 0
    for r in ctx.records:
        allowed = tx.brand_categories(r.brand)
        if allowed is None:
            continue
        evaluated += 1
        if r.category not in allowed:
            findings.append(
                _f(
                    "brand_category",
                    W,
                    r,
                    f"brand {r.brand} is not expected in category {r.category.value}",
                    field="brand",
                )
            )
    return CheckOutcome(evaluated, tuple(findings))


_RAM_RE = re.compile(r"(\d+)\s*gb\s*ram")
_STORAGE_RE = re.compile(r"(\d+)\s*(gb|tb)\s*(ssd|hdd|nvme|storage|rom)")
_GENDER_RE = re.compile(r"\b(women|men|unisex|kids)\b")


def title_attribute_contradiction(ctx: Context) -> CheckOutcome:
    """Conservative: only explicit `N GB RAM` / `N GB|TB SSD|HDD|NVMe|storage` patterns and a
    few explicit words are compared; nothing is inferred from a title."""
    findings = []

    def warn(r: CatalogRecord, message: str, field_name: str) -> None:
        findings.append(_f("title_attribute_contradiction", W, r, message, field=field_name))

    for r in ctx.records:
        title = _title_key(r.title)
        if r.brand.casefold() not in title:
            warn(r, f"brand {r.brand!r} does not appear in the title", "brand")
        ram, storage = _RAM_RE.search(title), _STORAGE_RE.search(title)
        if r.category in (Category.LAPTOP, Category.PHONE):
            if ram and r.spec.ram_gb is not None and int(ram[1]) != r.spec.ram_gb:
                warn(r, f"title says {ram[1]} GB RAM but ram_gb={r.spec.ram_gb}", "ram_gb")
            if storage:
                title_gb = int(storage[1]) * (1024 if storage[2] == "tb" else 1)
                if r.spec.storage_gb is not None and title_gb != r.spec.storage_gb:
                    warn(
                        r,
                        f"title says {title_gb} GB storage but storage_gb={r.spec.storage_gb}",
                        "storage_gb",
                    )
                kind = storage[3]
                stype = getattr(r.spec, "storage_type", None)
                if kind in ("ssd", "hdd") and stype is not None and stype.value.lower() != kind:
                    warn(
                        r,
                        f"title says {kind.upper()} but storage_type={stype.value}",
                        "storage_type",
                    )
                interface = getattr(r.spec, "storage_interface", None)
                if (
                    kind == "nvme"
                    and interface is not None
                    and interface is not tx.StorageInterface.NVME
                ):
                    warn(
                        r,
                        f"title says NVMe but storage_interface={interface.value}",
                        "storage_interface",
                    )
        elif ram or storage:
            warn(r, f"title mentions RAM/storage but category is {r.category.value}", "title")
        if r.category is Category.HEADPHONES:
            wireless = r.spec.wireless
            if re.search(r"\bwireless\b", title) and wireless is False:
                warn(r, "title says wireless but wireless=false", "wireless")
            if re.search(r"\bwired\b", title) and wireless is True:
                warn(r, "title says wired but wireless=true", "wireless")
            if "noise cancel" in title and r.spec.anc is False:
                warn(r, "title mentions noise cancelling but anc=false", "anc")
        if r.category is Category.SHOES and r.spec.gender is not None:
            found = set(_GENDER_RE.findall(title))
            if len(found) == 1 and r.spec.gender.value not in found:
                warn(
                    r, f"title says {next(iter(found))} but gender={r.spec.gender.value}", "gender"
                )
    return CheckOutcome(len(ctx.records), tuple(findings))


def _near_duplicate_pairs(
    records: Sequence[CatalogRecord],
) -> list[tuple[CatalogRecord, CatalogRecord, Decimal]]:
    groups: dict[tuple, list[CatalogRecord]] = defaultdict(list)
    for r in records:
        groups[(r.category, r.brand)].append(r)
    pairs = []
    for members in groups.values():
        members = sorted(members, key=lambda m: m.product_id)
        tokens = [_tokens(m.title) for m in members]
        for i, a in enumerate(members):
            for j in range(i + 1, len(members)):
                b = members[j]
                if _title_key(a.title) == _title_key(b.title):
                    continue  # exact duplicates are an error elsewhere
                union = tokens[i] | tokens[j]
                if not union:
                    continue
                jaccard = Decimal(len(tokens[i] & tokens[j])) / Decimal(len(union))
                if jaccard >= qp.NEAR_DUPLICATE_JACCARD:
                    pairs.append((a, b, jaccard))
    return pairs


def near_duplicate(ctx: Context) -> CheckOutcome:
    findings = []
    for a, b, jaccard in _near_duplicate_pairs(ctx.records):
        for x, y in ((a, b), (b, a)):
            findings.append(
                _f(
                    "near_duplicate",
                    W,
                    x,
                    f"title token Jaccard {jaccard:.3f} with {y.product_id} "
                    f"(threshold {qp.NEAR_DUPLICATE_JACCARD})",
                    field="title",
                )
            )
    return CheckOutcome(len(ctx.records), tuple(findings))


def rating_without_reviews(ctx: Context) -> CheckOutcome:
    findings = [
        _f(
            "rating_without_reviews",
            W,
            r,
            f"rating {r.rating} present with review_count=0",
            field="rating",
        )
        for r in ctx.records
        if r.rating is not None and r.review_count == 0
    ]
    return CheckOutcome(len(ctx.records), tuple(findings))


def missing_expected_attributes(ctx: Context) -> CheckOutcome:
    findings = []
    for r in ctx.records:
        for name in qp.EXPECTED_FIELDS[r.category]:
            if getattr(r.spec, name) is None:
                findings.append(
                    _f(
                        "missing_expected_attributes",
                        W,
                        r,
                        f"expected {r.category.value} attribute {name} is missing",
                        field=name,
                    )
                )
    return CheckOutcome(len(ctx.records), tuple(findings))


# ---- info-level (aggregated) checks -----------------------------------------------------


def _info(check: str, message: str, count: int, field_name: str | None = None) -> Finding:
    return Finding(check=check, severity=I, message=message, field=field_name, count=count)


def optional_field_completeness(ctx: Context) -> CheckOutcome:
    findings = []
    by_category = defaultdict(list)
    for r in ctx.records:
        by_category[r.category].append(r)
    for category in Category:
        members = by_category.get(category, [])
        if not members:
            continue
        names = [(n, lambda r, n=n: getattr(r, n)) for n in qp.OPTIONAL_CORE_FIELDS] + [
            (n, lambda r, n=n: getattr(r.spec, n)) for n in qp.OPTIONAL_FIELDS[category]
        ]
        for name, getter in names:
            missing = sum(1 for r in members if getter(r) is None)
            if missing:
                findings.append(
                    _info(
                        "optional_field_completeness",
                        f"{category.value}: {name} missing on {missing} of {len(members)} records",
                        missing,
                        name,
                    )
                )
    return CheckOutcome(len(ctx.records), tuple(findings))


def missing_seller_id(ctx: Context) -> CheckOutcome:
    missing = sum(1 for r in ctx.records if r.seller_id is None)
    findings = (
        (
            _info(
                "missing_seller_id",
                f"seller_id missing on {missing} of {len(ctx.records)} records",
                missing,
                "seller_id",
            ),
        )
        if missing
        else ()
    )
    return CheckOutcome(len(ctx.records), findings)


def brand_dictionary(ctx: Context) -> CheckOutcome:
    unknown = Counter(r.brand for r in ctx.records if tx.brand_categories(r.brand) is None)
    findings = tuple(
        _info(
            "brand_dictionary",
            f"brand {brand!r} is not in the brand dictionary ({n} records); "
            "coherence not assessable",
            n,
            "brand",
        )
        for brand, n in sorted(unknown.items())
    )
    return CheckOutcome(len(ctx.records), findings)


def duplicate_share(ctx: Context) -> CheckOutcome:
    in_group = {p.product_id for a, b, _ in _near_duplicate_pairs(ctx.records) for p in (a, b)}
    in_group |= {m.product_id for group in _title_groups(ctx.records) for m in group}
    content = Counter(_content_key(r) for r in ctx.records)
    in_group |= {r.product_id for r in ctx.records if content[_content_key(r)] > 1}
    total = len(ctx.records)
    pct = (100 * len(in_group) / total) if total else 0.0
    return CheckOutcome(
        total,
        (
            _info(
                "duplicate_share",
                f"{len(in_group)} of {total} records ({pct:.1f}%) are in an exact, same-title "
                "or near-duplicate group",
                len(in_group),
            ),
        ),
    )


NOT_APPLICABLE_DENSE_NOTE = (
    "Not applicable for file reports: product embeddings are derived rows that exist only in a "
    "database (Milestone 4). Database audits evaluate them: missing or stale rows are warnings; "
    "malformed, tampered or invalid rows are errors."
)

NOT_APPLICABLE_LEAKAGE_NOTE = (
    "Not applicable for file reports: search documents are derived rows that exist only in a "
    "database. Database audits evaluate them (missing, stale, source-hash mismatch, rebuilt-vector "
    "equality and forbidden review/provenance text). Structural separation is also covered by "
    "schema_no_label_fields and evaluation_label_fields."
)

# Order is the report order. "external" checks are computed by the file/database loaders.
CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        "catalog_non_empty",
        E,
        "The catalog contains at least one record (an empty catalog is an error, never a pass)",
        "external",
    ),
    CheckSpec(
        "schema_validation",
        E,
        "Every record parses and satisfies types, required values, "
        "hard bounds, taxonomy values and category/attribute coherence (incl. NVME requires SSD)",
        "external",
    ),
    CheckSpec("unique_product_id", E, "Product IDs are unique", "record", unique_product_id),
    CheckSpec(
        "provenance_flags",
        E,
        "Record synthetic flag agrees with its dataset provenance",
        "record",
        provenance_flags,
    ),
    CheckSpec(
        "exact_duplicate",
        E,
        "No two product ids have identical content in every other field",
        "record",
        exact_duplicate,
    ),
    CheckSpec(
        "evaluation_label_fields",
        E,
        "Seed/raw records contain no evaluation-label or review fields",
        "external",
    ),
    CheckSpec(
        "schema_no_label_fields",
        E,
        "Catalog schemas and tables contain no evaluation-label or review fields",
        "record",
        schema_no_label_fields,
    ),
    CheckSpec(
        "spec_linkage",
        E,
        "Every product has exactly one attribute row of its category (database audit only)",
        "external",
    ),
    CheckSpec(
        "raw_normalized_consistency",
        E,
        "Stored normalized values equal a fresh normalization of the stored raw line "
        "(database audit only)",
        "external",
    ),
    CheckSpec(
        "plausibility_common_values",
        W,
        "RAM/storage within common-value sets (uncommon is a warning only)",
        "record",
        plausibility_common_values,
    ),
    CheckSpec(
        "plausibility_ranges",
        W,
        "Screen, weight, battery and shoe size within plausible ranges",
        "record",
        plausibility_ranges,
    ),
    CheckSpec("price_band", W, "Price within the category sanity band", "record", price_band),
    CheckSpec(
        "brand_category", W, "Known brand plausible for the category", "record", brand_category
    ),
    CheckSpec(
        "title_attribute_contradiction",
        W,
        "Title does not contradict explicit attributes",
        "record",
        title_attribute_contradiction,
    ),
    CheckSpec(
        "same_title_listing",
        W,
        "Same category, brand and normalized title with different content (e.g. other seller)",
        "record",
        same_title_listing,
    ),
    CheckSpec(
        "near_duplicate",
        W,
        "No near-duplicate titles within a brand and category",
        "record",
        near_duplicate,
    ),
    CheckSpec(
        "rating_without_reviews",
        W,
        "Rating is not present with zero reviews",
        "record",
        rating_without_reviews,
    ),
    CheckSpec(
        "missing_expected_attributes",
        W,
        "Expected category attributes are present",
        "record",
        missing_expected_attributes,
    ),
    CheckSpec(
        "optional_field_completeness",
        I,
        "Optional-field completeness",
        "record",
        optional_field_completeness,
    ),
    CheckSpec(
        "missing_seller_id", I, "Nullable seller_id completeness", "record", missing_seller_id
    ),
    CheckSpec(
        "brand_dictionary", I, "Brands absent from the brand dictionary", "record", brand_dictionary
    ),
    CheckSpec(
        "duplicate_share",
        I,
        "Share of records in exact/same-title/near-duplicate groups",
        "record",
        duplicate_share,
    ),
    CheckSpec(
        "embedding_search_text_leakage",
        E,
        "Embedding/search text contains no evaluation labels",
        "static",
    ),
    CheckSpec(
        "dense_embedding_consistency",
        E,
        "Product embeddings are current, well-formed and built from the whitelisted embedding text",
        "static",
        not_applicable_note=NOT_APPLICABLE_DENSE_NOTE,
    ),
)


def run_checks(
    records: Sequence[CatalogRecord],
    dataset_synthetic: Mapping[str, bool],
    external: Mapping[str, CheckOutcome],
) -> dict[str, CheckOutcome]:
    """Run every check. Missing external outcomes and static checks become not-applicable."""
    ctx = Context(records=records, dataset_synthetic=dataset_synthetic)
    outcomes: dict[str, CheckOutcome] = {}
    for spec in CHECKS:
        if spec.kind == "record" and spec.fn is not None:
            outcomes[spec.name] = spec.fn(ctx)
        elif spec.kind == "static":
            # Not applicable unless a database audit supplies a real outcome.
            note = spec.not_applicable_note or NOT_APPLICABLE_LEAKAGE_NOTE
            outcomes[spec.name] = external.get(spec.name, CheckOutcome(0, (), note))
        else:
            outcomes[spec.name] = external.get(
                spec.name, CheckOutcome(0, (), "not computed for this source")
            )
    return outcomes
