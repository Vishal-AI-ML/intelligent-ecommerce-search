import pytest

from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.findings import CheckOutcome, Severity
from ecommerce_search.catalog.quality.rules import CHECKS, run_checks


def outcome(check, *records, synthetic=True, external=None):
    flags = {r.product_id: synthetic for r in records}
    return run_checks(list(records), flags, external or {})[check]


def messages(check, *records, **kwargs):
    return [f.message for f in outcome(check, *records, **kwargs).findings]


def severities(check, *records, **kwargs):
    return {f.severity for f in outcome(check, *records, **kwargs).findings}


# ---- errors -----------------------------------------------------------------------------


def test_unique_product_id(parsed):
    assert not outcome("unique_product_id", parsed("laptop"), parsed("phone")).findings
    dup = outcome("unique_product_id", parsed("laptop"), parsed("laptop", title="Other HP laptop"))
    assert len(dup.findings) == 2 and severities(
        "unique_product_id", parsed("laptop"), parsed("laptop")
    ) == {Severity.ERROR}


def test_provenance_flags(parsed):
    assert not outcome("provenance_flags", parsed("laptop")).findings
    bad = outcome("provenance_flags", parsed("laptop"), synthetic=False)
    assert [f.severity for f in bad.findings] == [Severity.ERROR]


def test_exact_duplicate_requires_identical_content_except_identity(parsed):
    a = parsed("laptop", product_id="A")
    b = parsed("laptop", product_id="B")  # identical in every field except product_id
    found = outcome("exact_duplicate", a, b).findings
    assert len(found) == 2 and {f.severity for f in found} == {Severity.ERROR}
    assert not outcome("exact_duplicate", a).findings
    # a whitespace/case-insensitive title match alone is NOT an exact duplicate
    case_only = parsed("laptop", product_id="C", title=a.title.upper() + "  ")
    assert not outcome("exact_duplicate", a, case_only).findings


@pytest.mark.parametrize(
    "overrides",
    [
        {"seller_id": "other-seller"},
        {"price": 99999},
        {"rating": 3.0, "review_count": 3},
        {"description": "Different text."},
        {"ram_gb": 8},
    ],
)
def test_same_title_with_different_content_is_a_warning_not_an_error(parsed, overrides):
    a = parsed("laptop", product_id="A")
    b = parsed("laptop", product_id="B", **overrides)
    assert not outcome("exact_duplicate", a, b).findings  # never blocks ingestion
    found = outcome("same_title_listing", a, b).findings
    assert len(found) == 2 and {f.severity for f in found} == {Severity.WARNING}
    assert "different content" in found[0].message


def test_same_title_rule_negatives(parsed):
    a = parsed("laptop", product_id="A")
    # different brand (and different category) are never duplicates, even with the same title
    other_brand = parsed("laptop", product_id="B", brand="Dell", title=a.title)
    assert not outcome("same_title_listing", a, other_brand).findings
    assert not outcome("exact_duplicate", a, other_brand).findings
    other_category = parsed("phone", product_id="C", title=a.title, brand="HP")
    assert not outcome("same_title_listing", a, other_category).findings
    # a record is not a duplicate of itself, and exact-content groups are errors not warnings
    assert not outcome("same_title_listing", a).findings
    assert not outcome("same_title_listing", a, parsed("laptop", product_id="Z")).findings


def test_near_duplicate_threshold_boundary_is_inclusive(parsed):
    base = "HP " + " ".join(f"w{i}" for i in range(17))  # 18 tokens
    a = parsed("laptop", product_id="A", title=base + " x1")  # 19 tokens
    b = parsed("laptop", product_id="B", title=base + " x2")  # Jaccard 18/20 = 0.9
    c = parsed("laptop", product_id="C", title=base + " y1 y2")  # 18/21 < 0.9 vs a
    assert len(outcome("near_duplicate", a, b).findings) == 2
    assert not outcome("near_duplicate", a, c).findings


def test_exact_and_title_groups_feed_the_duplicate_share(parsed):
    a, b = parsed("laptop", product_id="A"), parsed("laptop", product_id="B")
    finding = outcome("duplicate_share", a, b, parsed("phone")).findings[0]
    assert finding.count == 2 and finding.severity is Severity.INFO


def test_schema_has_no_label_fields_and_detects_them(monkeypatch):
    clean = run_checks([], {}, {})["schema_no_label_fields"]
    assert clean.evaluated > 0 and not clean.findings
    monkeypatch.setattr(qp, "FORBIDDEN_LABEL_FIELDS", qp.FORBIDDEN_LABEL_FIELDS | {"processor"})
    dirty = run_checks([], {}, {})["schema_no_label_fields"]
    assert dirty.findings and dirty.findings[0].severity is Severity.ERROR


# ---- warnings ---------------------------------------------------------------------------


def test_common_value_check_warns_but_never_errors(parsed):
    assert not outcome("plausibility_common_values", parsed("laptop")).findings
    unusual = parsed("laptop", ram_gb=48, title="HP 48GB RAM laptop")
    result = outcome("plausibility_common_values", unusual)
    assert [f.severity for f in result.findings] == [Severity.WARNING]
    assert "unusual" in result.findings[0].message
    assert severities("plausibility_common_values", parsed("phone", storage_gb=100)) == {
        Severity.WARNING
    }


def test_plausibility_ranges(parsed):
    assert not outcome(
        "plausibility_ranges", parsed("laptop"), parsed("phone"), parsed("shoes")
    ).findings
    assert messages("plausibility_ranges", parsed("laptop", screen_size_inches=30.0))
    assert messages("plausibility_ranges", parsed("laptop", weight_kg=9.0))
    assert messages("plausibility_ranges", parsed("phone", battery_mah=500))
    assert messages("plausibility_ranges", parsed("shoes", size=30, size_system="UK"))
    assert not messages("plausibility_ranges", parsed("shoes", size=42, size_system="EU"))
    assert messages("plausibility_ranges", parsed("headphones", battery_life_hours=500))


def test_price_band(parsed):
    assert not outcome("price_band", parsed("laptop")).findings
    assert severities("price_band", parsed("laptop", price=500)) == {Severity.WARNING}
    assert severities("price_band", parsed("shoes", price=900000)) == {Severity.WARNING}
    assert "not assessed" in messages("price_band", parsed("laptop", currency="USD"))[0]


def test_brand_category_coherence(parsed):
    assert not outcome("brand_category", parsed("laptop")).findings
    assert severities("brand_category", parsed("laptop", brand="Nike", title="Nike laptop")) == {
        Severity.WARNING
    }
    unknown = parsed("laptop", brand="Zorbo", title="Zorbo laptop")
    assert outcome("brand_category", unknown).evaluated == 0
    info = outcome("brand_dictionary", unknown)
    assert info.findings[0].severity is Severity.INFO and info.findings[0].count == 1


@pytest.mark.parametrize(
    ("category", "overrides"),
    [
        ("laptop", {"ram_gb": 8}),
        ("laptop", {"storage_gb": 256}),
        ("laptop", {"storage_type": "HDD", "storage_interface": "SATA"}),
        ("laptop", {"storage_interface": "SATA"}),  # title claims nothing about interface
        ("phone", {"ram_gb": 4}),
        ("phone", {"storage_gb": 64}),
        ("shoes", {"title": "Nike Runner 8GB RAM Shoes"}),
        ("headphones", {"wireless": False, "connectivity": "wired"}),
        ("headphones", {"title": "Sony Wired Headphones", "wireless": True}),
        ("headphones", {"title": "Sony Noise Cancelling Headphones", "anc": False}),
        ("shoes", {"gender": "women"}),
        ("laptop", {"brand": "Dell"}),
    ],
)
def test_title_attribute_contradictions_are_warnings(parsed, category, overrides):
    record = parsed(category, **overrides)
    if category == "laptop" and overrides == {"storage_interface": "SATA"}:
        record = parsed(category, title="HP Laptop 512GB NVMe", **overrides)
    assert severities("title_attribute_contradiction", record) == {Severity.WARNING}


def test_title_check_passes_for_consistent_titles(parsed):
    for category in ("laptop", "phone", "shoes", "headphones"):
        assert not outcome("title_attribute_contradiction", parsed(category)).findings
    tb = parsed("laptop", title="HP Laptop 1TB SSD", storage_gb="1TB")
    assert not outcome("title_attribute_contradiction", tb).findings


def test_near_duplicates(parsed):
    long = "HP Alpha 15.6-inch Laptop Intel Core i5 16GB RAM 512GB SSD Windows 11 Home Silver Edition Model"
    a = parsed("laptop", product_id="A", title=f"{long} X1")
    b = parsed("laptop", product_id="B", title=f"{long} X2")
    result = outcome("near_duplicate", a, b)
    assert len(result.findings) == 2 and {f.severity for f in result.findings} == {Severity.WARNING}
    far = parsed("laptop", product_id="C", title="HP Beta ultraportable Ryzen 7 32GB RAM 1TB SSD")
    assert not outcome("near_duplicate", a, far).findings
    dup_share = outcome("duplicate_share", a, b, far).findings[0]
    assert dup_share.severity is Severity.INFO and dup_share.count == 2


def test_rating_without_reviews(parsed):
    assert not outcome("rating_without_reviews", parsed("laptop")).findings
    assert severities("rating_without_reviews", parsed("laptop", rating=4.0, review_count=0)) == {
        Severity.WARNING
    }
    assert not outcome(
        "rating_without_reviews", parsed("laptop", rating=None, review_count=0)
    ).findings


def test_missing_expected_attributes_warn_but_optional_ones_only_inform(parsed):
    assert not outcome("missing_expected_attributes", parsed("laptop")).findings
    missing = outcome("missing_expected_attributes", parsed("laptop", processor=None, ram_gb=None))
    assert {f.field for f in missing.findings} == {"processor", "ram_gb"}
    assert {f.severity for f in missing.findings} == {Severity.WARNING}
    optional_only = parsed("laptop", gpu=None, weight_kg=None)
    assert not outcome("missing_expected_attributes", optional_only).findings
    info = outcome("optional_field_completeness", optional_only).findings
    assert {f.field for f in info} == {"gpu", "weight_kg"} and {f.severity for f in info} == {
        Severity.INFO
    }


# ---- info -------------------------------------------------------------------------------


def test_missing_seller_id_is_info(parsed):
    assert not outcome("missing_seller_id", parsed("laptop")).findings
    finding = outcome("missing_seller_id", parsed("laptop", seller_id=None)).findings[0]
    assert finding.severity is Severity.INFO and finding.count == 1


# ---- structure of the check set ---------------------------------------------------------


def test_every_check_has_a_severity_and_unique_name():
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))
    assert {c.severity for c in CHECKS} == set(Severity)


def test_static_and_missing_external_checks_are_not_applicable_not_pass(parsed):
    result = run_checks([parsed("laptop")], {}, {})
    leakage = result["embedding_search_text_leakage"]
    assert leakage.evaluated == 0 and "Not applicable" in leakage.note
    assert result["spec_linkage"].evaluated == 0


def test_external_outcomes_are_used(parsed):
    supplied = CheckOutcome(3, ())
    assert run_checks([], {}, {"schema_validation": supplied})["schema_validation"] is supplied


def test_provisional_parameters_are_versioned_and_complete():
    params = qp.parameters_as_dict()
    assert params["rules_version"] == qp.RULES_VERSION
    assert params["near_duplicate_jaccard_threshold"] == "0.9"
    assert params["price_bands_inr"]["laptop"] == ["10000", "500000"]
    assert "not measured marketplace facts" in params["kind"]
