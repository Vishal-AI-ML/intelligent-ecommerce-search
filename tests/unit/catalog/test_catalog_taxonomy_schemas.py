from decimal import Decimal

import pytest

from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.catalog.schemas import (
    RecordValidationError,
    parse_raw,
    record_content_hash,
)


def errors_of(raw) -> dict[str, str]:
    with pytest.raises(RecordValidationError) as excinfo:
        parse_raw(raw)
    return dict(excinfo.value.errors)


# ---- taxonomy and brand normalization ---------------------------------------------------


def test_brand_normalization_is_case_insensitive_and_keeps_unknown_brands():
    assert tx.canonical_brand("hp") == "HP"
    assert tx.canonical_brand("BOAT") == "boAt"
    assert tx.canonical_brand("one plus") == "OnePlus"
    assert tx.canonical_brand("Hewlett-Packard") == "HP"
    assert tx.canonical_brand("Unknown Brand") == "Unknown Brand"
    assert tx.brand_categories("Unknown Brand") is None


def test_every_brand_has_categories_and_every_category_has_subcategories():
    assert all(cats for cats in tx.BRAND_CATEGORIES.values())
    assert set(tx.SUBCATEGORIES) == set(tx.Category)


def test_nvme_is_an_interface_not_a_storage_medium():
    assert {m.value for m in tx.StorageType} == {"SSD", "HDD"}
    assert {m.value for m in tx.StorageInterface} == {"NVME", "SATA"}


# ---- per-category validation ------------------------------------------------------------


@pytest.mark.parametrize("category", ["laptop", "phone", "shoes", "headphones"])
def test_valid_record_parses_for_every_category(parsed, category):
    record = parsed(category)
    assert record.category.value == category
    assert record.is_synthetic and record.source_type is tx.SourceType.SYNTHETIC


def test_brand_and_whitespace_are_normalized(parsed):
    record = parsed("laptop", brand="  hp ", title="  HP   Test  Laptop ")
    assert record.brand == "HP"
    assert record.title == "HP Test Laptop"


@pytest.mark.parametrize(
    ("raw_price", "expected"),
    [
        (54999, "54999.00"),
        ("54,999", "54999.00"),
        ("Rs 54,999", "54999.00"),
        ("rs.1,299.50", "1299.50"),
        ("₹54,999", "54999.00"),
        (999.5, "999.50"),
    ],
)
def test_price_normalization(parsed, raw_price, expected):
    assert parsed("laptop", price=raw_price).price == Decimal(expected)


@pytest.mark.parametrize(
    ("ram", "storage", "expected"),
    [
        (16, 512, (16, 512)),
        ("16 GB", "512GB", (16, 512)),
        ("8gb", "1TB", (8, 1024)),
        ("8 GB", "0.5 TB", (8, 512)),
        ("8", "256", (8, 256)),
    ],
)
def test_unit_and_storage_normalization(parsed, ram, storage, expected):
    record = parsed("laptop", ram_gb=ram, storage_gb=storage)
    assert (record.spec.ram_gb, record.spec.storage_gb) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ram_gb", "16 MB"),
        ("ram_gb", "1TB"),  # RAM is GB only
        ("ram_gb", "8 GiB"),
        ("ram_gb", "eight"),
        ("ram_gb", 8.5),
        ("ram_gb", True),
        ("storage_gb", "512 mb"),
        ("storage_gb", "0.3 TB"),  # not a whole number of GB
        ("screen_size_inches", "15.6 cm"),
        ("weight_kg", "1.7 lb"),
        ("screen_size_inches", 15.66),  # too many decimal places
    ],
)
def test_unknown_units_and_bad_numbers_are_rejected(raw_record, field, value):
    assert field in errors_of(raw_record("laptop", **{field: value}))


def test_uncommon_but_possible_values_are_not_structurally_invalid(parsed):
    assert parsed("laptop", ram_gb=48, title="HP 48GB RAM Laptop, 512GB SSD").spec.ram_gb == 48


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", 0),
        ("price", -5),
        ("price", "abc"),
        ("price", True),
        ("price", None),
        ("rating", 5.1),
        ("rating", -0.1),
        ("rating", 4.25),
        ("review_count", -1),
        ("review_count", 1.5),
        ("ram_gb", 0),
        ("ram_gb", 4097),
        ("storage_gb", 65537),
    ],
)
def test_invalid_numeric_values(raw_record, field, value):
    assert field in errors_of(raw_record("laptop", **{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "brand",
        "price",
        "currency",
        "category",
        "availability",
        "source_type",
        "is_synthetic",
        "product_id",
    ],
)
def test_missing_required_values(raw_record, field):
    assert field in errors_of(raw_record("laptop", drop=[field]))


def test_blank_required_text_is_missing(raw_record):
    assert "title" in errors_of(raw_record("laptop", title="   "))


@pytest.mark.parametrize(
    ("category", "field", "value"),
    [
        ("laptop", "category", "shoe"),
        ("laptop", "availability", "gone"),
        ("laptop", "subcategory", "gaming"),
        ("laptop", "storage_type", "NVME"),
        ("laptop", "storage_type", "eMMC"),
        ("laptop", "storage_interface", "IDE"),
        ("shoes", "size_system", "JP"),
        ("shoes", "gender", "robot"),
        ("headphones", "connectivity", "telepathy"),
        ("laptop", "source_type", "scraped"),
    ],
)
def test_invalid_taxonomy_values_are_rejected(raw_record, category, field, value):
    assert field in errors_of(raw_record(category, **{field: value}))


@pytest.mark.parametrize(
    ("category", "field", "value"),
    [
        ("shoes", "ram_gb", 8),
        ("shoes", "storage_gb", 256),
        ("laptop", "size", 9),
        ("laptop", "size_system", "UK"),
        ("headphones", "screen_size_inches", 6.5),
        ("phone", "storage_type", "SSD"),
        ("laptop", "wireless", True),
        ("phone", "definitely_unknown_attribute", 1),
    ],
)
def test_cross_category_attributes_are_rejected(raw_record, category, field, value):
    assert field in errors_of(raw_record(category, **{field: value}))


def test_storage_combinations(raw_record, parsed):
    assert parsed("laptop", storage_type="SSD", storage_interface="NVME")
    assert parsed("laptop", storage_type="SSD", storage_interface="SATA")
    assert parsed("laptop", storage_type="HDD", storage_interface="SATA")
    assert parsed("laptop", storage_type="ssd ", storage_interface="nvme").spec.storage_interface
    assert "<record>" in errors_of(
        raw_record("laptop", storage_type="HDD", storage_interface="NVME")
    )
    assert errors_of(raw_record("laptop", storage_type=None, storage_interface="NVME"))


def test_synthetic_flag_must_agree_with_source_type(raw_record):
    assert errors_of(raw_record("laptop", is_synthetic=False))
    assert errors_of(raw_record("laptop", source_type="public_dataset"))


def test_paired_and_contradictory_attributes(raw_record):
    assert errors_of(raw_record("shoes", size_system=None))
    assert errors_of(raw_record("shoes", size=None))
    assert errors_of(raw_record("headphones", connectivity="wired", wireless=True))
    assert errors_of(raw_record("headphones", connectivity="bluetooth", wireless=False))


def test_all_problems_are_reported_together(raw_record):
    errors = errors_of(raw_record("laptop", price=-1, ram_gb="1 MB", brand=None, colour="red"))
    assert {"price", "ram_gb", "brand"} <= set(errors)


def test_non_object_records_are_rejected():
    assert "<record>" in errors_of(["not", "an", "object"])


def test_content_hash_is_canonical_and_sensitive(parsed):
    a = parsed("laptop", price=55000, ram_gb=16)
    b = parsed("laptop", price="Rs 55,000.00", ram_gb="16 GB")
    assert record_content_hash(a) == record_content_hash(b)
    assert record_content_hash(a) != record_content_hash(parsed("laptop", ram_gb=8))
