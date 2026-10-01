import pytest

from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.embeddings.text import (
    INPUT_FIELDS,
    build_embedding_text,
    capacity_text,
    number_text,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def test_laptop_text(parsed):
    assert build_embedding_text(parsed("laptop")) == (
        "HP Testline 15.6-inch Laptop - Intel Core i5, 16GB RAM, 512GB SSD Model T1\n"
        "Brand: HP. Category: laptop (business).\n"
        "Specifications: 16 GB RAM; 512 GB SSD storage (NVMe); processor Intel Core i5; "
        "Windows 11; 15.6-inch screen; weight 1.7 kg.\n"
        "Test laptop."
    )


def test_phone_shoe_and_headphone_texts(parsed):
    assert build_embedding_text(parsed("phone")).splitlines()[2] == (
        "Specifications: 8 GB RAM; 128 GB storage; camera 50 MP dual camera; "
        "5000 mAh battery; 6.6-inch screen; Android."
    )
    assert build_embedding_text(parsed("shoes")).splitlines()[2] == (
        "Specifications: color Black; material Mesh; for men."
    )
    assert build_embedding_text(parsed("headphones")).splitlines()[2] == (
        "Specifications: wireless; Bluetooth connectivity; up to 30 hours battery life."
    )


def test_text_is_deterministic(parsed):
    assert build_embedding_text(parsed("laptop")) == build_embedding_text(parsed("laptop"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(14, "14"), ("14.0", "14"), ("1.40", "1.4"), ("15.6", "15.6"), ("0.50", "0.5")],
)
def test_number_text(value, expected):
    from decimal import Decimal

    assert number_text(value if isinstance(value, int) else Decimal(value)) == expected


@pytest.mark.parametrize(
    ("gb", "expected"), [(256, "256 GB"), (1024, "1 TB"), (2048, "2 TB"), (1536, "1536 GB")]
)
def test_capacity_text(gb, expected):
    assert capacity_text(gb) == expected


def test_seller_style_values_give_the_same_text(parsed):
    spaced = parsed("laptop", ram_gb="16 GB", storage_gb="512GB", brand="hp")
    assert build_embedding_text(spaced) == build_embedding_text(parsed("laptop"))


def test_missing_attributes_emit_nothing(parsed):
    laptop = parsed(
        "laptop",
        ram_gb=None,
        storage_gb=None,
        storage_type=None,
        storage_interface=None,
        processor=None,
        gpu=None,
        operating_system=None,
        screen_size_inches=None,
        weight_kg=None,
        description=None,
        subcategory=None,
    )
    assert build_embedding_text(laptop) == (
        "HP Testline 15.6-inch Laptop - Intel Core i5, 16GB RAM, 512GB SSD Model T1\n"
        "Brand: HP. Category: laptop."
    )


def test_negative_booleans_never_emit_positive_phrases(parsed):
    wired = build_embedding_text(
        parsed("headphones", wireless=False, anc=False, connectivity="wired")
    )
    assert "wired connection" in wired
    spec_line = wired.splitlines()[2].lower()
    assert "wireless" not in spec_line and "noise" not in spec_line and "anc" not in spec_line
    anc = build_embedding_text(parsed("headphones", anc=True))
    assert "active noise cancellation (ANC)" in anc


def test_whitespace_and_unicode_are_normalized(parsed):
    decomposed = "Cafe" + chr(0x0301) + "   Edition"  # e + combining acute accent (NFD)
    text = build_embedding_text(parsed("laptop", description=f"  {decomposed}" + chr(9) + "line "))
    assert text.splitlines()[-1] == "Caf" + chr(0x00E9) + " Edition line"  # NFC, collapsed
    assert all(line == line.strip() for line in text.splitlines())


class _Spy:
    def __init__(self, target, seen):
        self._target, self._seen = target, seen

    def __getattr__(self, name):
        self._seen.add(name)
        value = getattr(self._target, name)
        return _Spy(value, self._seen) if name == "spec" else value


@pytest.mark.parametrize("kind", ["laptop", "phone", "shoes", "headphones"])
def test_builder_reads_only_whitelisted_fields(parsed, kind):
    seen: set[str] = set()
    build_embedding_text(_Spy(parsed(kind), seen))
    assert seen - {"spec"} <= INPUT_FIELDS


def test_whitelist_excludes_labels_provenance_and_ranking_fields():
    assert not INPUT_FIELDS & qp.FORBIDDEN_LABEL_FIELDS
    excluded = {
        "seller_id",
        "price",
        "currency",
        "availability",
        "rating",
        "review_count",
        "is_synthetic",
        "source_type",
        "product_id",
        "content_sha256",
        "dataset_pk",
        "raw_line",
        "size",
        "size_system",
    }
    assert not INPUT_FIELDS & excluded


@pytest.mark.parametrize("kind", ["laptop", "shoes"])
def test_text_never_contains_excluded_values(parsed, kind):
    record = parsed(
        kind,
        seller_id="SELLER-MARKER-7",
        price=77777.5,
        rating=3.9,
        review_count=424242,
        availability="out_of_stock",
        product_id="PID-MARKER-9",
    )
    blob = build_embedding_text(record).lower()
    for marker in (
        "seller-marker",
        "77777",
        "424242",
        "out_of_stock",
        "out of stock",
        "inr",
        "3.9",
        "pid-marker",
        "public_dataset",
        "source_type",
    ):
        assert marker not in blob


def test_shoe_size_is_excluded(parsed):
    text = build_embedding_text(parsed("shoes", size=11.5, size_system="US"))
    assert "11.5" not in text and "US" not in text.split()
