import pytest

from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.taxonomy import SUBCATEGORIES, Category
from ecommerce_search.search import documents as docs
from ecommerce_search.search.documents import (
    CATEGORY_FORMS,
    INPUT_FIELDS,
    SUBCATEGORY_FORMS,
    build_document,
    capacity_terms,
    identifier_variants,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def terms(text: str) -> list[str]:
    return text.split()


def test_laptop_sections(parsed):
    doc = build_document(parsed("laptop"))
    assert doc.name == (
        "HP Testline 15.6-inch Laptop - Intel Core i5, 16GB RAM, 512GB SSD Model T1 HP"
    )
    assert terms(doc.taxonomy) == ["laptop", "laptops", "business"]
    assert doc.attributes == "16gb ram 512gb ssd nvme Intel Core i5 Windows 11"
    assert doc.description == "Test laptop."


def test_sections_are_deterministic(parsed):
    assert build_document(parsed("laptop")) == build_document(parsed("laptop"))
    assert list(build_document(parsed("phone")).sections()) == list(docs.SECTION_WEIGHTS)


@pytest.mark.parametrize(
    ("gb", "expected"),
    [(256, ["256gb"]), (512, ["512gb"]), (1024, ["1024gb", "1tb"]), (2048, ["2048gb", "2tb"])],
)
def test_capacity_terms_are_canonical(gb, expected):
    assert capacity_terms(gb) == expected


def test_seller_style_values_normalize_to_the_same_document(parsed):
    spaced = parsed("laptop", ram_gb="16 GB", storage_gb="512GB")
    assert build_document(spaced).attributes == build_document(parsed("laptop")).attributes
    terabyte = build_document(parsed("laptop", storage_gb="1TB", storage_type="ssd"))
    assert "1024gb 1tb" in terabyte.attributes


def test_phone_shoe_and_headphone_attributes(parsed):
    assert build_document(parsed("phone")).attributes == (
        "8gb ram 128gb 50 MP dual camera 5000mah Android"
    )
    assert build_document(parsed("shoes")).attributes == "Black Mesh men"
    assert build_document(parsed("headphones")).attributes == "wireless bluetooth"


def test_no_invented_defaults_for_missing_attributes(parsed):
    laptop = parsed(
        "laptop",
        ram_gb=None,
        storage_gb=None,
        storage_type=None,
        storage_interface=None,
        processor=None,
        gpu=None,
        operating_system=None,
        description=None,
    )
    doc = build_document(laptop)
    assert doc.attributes == "" and doc.description == ""
    shoes = build_document(parsed("shoes", color=None, material=None, gender=None))
    assert shoes.attributes == ""
    assert build_document(parsed("laptop", subcategory=None)).taxonomy == "laptop laptops"


def test_negative_booleans_never_emit_positive_tokens(parsed):
    wired = build_document(parsed("headphones", wireless=False, anc=False, connectivity="wired"))
    assert terms(wired.attributes) == ["wired"]
    assert "anc" not in wired.attributes and "wireless" not in wired.attributes
    unknown = build_document(parsed("headphones", wireless=None, anc=None, connectivity=None))
    assert unknown.attributes == ""
    assert terms(build_document(parsed("headphones", anc=True)).attributes) == [
        "wireless",
        "bluetooth",
        "anc",
    ]


def test_connectivity_2_4ghz_is_written_out_explicitly(parsed):
    doc = build_document(parsed("headphones", connectivity="wireless_2_4ghz"))
    assert doc.attributes == "wireless 2.4ghz wireless"


def test_model_identifier_variants():
    assert identifier_variants("Sony WH-1000XM5 Headphones") == ["WH1000XM5"]
    assert identifier_variants("Model (XC-90X), WH-1000XM5 WH-1000XM5") == ["XC90X", "WH1000XM5"]
    # not model-like: no digit, no letter, plain token, decimal with hyphen
    for title in ("In-ear Wireless", "123-456", "LP101 i5 8GB", "14.0-inch Laptop"):
        assert identifier_variants(title) == []


def test_identifier_variants_reach_the_taxonomy_section(parsed):
    doc = build_document(parsed("headphones", title="Sony WH-1000XM5 Headphones"))
    assert terms(doc.taxonomy)[-1] == "WH1000XM5"


def test_explicit_plural_forms_cover_exactly_the_taxonomy():
    assert set(CATEGORY_FORMS) == set(Category)
    assert {s for subs in SUBCATEGORIES.values() for s in subs} == set(SUBCATEGORY_FORMS)
    assert CATEGORY_FORMS[Category.LAPTOP] == ("laptop", "laptops")
    assert CATEGORY_FORMS[Category.PHONE] == ("phone", "phones")
    assert CATEGORY_FORMS[Category.SHOES] == ("shoe", "shoes")
    assert CATEGORY_FORMS[Category.HEADPHONES] == ("headphone", "headphones")
    assert SUBCATEGORY_FORMS["smartphone"] == ("smartphone", "smartphones")
    # explicit mapping, not a generic "add s" rule: values that gain nothing stay singular
    assert SUBCATEGORY_FORMS["running"] == ("running",)
    assert SUBCATEGORY_FORMS["sports"] == ("sports",)


class _Spy:
    """Wraps a record and logs every attribute the builder reads."""

    def __init__(self, target, seen):
        self._target, self._seen = target, seen

    def __getattr__(self, name):
        self._seen.add(name)
        value = getattr(self._target, name)
        return _Spy(value, self._seen) if name == "spec" else value


@pytest.mark.parametrize("kind", ["laptop", "phone", "shoes", "headphones"])
def test_builder_reads_only_whitelisted_fields(parsed, kind):
    seen: set[str] = set()
    build_document(_Spy(parsed(kind), seen))
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
        "screen_size_inches",
        "weight_kg",
        "battery_life_hours",
    }
    assert not INPUT_FIELDS & excluded


def test_document_text_never_contains_excluded_values(parsed):
    record = parsed(
        "laptop",
        seller_id="SELLER-MARKER-7",
        price=77777.5,
        rating=3.9,
        review_count=424242,
        availability="out_of_stock",
    )
    doc = build_document(record)
    blob = " ".join(doc.sections().values()).lower()
    for marker in (
        "seller-marker",
        "77777",
        "424242",
        "out_of_stock",
        "synthetic",
        "public_dataset",
    ):
        assert marker not in blob
