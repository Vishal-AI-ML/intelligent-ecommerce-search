from ecommerce_search.ingestion.service import ingest_lock_key
from ecommerce_search.search import documents as docs
from ecommerce_search.search.indexing import search_lock_key


def test_section_to_weight_assignment_is_the_documented_baseline():
    assert docs.SECTION_WEIGHTS == {
        "name": "A",  # title, brand
        "taxonomy": "B",  # category/subcategory forms, model identifier variants
        "attributes": "C",  # canonical technical attributes
        "description": "D",
    }


def test_document_version_and_fts_config_are_explicit():
    # Changing the builder, the section assignment or the FTS config requires a version bump.
    assert docs.DOCUMENT_VERSION == "1"
    assert docs.FTS_CONFIG == "simple"


def test_search_lock_key_is_deterministic_signed_64_bit_and_distinct_from_ingest_locks():
    assert search_lock_key() == search_lock_key()
    assert -(2**63) <= search_lock_key() < 2**63
    assert search_lock_key() != ingest_lock_key("synthetic-seed")
