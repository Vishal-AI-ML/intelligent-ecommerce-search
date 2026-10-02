"""Pure parts of dense indexing: staleness classification, vector checks, literals, lock keys."""

import math
from dataclasses import replace

import pytest

from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.ingestion.service import ingest_lock_key
from ecommerce_search.search import dense_indexing as di
from ecommerce_search.search.indexing import search_lock_key

pytestmark = pytest.mark.usefixtures("no_socket_connect")

SPEC = ALL_MINILM_L6_V2
SRC = "a" * 64
TXT = "b" * 64


def current_row(**changes) -> dict:
    row = {
        "model_id": SPEC.model_id,
        "model_revision": SPEC.revision,
        "dimension": 384,
        "stored_dims": 384,
        "normalized": True,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_config_sha256": SPEC.config_sha256(),
        "source_content_sha256": SRC,
        "embedding_text_sha256": TXT,
        "stored_norm": 1.0,
    }
    row.update(changes)
    return row


def test_a_current_row_has_no_reasons():
    assert di.stale_reasons(current_row(), SPEC, SRC, TXT) == []
    assert di.stale_reasons(None, SPEC, SRC, TXT) == ["missing"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"model_id": "other/model"}, ["stale_model"]),
        ({"model_revision": "f" * 40}, ["stale_model"]),
        ({"embedding_text_version": "0"}, ["stale_text_version"]),
        ({"embedding_config_sha256": "c" * 64}, ["stale_config"]),
        ({"dimension": 768}, ["dimension_mismatch"]),
        ({"stored_dims": 768}, ["dimension_mismatch"]),
        ({"normalized": False}, ["normalization_mismatch"]),
        ({"source_content_sha256": "d" * 64}, ["source_hash_mismatch"]),
        ({"embedding_text_sha256": "e" * 64}, ["text_hash_mismatch"]),
    ],
)
def test_every_staleness_condition_is_detected(changes, expected):
    assert di.stale_reasons(current_row(**changes), SPEC, SRC, TXT) == expected


def test_a_different_spec_makes_rows_stale_by_model_and_config():
    other = replace(SPEC, revision="f" * 40)
    assert di.stale_reasons(current_row(), other, SRC, TXT) == ["stale_model", "stale_config"]


@pytest.mark.parametrize(
    ("norm", "normalized", "invalid"),
    [
        (1.0, True, False),
        (1.0005, True, False),
        (1.01, True, True),
        (0.0, True, True),
        (0.0, False, True),
        (math.nan, False, True),
        (math.inf, False, True),
        (None, False, True),
        (3.0, False, False),
    ],
)
def test_invalid_vector(norm, normalized, invalid):
    assert di.invalid_vector({"stored_norm": norm, "normalized": normalized}) is invalid


def test_vector_literal_is_exact_and_parameter_safe():
    assert di.vector_literal([0.1, -2, 1e-07]) == "[0.1,-2.0,1e-07]"
    assert float(di.vector_literal([1 / 3])[1:-1]) == 1 / 3  # repr round-trips exactly


def test_lock_keys_are_deterministic_and_distinct():
    assert di.embedding_lock_key() == di.embedding_lock_key()
    keys = {di.embedding_lock_key(), search_lock_key(), ingest_lock_key("synthetic-seed")}
    assert len(keys) == 3
    assert -(2**63) <= di.embedding_lock_key() < 2**63


def test_spec_with_another_dimension_is_refused_before_any_work():
    with pytest.raises(di.EmbeddingError, match="new migration"):
        di.check_spec(replace(SPEC, dimension=768))


def test_text_hash_is_sha256_of_utf8():
    import hashlib

    value = "Caf" + chr(0xE9)
    assert di.text_sha256(value) == hashlib.sha256(value.encode("utf-8")).hexdigest()
