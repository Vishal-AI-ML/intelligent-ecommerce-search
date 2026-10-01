"""Binds the semantics of the embedding text to EMBEDDING_TEXT_VERSION.

The digest covers the builder's output for every committed-seed product. Embeddings generated
from different text are different vectors, so a change must be explicit.

IF THIS TEST FAILS because you intentionally changed the embedding text:
  1. bump EMBEDDING_TEXT_VERSION in `ecommerce_search/embeddings/text.py`;
  2. review the new corpus (`corpus_digest()` below) and add its value for the NEW version to
     EMBEDDING_CORPUS_SHA256_BY_VERSION (keep the old entries: they document history);
  3. regenerate every embedding and rerun any evaluation.
This test never updates itself, and the expected value is a constant.
"""

import hashlib
import json
from pathlib import Path

import pytest

from ecommerce_search.embeddings import text as et
from ecommerce_search.ingestion.loader import load_catalog

SEED = Path(__file__).resolve().parents[3] / "data" / "seed" / "catalog_seed_v1.jsonl"

EMBEDDING_CORPUS_SHA256_BY_VERSION = {
    "1": "7faa9d147f03da730bab18d6ca93894700b14508ed5e968e569cfd1c014618e4",
}


def corpus_digest() -> str:
    catalog = load_catalog(SEED)
    semantics = {
        "connectivity_phrases": {k.value: v for k, v in et.CONNECTIVITY_PHRASES.items()},
        "texts": [
            {"product_id": line.record.product_id, "text": et.build_embedding_text(line.record)}
            for line in sorted(catalog.lines, key=lambda line: line.record.product_id)
        ],
    }
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_every_text_version_in_use_has_a_pinned_corpus_digest():
    assert et.EMBEDDING_TEXT_VERSION in EMBEDDING_CORPUS_SHA256_BY_VERSION


def test_unchanged_builder_semantics_reproduce_the_pinned_digest():
    pinned = EMBEDDING_CORPUS_SHA256_BY_VERSION[et.EMBEDDING_TEXT_VERSION]
    assert corpus_digest() == pinned, (
        "the embedding text changed: bump EMBEDDING_TEXT_VERSION, review and pin the new digest, "
        "regenerate embeddings and rerun evaluation (see this module's docstring)"
    )


def test_digest_is_deterministic():
    assert corpus_digest() == corpus_digest()


@pytest.mark.parametrize("mutation", ["connectivity", "capacity", "number"])
def test_every_semantic_input_changes_the_digest(monkeypatch, mutation):
    baseline = corpus_digest()
    if mutation == "connectivity":
        monkeypatch.setattr(
            et,
            "CONNECTIVITY_PHRASES",
            {**et.CONNECTIVITY_PHRASES, et.Connectivity.BLUETOOTH: "BT"},
        )
    elif mutation == "capacity":
        monkeypatch.setattr(et, "capacity_text", lambda gb: f"{gb}GB")
    else:
        monkeypatch.setattr(et, "number_text", lambda value: str(value))
    assert corpus_digest() != baseline
