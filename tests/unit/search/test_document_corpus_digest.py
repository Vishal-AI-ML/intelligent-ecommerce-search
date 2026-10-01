"""Binds the semantics of the search documents to DOCUMENT_VERSION.

The digest below covers everything whose change requires a reindex: the four section texts of
every committed-seed document (so the builder's output), the section-to-weight assignment,
FTS_CONFIG and the explicit category/subcategory plural mapping.

IF THIS TEST FAILS because you intentionally changed builder semantics:
  1. bump DOCUMENT_VERSION in `ecommerce_search/search/documents.py`;
  2. review the new corpus (`corpus_digest()` below) and add its value for the NEW version to
     DOCUMENT_CORPUS_SHA256_BY_VERSION (keep the old entries: they document history);
  3. reindex every database (`python -m ecommerce_search.search reindex --database NAME`) and
     rerun any evaluation, because search results change.
This test never updates itself, and the expected value is a constant, not computed from the
implementation at assertion time. Formatting-only refactors do not change the digest because it
is computed from the builder's output, not its source text.
"""

import hashlib
import json
from pathlib import Path

import pytest

from ecommerce_search.ingestion.loader import load_catalog
from ecommerce_search.search import documents as docs

SEED = Path(__file__).resolve().parents[3] / "data" / "seed" / "catalog_seed_v1.jsonl"

DOCUMENT_CORPUS_SHA256_BY_VERSION = {
    "1": "86bf819eb93d57298432d9638e32da3f63ddee6df18bae6b88bad62fceda1b2d",
}


def corpus_digest() -> str:
    """Canonical SHA-256 over the semantics of every seed document."""
    catalog = load_catalog(SEED)
    semantics = {
        "fts_config": docs.FTS_CONFIG,
        "section_weights": docs.SECTION_WEIGHTS,
        "category_forms": {c.value: list(forms) for c, forms in docs.CATEGORY_FORMS.items()},
        "subcategory_forms": {k: list(v) for k, v in docs.SUBCATEGORY_FORMS.items()},
        "documents": [
            {
                "product_id": line.record.product_id,
                "sections": docs.build_document(line.record).sections(),
            }
            for line in sorted(catalog.lines, key=lambda line: line.record.product_id)
        ],
    }
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_the_seed_has_240_documents():
    assert len(load_catalog(SEED).lines) == 240


def test_every_document_version_in_use_has_a_pinned_corpus_digest():
    version = docs.DOCUMENT_VERSION
    assert version in DOCUMENT_CORPUS_SHA256_BY_VERSION, (
        f"DOCUMENT_VERSION {version!r} has no pinned corpus digest: review the new corpus and add "
        "its digest (see this module's docstring)"
    )


def test_unchanged_builder_semantics_reproduce_the_pinned_digest():
    pinned = DOCUMENT_CORPUS_SHA256_BY_VERSION[docs.DOCUMENT_VERSION]
    assert corpus_digest() == pinned, (
        "the search documents changed: bump DOCUMENT_VERSION, review and pin the new digest, "
        "reindex and rerun evaluation (see this module's docstring)"
    )


def test_digest_is_deterministic():
    assert corpus_digest() == corpus_digest()


@pytest.mark.parametrize(
    "mutation",
    ["fts_config", "section_weights", "category_forms", "subcategory_forms", "builder_text"],
)
def test_every_semantic_input_changes_the_digest(monkeypatch, mutation):
    baseline = corpus_digest()
    if mutation == "fts_config":
        monkeypatch.setattr(docs, "FTS_CONFIG", "english")
    elif mutation == "section_weights":
        monkeypatch.setattr(docs, "SECTION_WEIGHTS", {**docs.SECTION_WEIGHTS, "name": "B"})
    elif mutation == "category_forms":
        monkeypatch.setattr(
            docs, "CATEGORY_FORMS", {**docs.CATEGORY_FORMS, docs.Category.SHOES: ("shoe",)}
        )
    elif mutation == "subcategory_forms":
        monkeypatch.setattr(docs, "SUBCATEGORY_FORMS", {**docs.SUBCATEGORY_FORMS, "running": ()})
    else:
        monkeypatch.setattr(docs, "capacity_terms", lambda gb: [f"{gb}gb", "extra"])
    assert corpus_digest() != baseline
