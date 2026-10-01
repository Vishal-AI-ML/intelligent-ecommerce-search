"""Database audit of the search index (data-quality check 10, database audits only).

Detects: a missing document, a stale document version, a source-hash mismatch, a vector that
differs from one rebuilt from the stored catalog (tampered or unexpected lexemes), and
unexpected lexemes that are review, evaluation or provenance text. It also returns the exact
search-index configuration and counts it audited, for the quality report's `search_index`
section.
"""

from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ecommerce_search.catalog.quality.findings import CheckOutcome, Finding, Severity
from ecommerce_search.models.catalog import CatalogDataset, CatalogReview
from ecommerce_search.search.documents import DOCUMENT_VERSION, FTS_CONFIG, build_document
from ecommerce_search.search.indexing import (
    EXPECTED_VECTOR_SQL,
    document_params,
    load_catalog_records,
)
from ecommerce_search.search.query import SEARCH_VERSION

CHECK_NAME = "embedding_search_text_leakage"
NO_INDEX_NOTE = (
    "Not applicable: the search index (product_search_documents) does not exist in this "
    "database (not migrated to revision 0003)."
)
_STORED_SQL = text(
    "SELECT product_id, document_version, source_content_sha256, search_vector::text, "
    "tsvector_to_array(search_vector) FROM product_search_documents"
)
_LEXEMES_SQL = text(
    "SELECT tsvector_to_array(to_tsvector(CAST(:cfg AS regconfig), CAST(:t AS text)))"
)
_EXPECTED_LEXEMES_SQL = text("SELECT tsvector_to_array(CAST(:v AS tsvector))")


@dataclass(frozen=True)
class SearchAudit:
    outcome: CheckOutcome
    # The `search_index` section of the quality report: configuration and counts audited.
    metadata: dict


def _forbidden_lexemes(session: Session) -> set[str]:
    """Lexemes of provenance and human-review text that must never be searchable."""
    parts: list[str] = ["synthetic", "public_dataset"]
    parts += list(session.scalars(select(CatalogDataset.dataset_id)))
    for review in session.execute(
        select(CatalogReview.verdict, CatalogReview.issue_fields, CatalogReview.notes)
    ):
        parts += [value for value in review if value]
    lexemes = session.execute(_LEXEMES_SQL, {"cfg": FTS_CONFIG, "t": " ".join(parts)}).scalar_one()
    return set(lexemes)


def audit_search_index(session: Session) -> SearchAudit:
    if session.scalar(text("SELECT to_regclass('public.product_search_documents')")) is None:
        return SearchAudit(
            CheckOutcome(0, (), NO_INDEX_NOTE),
            {"applicable": False, "reason": NO_INDEX_NOTE},
        )
    records, failed = load_catalog_records(session)  # invalid rows are reported elsewhere
    stored = {
        pid: (version, sha, vector, set(lexemes))
        for pid, version, sha, vector, lexemes in session.execute(_STORED_SQL)
    }
    forbidden = _forbidden_lexemes(session)
    findings: list[Finding] = []
    counts = {"missing": 0, "stale_version": 0, "hash_mismatch": 0, "tampered_vector": 0}

    def error(pid: str, message: str) -> None:
        findings.append(Finding(CHECK_NAME, Severity.ERROR, message, product_id=pid))

    for record, source_sha in records:
        pid = record.product_id
        row = stored.get(pid)
        if row is None:
            counts["missing"] += 1
            error(pid, "search document is missing")
            continue
        version, sha, vector, lexemes = row
        if version != DOCUMENT_VERSION:
            counts["stale_version"] += 1
            error(
                pid, f"search document version {version!r} is stale (expected {DOCUMENT_VERSION!r})"
            )
        if sha != source_sha:
            counts["hash_mismatch"] += 1
            error(pid, "search document was built from different product content (hash mismatch)")
        expected = session.execute(
            EXPECTED_VECTOR_SQL, document_params(build_document(record))
        ).scalar_one()
        if vector != str(expected):
            counts["tampered_vector"] += 1
            expected_lexemes = set(
                session.execute(_EXPECTED_LEXEMES_SQL, {"v": str(expected)}).scalar_one()
            )
            leaked = sorted((lexemes - expected_lexemes) & forbidden)
            if leaked:
                error(pid, f"search vector contains forbidden review/provenance text: {leaked[:5]}")
            else:
                error(pid, "search vector differs from the vector rebuilt from the catalog")
    metadata = {
        "applicable": True,
        "document_version": DOCUMENT_VERSION,
        "fts_config": FTS_CONFIG,
        "search_version": SEARCH_VERSION,
        "products": len(records) + len(failed),
        "products_audited": len(records),
        "search_documents": len(stored),
        **counts,
    }
    return SearchAudit(CheckOutcome(len(records), tuple(findings)), metadata)
