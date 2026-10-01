"""Create, refresh and inspect search documents. Everything runs in the caller's transaction.

Concurrency: every writer takes the transaction-scoped advisory lock from `search_lock_key()`.
Ingestion already holds its per-dataset lock when it calls `sync_documents`, so the order is
always dataset lock first, search lock second, and reindex takes only the search lock. There is
no cycle.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ecommerce_search.catalog.mapping import flat_from_rows
from ecommerce_search.catalog.schemas import CatalogRecord, RecordValidationError, parse_raw
from ecommerce_search.catalog.taxonomy import Category
from ecommerce_search.models.catalog import SPEC_TABLES, Product
from ecommerce_search.models.search import ProductSearchDocument as Doc
from ecommerce_search.search.documents import (
    DOCUMENT_VERSION,
    FTS_CONFIG,
    SearchDocument,
    build_document,
)

_CHUNK = 1000

# The weighted-vector expression appears in exactly these two statements; keep them identical.
UPSERT_SQL = text(
    """
    INSERT INTO product_search_documents
        (product_id, document_version, source_content_sha256, search_vector)
    VALUES (
        :product_id, :document_version, :source_sha,
        setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:name AS text)), 'A')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:taxonomy AS text)), 'B')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:attributes AS text)), 'C')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:description AS text)), 'D')
    )
    ON CONFLICT (product_id) DO UPDATE SET
        document_version = EXCLUDED.document_version,
        source_content_sha256 = EXCLUDED.source_content_sha256,
        search_vector = EXCLUDED.search_vector,
        built_at = now()
    """
)
EXPECTED_VECTOR_SQL = text(
    """
    SELECT CAST(
        setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:name AS text)), 'A')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:taxonomy AS text)), 'B')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:attributes AS text)), 'C')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), CAST(:description AS text)), 'D')
    AS text)
    """
)


def search_lock_key() -> int:
    """Deterministic signed 64-bit advisory-lock key for search-index writers."""
    digest = hashlib.sha256(b"search-index:product_search_documents").digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def document_params(document: SearchDocument) -> dict[str, str]:
    return {
        "cfg": FTS_CONFIG,
        "name": document.name,
        "taxonomy": document.taxonomy,
        "attributes": document.attributes,
        "description": document.description,
    }


@dataclass
class SyncResult:
    inserted: int = 0  # no document existed
    updated: int = 0  # a stale document was rebuilt (= stale_version + hash_mismatch)
    unchanged: int = 0  # already current: not rewritten, built_at preserved
    stale_version: int = 0
    hash_mismatch: int = 0

    def summary(self) -> str:
        return (
            f"inserted={self.inserted} updated={self.updated} unchanged={self.unchanged} "
            f"stale_found={self.updated} (document_version={self.stale_version}, "
            f"content_hash={self.hash_mismatch})"
        )


def take_search_lock(session: Session) -> None:
    session.execute(select(func.pg_advisory_xact_lock(search_lock_key())))


def sync_documents(session: Session, items: Sequence[tuple[CatalogRecord, str]]) -> SyncResult:
    """Create missing and rebuild stale documents for `(record, products.content_sha256)` pairs.

    Current documents are not rewritten. Call inside the transaction that wrote the products."""
    take_search_lock(session)
    result = SyncResult()
    existing: dict[str, tuple[str, str]] = {}
    ids = [record.product_id for record, _ in items]
    for start in range(0, len(ids), _CHUNK):
        rows = session.execute(
            select(Doc.product_id, Doc.document_version, Doc.source_content_sha256).where(
                Doc.product_id.in_(ids[start : start + _CHUNK])
            )
        )
        existing.update({pid: (version, sha) for pid, version, sha in rows})

    pending: list[dict[str, str]] = []
    for record, source_sha in sorted(items, key=lambda item: item[0].product_id):
        current = existing.get(record.product_id)
        if current is None:
            result.inserted += 1
        elif current[0] != DOCUMENT_VERSION:
            result.updated += 1
            result.stale_version += 1
        elif current[1] != source_sha:
            result.updated += 1
            result.hash_mismatch += 1
        else:
            result.unchanged += 1
            continue
        pending.append(
            {
                "product_id": record.product_id,
                "document_version": DOCUMENT_VERSION,
                "source_sha": source_sha,
                **document_params(build_document(record)),
            }
        )
    if pending:
        session.execute(UPSERT_SQL, pending)
    return result


class IndexingError(Exception):
    """Stored catalog rows cannot be turned into documents."""


def load_catalog_records(
    session: Session,
) -> tuple[list[tuple[CatalogRecord, str]], list[str]]:
    """Rebuild validated records from stored rows. Returns (records with stored content hash,
    ids of products whose stored rows no longer validate)."""
    specs = {}
    for category, table in SPEC_TABLES.items():
        for spec in session.scalars(select(table)):
            specs[(category.value, spec.product_id)] = spec
    records: list[tuple[CatalogRecord, str]] = []
    failed: list[str] = []
    for product in session.scalars(select(Product).order_by(Product.product_id)):
        spec = specs.get((Category(product.category).value, product.product_id))
        if spec is None:
            failed.append(product.product_id)
            continue
        try:
            records.append((parse_raw(flat_from_rows(product, spec)), product.content_sha256))
        except RecordValidationError:
            failed.append(product.product_id)
    return records, failed


def reindex(session: Session) -> SyncResult:
    """Heal missing and stale documents from the stored catalog (all or nothing)."""
    if session.scalar(text("SELECT to_regclass('public.product_search_documents')")) is None:
        # Without this check an empty catalog would "succeed" while no table exists at all.
        raise IndexingError(
            "the search index table does not exist: the database is not migrated to the "
            "current revision; run `uv run alembic upgrade head` and retry. Nothing was written"
        )
    take_search_lock(session)
    records, failed = load_catalog_records(session)
    if failed:
        shown = ", ".join(failed[:5])
        raise IndexingError(
            f"{len(failed)} product(s) cannot be indexed because their stored rows do not "
            f"validate (e.g. {shown}); nothing was written. Run the catalog check first"
        )
    return sync_documents(session, records)


@dataclass(frozen=True)
class IndexStatus:
    total_products: int
    total_documents: int
    missing: int
    stale_version: int
    hash_mismatch: int

    @property
    def current(self) -> bool:
        return not (self.missing or self.stale_version or self.hash_mismatch)

    def describe(self) -> str:
        return (
            f"products={self.total_products} documents={self.total_documents} "
            f"missing={self.missing} stale_document_version={self.stale_version} "
            f"content_hash_mismatch={self.hash_mismatch} "
            f"expected_document_version={DOCUMENT_VERSION} "
            f"state={'CURRENT' if self.current else 'INCOMPLETE'}"
        )


def index_status(session: Session) -> IndexStatus:
    """Read-only completeness summary (no search request ever claims completeness)."""
    scalar = session.scalar
    return IndexStatus(
        total_products=scalar(select(func.count()).select_from(Product)) or 0,
        total_documents=scalar(select(func.count()).select_from(Doc)) or 0,
        missing=scalar(
            select(func.count())
            .select_from(Product)
            .outerjoin(Doc, Doc.product_id == Product.product_id)
            .where(Doc.product_id.is_(None))
        )
        or 0,
        stale_version=scalar(
            select(func.count()).select_from(Doc).where(Doc.document_version != DOCUMENT_VERSION)
        )
        or 0,
        hash_mismatch=scalar(
            select(func.count())
            .select_from(Doc)
            .join(Product, Product.product_id == Doc.product_id)
            .where(Doc.source_content_sha256 != Product.content_sha256)
        )
        or 0,
    )
