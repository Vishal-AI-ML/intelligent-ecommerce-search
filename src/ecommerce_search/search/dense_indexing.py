"""Generate, heal and inspect product embeddings (Milestone 4).

`embed` runs in three separated phases:

1. **plan** (short read-only transaction, no lock): rebuild every product's validated record from
   the stored catalog rows, build its embedding text and classify its stored embedding as
   current, missing or stale (and why);
2. **encode** (no transaction, no lock): check every pending text against the model's token
   limit (any over-long text aborts the run: nothing is ever truncated), encode in batches and
   validate every vector;
3. **write** (one transaction): take the embedding advisory lock, re-read the content hash of
   every pending product and the current embedding rows, skip products whose content changed
   since the plan (`changed_during_run`, a re-run heals them) and rows a concurrent run already
   made current, then upsert the rest in one statement. Any failure in any phase writes nothing.

Current rows are never rewritten, so their `embedded_at` is preserved.

Locks: the global order is dataset lock -> search lock -> embedding lock. Ingestion takes the
first two (and never touches embeddings or loads a model), lexical reindex the second, `embed`
only the third; it takes no row locks on `products`. No process takes a lower lock after a higher
one, so there is no cycle. A product updated by an ingestion that commits after `embed` has
checked it keeps a row recording the OLD content hash: it is detectably stale (and excluded from
dense retrieval) until the next `embed`, never silently current.

`embedding_status` is read-only and never loads the model.
"""

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session

from ecommerce_search.catalog.schemas import CatalogRecord
from ecommerce_search.embeddings.provider import Embedder, EmbedderUnavailable, validate_vectors
from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION, build_embedding_text
from ecommerce_search.models.catalog import Product
from ecommerce_search.models.embeddings import EMBEDDING_DIMENSION, NORM_TOLERANCE_SQL
from ecommerce_search.models.embeddings import ProductEmbedding as Emb
from ecommerce_search.search.indexing import load_catalog_records

_CHUNK = 1000
ENCODE_BATCH = 64  # texts handed to the encoder per call (the provider batches internally too)
REENCODE_MIN_COSINE = 0.9999

# Why a stored embedding is not current, in the order they are reported for one product.
STALE_REASONS = (
    "missing",
    "stale_model",
    "stale_text_version",
    "stale_config",
    "dimension_mismatch",
    "normalization_mismatch",
    "source_hash_mismatch",
    "text_hash_mismatch",
)

UPSERT_SQL = text(
    """
    INSERT INTO product_embeddings
        (product_id, embedding, model_id, model_revision, dimension, normalized,
         embedding_text_version, embedding_config_sha256, source_content_sha256,
         embedding_text_sha256)
    VALUES (:product_id, CAST(:embedding AS vector), :model_id, :model_revision, :dimension,
            :normalized, :text_version, :config_sha, :source_sha, :text_sha)
    ON CONFLICT (product_id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        model_id = EXCLUDED.model_id,
        model_revision = EXCLUDED.model_revision,
        dimension = EXCLUDED.dimension,
        normalized = EXCLUDED.normalized,
        embedding_text_version = EXCLUDED.embedding_text_version,
        embedding_config_sha256 = EXCLUDED.embedding_config_sha256,
        source_content_sha256 = EXCLUDED.source_content_sha256,
        embedding_text_sha256 = EXCLUDED.embedding_text_sha256,
        embedded_at = now()
    """
)
_METADATA_SQL = text(
    "SELECT product_id, model_id, model_revision, dimension, normalized, "
    "embedding_text_version, embedding_config_sha256, source_content_sha256, "
    "embedding_text_sha256, vector_dims(embedding) AS stored_dims, "
    "vector_norm(embedding) AS stored_norm FROM product_embeddings"
)


class EmbeddingError(Exception):
    """An operational problem with an already-sanitized message. Nothing was written."""


def embedding_lock_key() -> int:
    """Deterministic signed 64-bit advisory-lock key for embedding writers."""
    digest = hashlib.sha256(b"embedding-index:product_embeddings").digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def take_embedding_lock(session: Session) -> None:
    session.execute(select(func.pg_advisory_xact_lock(embedding_lock_key())))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text input; `repr` keeps every float exactly (no rounding)."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def require_table(session: Session) -> None:
    if session.scalar(text("SELECT to_regclass('public.product_embeddings')")) is None:
        raise EmbeddingError(
            "the embedding table does not exist: the database is not migrated to the current "
            "revision; run `uv run alembic upgrade head` and retry. Nothing was written"
        )


def check_spec(spec: EmbeddingModelSpec) -> None:
    if spec.dimension != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"the model produces {spec.dimension}-dimensional vectors but the table stores "
            f"{EMBEDDING_DIMENSION}; a new migration and ADR are required"
        )


def stale_reasons(row: dict | None, spec: EmbeddingModelSpec, source_sha: str, text_sha: str):
    """Every reason a stored row is not current for the expected configuration (empty = current)."""
    if row is None:
        return ["missing"]
    reasons = []
    if row["model_id"] != spec.model_id or row["model_revision"] != spec.revision:
        reasons.append("stale_model")
    if row["embedding_text_version"] != EMBEDDING_TEXT_VERSION:
        reasons.append("stale_text_version")
    if row["embedding_config_sha256"] != spec.config_sha256():
        reasons.append("stale_config")
    if row["dimension"] != spec.dimension or row["stored_dims"] != spec.dimension:
        reasons.append("dimension_mismatch")
    if row["normalized"] != spec.normalize:
        reasons.append("normalization_mismatch")
    if row["source_content_sha256"] != source_sha:
        reasons.append("source_hash_mismatch")
    if row["embedding_text_sha256"] != text_sha:
        reasons.append("text_hash_mismatch")
    return reasons


def invalid_vector(row: dict) -> bool:
    """Zero, non-finite or (when flagged normalized) non-unit stored vector."""
    norm = row["stored_norm"]
    if norm is None or not math.isfinite(norm) or norm == 0:
        return True
    return bool(row["normalized"]) and abs(norm - 1) >= float(NORM_TOLERANCE_SQL)


@dataclass(frozen=True)
class PlanItem:
    product_id: str
    source_sha: str
    text: str
    text_sha: str
    reasons: tuple[str, ...]  # empty = current


def _stored_rows(session: Session) -> dict[str, dict]:
    return {row["product_id"]: dict(row) for row in session.execute(_METADATA_SQL).mappings()}


def plan_embeddings(session: Session, spec: EmbeddingModelSpec) -> list[PlanItem]:
    require_table(session)
    records, failed = load_catalog_records(session)
    if failed:
        shown = ", ".join(failed[:5])
        raise EmbeddingError(
            f"{len(failed)} product(s) cannot be embedded because their stored rows do not "
            f"validate (e.g. {shown}); nothing was written. Run the catalog check first"
        )
    stored = _stored_rows(session)
    return [
        _item(record, source_sha, stored.get(record.product_id), spec)
        for record, source_sha in records
    ]


def _item(record: CatalogRecord, source_sha: str, row: dict | None, spec) -> PlanItem:
    body = build_embedding_text(record)
    digest = text_sha256(body)
    return PlanItem(
        record.product_id,
        source_sha,
        body,
        digest,
        tuple(stale_reasons(row, spec, source_sha, digest)),
    )


@dataclass
class EmbedResult:
    products: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    changed_during_run: int = 0  # product content changed after the plan: not written, rerun
    already_current: int = 0  # made current by a concurrent run between plan and write
    reasons: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        found = " ".join(f"{name}={self.reasons[name]}" for name in STALE_REASONS[1:])
        return (
            f"products={self.products} inserted={self.inserted} updated={self.updated} "
            f"unchanged={self.unchanged} changed_during_run={self.changed_during_run} "
            f"already_current={self.already_current} stale_found=({found})"
        )


def _encode(embedder: Embedder, items: list[PlanItem]) -> list[list[float]]:
    spec = embedder.spec
    texts = [item.text for item in items]
    counts = embedder.count_tokens(texts, "document")
    over = [
        item.product_id for item, n in zip(items, counts, strict=True) if n > spec.max_seq_length
    ]
    if over:
        raise EmbeddingError(
            f"{len(over)} embedding text(s) exceed the model limit of {spec.max_seq_length} "
            f"tokens (e.g. {', '.join(over[:5])}); texts are never truncated, nothing was written"
        )
    vectors: list[list[float]] = []
    for start in range(0, len(texts), ENCODE_BATCH):
        batch = texts[start : start + ENCODE_BATCH]
        vectors += validate_vectors(
            embedder.embed_documents(batch),
            count=len(batch),
            dimension=spec.dimension,
            normalized=spec.normalize,
        )
    return vectors


def embed(engine: Engine, embedder: Embedder, *, rebuild_all: bool = False) -> EmbedResult:
    """Create missing and rebuild stale embeddings (or all with `rebuild_all`). All or nothing."""
    spec = embedder.spec
    check_spec(spec)
    # Phase 1: plan (read-only).
    with Session(engine) as session:
        plan = plan_embeddings(session, spec)
    result = EmbedResult(products=len(plan))
    for item in plan:
        if item.reasons:
            result.reasons[item.reasons[0]] += 1
    pending = [item for item in plan if item.reasons or rebuild_all]
    result.unchanged = len(plan) - len(pending)
    if not pending:
        return result

    # Phase 2: encode (no transaction). Any error propagates before anything is written.
    vectors = _encode(embedder, pending)

    # Phase 3: write (one transaction, embedding lock).
    with Session(engine) as session, session.begin():
        take_embedding_lock(session)
        ids = [item.product_id for item in pending]
        current_sha: dict[str, str] = {}
        for start in range(0, len(ids), _CHUNK):
            chunk = ids[start : start + _CHUNK]
            current_sha.update(
                session.execute(
                    select(Product.product_id, Product.content_sha256).where(
                        Product.product_id.in_(chunk)
                    )
                ).all()
            )
        wanted = set(ids)
        stored = {pid: row for pid, row in _stored_rows(session).items() if pid in wanted}
        params = []
        for item, vector in zip(pending, vectors, strict=True):
            if current_sha.get(item.product_id) != item.source_sha:
                result.changed_during_run += 1
                continue
            row = stored.get(item.product_id)
            if not rebuild_all and not stale_reasons(row, spec, item.source_sha, item.text_sha):
                result.already_current += 1
                continue
            if row is None:
                result.inserted += 1
            else:
                result.updated += 1
            params.append(
                {
                    "product_id": item.product_id,
                    "embedding": vector_literal(vector),
                    "model_id": spec.model_id,
                    "model_revision": spec.revision,
                    "dimension": spec.dimension,
                    "normalized": spec.normalize,
                    "text_version": EMBEDDING_TEXT_VERSION,
                    "config_sha": spec.config_sha256(),
                    "source_sha": item.source_sha,
                    "text_sha": item.text_sha,
                }
            )
        if params:
            session.execute(UPSERT_SQL, params)
    return result


# ---- status -----------------------------------------------------------------------------------

# `unvalidated_products`: stored catalog rows that no longer validate, so no embedding text can
# be built for them (the catalog check reports why); they keep the status INCOMPLETE.
STATUS_COUNTS = (*STALE_REASONS, "invalid_vector", "vector_mismatch", "unvalidated_products")


@dataclass(frozen=True)
class EmbeddingStatus:
    spec: EmbeddingModelSpec
    total_products: int
    total_embeddings: int
    counts: dict[str, int]  # per condition; one row can count under several conditions
    vectors_verified: bool = False

    @property
    def current(self) -> bool:
        return not any(self.counts.values())

    def describe(self) -> str:
        counts = " ".join(f"{name}={self.counts[name]}" for name in STATUS_COUNTS)
        verified = "yes" if self.vectors_verified else "no (use --verify-vectors)"
        return (
            f"products={self.total_products} embeddings={self.total_embeddings} {counts} "
            f"expected_model={self.spec.model_id}@{self.spec.revision} "
            f"dimension={self.spec.dimension} normalized={self.spec.normalize} "
            f"text_version={EMBEDDING_TEXT_VERSION} config_sha256={self.spec.config_sha256()} "
            f"vectors_reencoded={verified} state={'CURRENT' if self.current else 'INCOMPLETE'}"
        )


def classify(session: Session, spec: EmbeddingModelSpec) -> tuple[list[PlanItem], dict, int]:
    """(plan items, stored rows, number of products whose rows do not validate)."""
    require_table(session)
    records, failed = load_catalog_records(session)
    stored = _stored_rows(session)
    items = [_item(r, sha, stored.get(r.product_id), spec) for r, sha in records]
    return items, stored, len(failed)


def embedding_status(
    session: Session, spec: EmbeddingModelSpec, embedder: Embedder | None = None
) -> EmbeddingStatus:
    """Read-only. With `embedder`, also re-encodes current rows and compares the vectors."""
    items, stored, unvalidated = classify(session, spec)
    counts = dict.fromkeys(STATUS_COUNTS, 0)
    counts["unvalidated_products"] = unvalidated
    for item in items:
        for reason in item.reasons:
            counts[reason] += 1
    counts["invalid_vector"] = sum(1 for row in stored.values() if invalid_vector(row))
    if embedder is not None:
        counts["vector_mismatch"] = len(vector_mismatches(session, embedder, items))
    return EmbeddingStatus(
        spec=spec,
        total_products=session.scalar(select(func.count()).select_from(Product)) or 0,
        total_embeddings=len(stored),
        counts=counts,
        vectors_verified=embedder is not None,
    )


def vector_mismatches(session: Session, embedder: Embedder, items: list[PlanItem]) -> list[str]:
    """Ids of current rows whose stored vector differs from a fresh encoding of their text."""
    current = [item for item in items if not item.reasons]
    if not current:
        return []
    fresh = _encode(embedder, current)
    stored = dict(
        session.execute(
            select(Emb.product_id, Emb.embedding).where(
                Emb.product_id.in_([item.product_id for item in current])
            )
        ).all()
    )
    mismatched = []
    for item, vector in zip(current, fresh, strict=True):
        old = [float(v) for v in stored[item.product_id]]
        dot = math.fsum(a * b for a, b in zip(old, vector, strict=True))
        norms = math.sqrt(math.fsum(a * a for a in old)) * math.sqrt(
            math.fsum(b * b for b in vector)
        )
        if norms == 0 or dot / norms < REENCODE_MIN_COSINE:
            mismatched.append(item.product_id)
    return mismatched


__all__ = [
    "EmbedResult",
    "EmbedderUnavailable",
    "EmbeddingError",
    "EmbeddingStatus",
    "embed",
    "embedding_lock_key",
    "embedding_status",
]
