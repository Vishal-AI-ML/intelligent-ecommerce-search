"""Database audit of the dense (embedding) index: check `dense_embedding_consistency`.

Read-only and model-free. For every product it compares the stored embedding with what the
active, reviewed configuration (ADR-006) would produce:

* **warnings** (operational state, healed by `embed`): missing embedding, stale model/revision,
  stale embedding-text version, stale configuration hash, and a source-content hash that no
  longer matches the product (the product changed since it was embedded);
* **errors** (malformed, tampered or invalid rows): an embedding-text hash that differs from the
  text rebuilt from the whitelisted catalog fields while the row claims the current
  configuration and content (the vector was not built from the approved text), a stored or
  declared dimension that differs from the configuration, a normalization flag that differs,
  and a zero, non-finite or non-unit stored vector.

A database not migrated to 0004 reports the check as not applicable with that reason. It also
returns the `dense_index` section of the quality report.
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.catalog.quality.findings import CheckOutcome, Finding, Severity
from ecommerce_search.embeddings.spec import (
    DEFAULT_EMBEDDING_MODEL_ID,
    EMBEDDING_MODELS,
    EmbeddingModelSpec,
)
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.search.dense import DISTANCE_METRIC
from ecommerce_search.search.dense_indexing import classify, invalid_vector

CHECK_NAME = "dense_embedding_consistency"
NO_TABLE_NOTE = (
    "Not applicable: the dense index (product_embeddings) does not exist in this database "
    "(not migrated to revision 0004)."
)
WARNING_REASONS = {
    "missing": "embedding is missing",
    "stale_model": "embedding was built with a different model or revision",
    "stale_text_version": "embedding was built from a different embedding-text version",
    "stale_config": "embedding was built with a different configuration",
    "source_hash_mismatch": "embedding was built from different product content (stale)",
}
ERROR_REASONS = {
    "dimension_mismatch": "embedding dimension differs from the configuration",
    "normalization_mismatch": "embedding normalization flag differs from the configuration",
}
COUNT_KEYS = (
    *WARNING_REASONS,
    "text_hash_mismatch",
    *ERROR_REASONS,
    "invalid_vector",
)


@dataclass(frozen=True)
class DenseAudit:
    outcome: CheckOutcome
    metadata: dict  # the `dense_index` section of the quality report


def audit_dense_index(session: Session, spec: EmbeddingModelSpec | None = None) -> DenseAudit:
    spec = spec or EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_ID]
    if session.scalar(text("SELECT to_regclass('public.product_embeddings')")) is None:
        return DenseAudit(
            CheckOutcome(0, (), NO_TABLE_NOTE), {"applicable": False, "reason": NO_TABLE_NOTE}
        )
    items, stored, invalid_rows = classify(session, spec)
    findings: list[Finding] = []
    counts = dict.fromkeys(COUNT_KEYS, 0)

    def add(severity: Severity, pid: str, message: str) -> None:
        findings.append(Finding(CHECK_NAME, severity, message, product_id=pid))

    for item in items:
        reasons = set(item.reasons)
        for reason, message in WARNING_REASONS.items():
            if reason in reasons:
                counts[reason] += 1
                add(Severity.WARNING, item.product_id, message)
        if "text_hash_mismatch" in reasons:
            counts["text_hash_mismatch"] += 1
            # When the row is also stale, the staleness warning above explains the difference.
            if not reasons & set(WARNING_REASONS):
                add(
                    Severity.ERROR,
                    item.product_id,
                    "embedding claims the current configuration and content but was not built "
                    "from the whitelisted embedding text (text hash mismatch)",
                )
        for reason, message in ERROR_REASONS.items():
            if reason in reasons:
                counts[reason] += 1
                add(Severity.ERROR, item.product_id, message)
        row = stored.get(item.product_id)
        if row is not None and invalid_vector(row):
            counts["invalid_vector"] += 1
            add(Severity.ERROR, item.product_id, "stored vector is zero, non-finite or not unit")
    metadata = {
        "applicable": True,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "dimension": spec.dimension,
        "normalized": spec.normalize,
        "max_seq_length": spec.max_seq_length,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "config_sha256": spec.config_sha256(),
        "distance_metric": DISTANCE_METRIC,
        "vector_index": "none (exact scan)",
        "products": len(items) + invalid_rows,
        "products_audited": len(items),
        "embeddings": len(stored),
        **counts,
    }
    return DenseAudit(CheckOutcome(len(items), tuple(findings)), metadata)
