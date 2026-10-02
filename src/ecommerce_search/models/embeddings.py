"""Product embeddings (Milestone 4): a derived, rebuildable dense index, one row per product.

Each row records exactly which model, immutable revision, configuration and embedding text
produced its vector, and which product content it was built from, so a stale row is always
detectable. Nothing here is a source of truth: the table can be dropped and regenerated with
`python -m ecommerce_search.search embed --database NAME`.

The CHECK constraints rely on pgvector 0.8.6 behaviour verified against a real server (the type
itself rejects NaN, infinite and wrong-dimension input; `vector_norm` exists).
"""

from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ecommerce_search.db.base import Base
from ecommerce_search.models.catalog import SHA256_FORMAT

# Fixed by migration 0004 and ADR-006. A model with another dimension needs a new migration.
EMBEDDING_DIMENSION = 384
# Unit-norm tolerance for normalized vectors (float4 storage rounding is far below this).
NORM_TOLERANCE_SQL = "0.001"


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"
    __table_args__ = (
        CheckConstraint("btrim(model_id) <> ''", name="model_id_not_blank"),
        CheckConstraint("model_revision ~ '^[0-9a-f]{40}$'", name="model_revision_format"),
        CheckConstraint("dimension > 0", name="dimension_positive"),
        CheckConstraint("vector_dims(embedding) = dimension", name="dimension_matches_vector"),
        CheckConstraint("vector_norm(embedding) > 0", name="embedding_nonzero"),
        CheckConstraint(
            f"NOT normalized OR abs(vector_norm(embedding) - 1) < {NORM_TOLERANCE_SQL}",
            name="embedding_normalized",
        ),
        CheckConstraint(
            "btrim(embedding_text_version) <> ''", name="embedding_text_version_not_blank"
        ),
        CheckConstraint(
            f"embedding_config_sha256 {SHA256_FORMAT}", name="embedding_config_sha256_format"
        ),
        CheckConstraint(
            f"source_content_sha256 {SHA256_FORMAT}", name="source_content_sha256_format"
        ),
        CheckConstraint(
            f"embedding_text_sha256 {SHA256_FORMAT}", name="embedding_text_sha256_format"
        ),
    )

    product_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("products.product_id", name="fk_product_embeddings_product_id_products"),
        primary_key=True,
    )
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIMENSION), nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    model_revision: Mapped[str] = mapped_column(Text, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    embedding_text_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_config_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    # products.content_sha256 of the product version this vector was built from.
    source_content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA-256 of the embedding text (builder output, before any model prefix).
    embedding_text_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
