"""Search documents (Milestone 3): a derived, rebuildable lexical index, one row per product.

Only the weighted `tsvector` is stored (no copy of catalog text). `document_version` and
`source_content_sha256` make a stale row detectable. Nothing here is a source of truth: the
table can be dropped and rebuilt from `products` and the spec tables.
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ecommerce_search.db.base import Base
from ecommerce_search.models.catalog import SHA256_FORMAT


class ProductSearchDocument(Base):
    __tablename__ = "product_search_documents"
    __table_args__ = (
        CheckConstraint("btrim(document_version) <> ''", name="document_version_not_blank"),
        CheckConstraint(
            f"source_content_sha256 {SHA256_FORMAT}", name="source_content_sha256_format"
        ),
        Index("ix_product_search_documents_search_vector", "search_vector", postgresql_using="gin"),
    )

    product_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("products.product_id", name="fk_product_search_documents_product_id_products"),
        primary_key=True,
    )
    document_version: Mapped[str] = mapped_column(Text, nullable=False)
    # products.content_sha256 of the product version this document was built from.
    source_content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[str] = mapped_column(TSVECTOR, nullable=False)
    built_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
