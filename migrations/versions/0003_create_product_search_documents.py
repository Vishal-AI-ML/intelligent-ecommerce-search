"""create product search documents

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-01

Creates the Milestone 3 lexical search table and its GIN index. The DDL is written out
literally (no application imports). The upgrade creates structure only: documents are built by
the application (`python -m ecommerce_search.search reindex --database NAME`, and by ingestion).
No vector column or index is created and the pgvector extension from 0001 is not touched.

Downgrade drops the derived search table (it holds no source data; reindex rebuilds it).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "product_search_documents",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("document_version", sa.Text(), nullable=False),
        sa.Column("source_content_sha256", sa.Text(), nullable=False),
        sa.Column("search_vector", TSVECTOR(), nullable=False),
        sa.Column(
            "built_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["products.product_id"],
            name="fk_product_search_documents_product_id_products",
        ),
        sa.PrimaryKeyConstraint("product_id", name="pk_product_search_documents"),
        sa.CheckConstraint(
            "btrim(document_version) <> ''",
            name=op.f("ck_product_search_documents_document_version_not_blank"),
        ),
        sa.CheckConstraint(
            "source_content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_product_search_documents_source_content_sha256_format"),
        ),
    )
    op.create_index(
        "ix_product_search_documents_search_vector",
        "product_search_documents",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_product_search_documents_search_vector", table_name="product_search_documents"
    )
    op.drop_table("product_search_documents")
