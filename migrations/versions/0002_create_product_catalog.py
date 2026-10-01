"""create product catalog

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-30

Creates the Milestone 2 catalog tables. The DDL is written out literally (no application
imports) so the migration stays frozen when application code changes. The pgvector extension
from 0001 is not touched, and no vector, tsvector or search columns are created.

Downgrade drops every catalog table and therefore destroys catalog data. Only run it against
a database you fully control.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SHA = "~ '^[0-9a-f]{64}$'"


def _ck(table: str, name: str, condition: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(condition, name=op.f(f"ck_{table}_{name}"))


def _spec_fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["product_id", "category"],
        ["products.product_id", "products.category"],
        name=f"fk_{table}_product_id_category_products",
    )


def _now() -> sa.TextClause:
    return sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "catalog_datasets",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("dataset_version", sa.Text(), nullable=False),
        sa.Column("source_description", sa.Text(), nullable=False),
        sa.Column("rights_note", sa.Text(), nullable=False),
        sa.Column("license_identifier", sa.Text(), nullable=True),
        sa.Column("authored_on", sa.Date(), nullable=False),
        sa.Column("generator", sa.Text(), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("transform_version", sa.Text(), nullable=False),
        sa.Column("taxonomy_version", sa.Text(), nullable=False),
        sa.Column("rules_version", sa.Text(), nullable=False),
        sa.Column("checksum_sha256", sa.Text(), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_catalog_datasets"),
        sa.UniqueConstraint(
            "dataset_id", "dataset_version", name="uq_catalog_datasets_dataset_id_dataset_version"
        ),
        _ck("catalog_datasets", "checksum_sha256_format", f"checksum_sha256 {SHA}"),
        _ck("catalog_datasets", "record_count_nonneg", "record_count >= 0"),
        _ck(
            "catalog_datasets",
            "dataset_version_numeric",
            "dataset_version ~ '^[1-9][0-9]{0,8}$'",
        ),
    )

    op.create_table(
        "raw_catalog_records",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("dataset_pk", sa.BigInteger(), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=False),
        sa.Column("raw_line", sa.Text(), nullable=False),
        sa.Column("raw_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_pk"],
            ["catalog_datasets.id"],
            name="fk_raw_catalog_records_dataset_pk_datasets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_raw_catalog_records"),
        sa.UniqueConstraint(
            "dataset_pk", "source_key", name="uq_raw_catalog_records_dataset_source"
        ),
        _ck("raw_catalog_records", "line_number_positive", "line_number >= 1"),
        _ck("raw_catalog_records", "raw_sha256_format", f"raw_sha256 {SHA}"),
    )

    op.create_table(
        "products",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("dataset_pk", sa.BigInteger(), nullable=False),
        sa.Column("raw_record_id", sa.BigInteger(), nullable=False),
        sa.Column("seller_id", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("subcategory", sa.Text(), nullable=True),
        sa.Column("brand", sa.Text(), nullable=False),
        sa.Column("price", sa.Numeric(12, 2), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("rating", sa.Numeric(2, 1), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=True),
        sa.Column("availability", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_pk"], ["catalog_datasets.id"], name="fk_products_dataset_pk_catalog_datasets"
        ),
        sa.ForeignKeyConstraint(
            ["raw_record_id"],
            ["raw_catalog_records.id"],
            name="fk_products_raw_record_id_raw_records",
        ),
        sa.PrimaryKeyConstraint("product_id", name="pk_products"),
        sa.UniqueConstraint("product_id", "category", name="uq_products_product_id_category"),
        sa.UniqueConstraint("raw_record_id", name="uq_products_raw_record_id"),
        _ck("products", "product_id_format", r"product_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'"),
        _ck("products", "title_not_blank", "btrim(title) <> ''"),
        _ck("products", "brand_not_blank", "btrim(brand) <> ''"),
        _ck(
            "products",
            "category_valid",
            "category IN ('laptop', 'phone', 'shoes', 'headphones')",
        ),
        _ck("products", "price_positive", "price > 0"),
        _ck("products", "currency_format", "currency ~ '^[A-Z]{3}$'"),
        _ck("products", "rating_range", "rating IS NULL OR (rating >= 0 AND rating <= 5)"),
        _ck("products", "review_count_nonneg", "review_count IS NULL OR review_count >= 0"),
        _ck(
            "products",
            "availability_valid",
            "availability IN ('in_stock', 'out_of_stock', 'discontinued')",
        ),
        _ck("products", "source_type_valid", "source_type IN ('synthetic', 'public_dataset')"),
        _ck("products", "synthetic_flag_consistent", "is_synthetic = (source_type = 'synthetic')"),
        _ck("products", "content_sha256_format", f"content_sha256 {SHA}"),
        _ck("products", "updated_not_before_created", "updated_at >= created_at"),
    )

    op.create_table(
        "laptop_specs",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("ram_gb", sa.SmallInteger(), nullable=True),
        sa.Column("storage_gb", sa.Integer(), nullable=True),
        sa.Column("storage_type", sa.Text(), nullable=True),
        sa.Column("storage_interface", sa.Text(), nullable=True),
        sa.Column("processor", sa.Text(), nullable=True),
        sa.Column("gpu", sa.Text(), nullable=True),
        sa.Column("screen_size_inches", sa.Numeric(4, 1), nullable=True),
        sa.Column("operating_system", sa.Text(), nullable=True),
        sa.Column("weight_kg", sa.Numeric(5, 2), nullable=True),
        _spec_fk("laptop_specs"),
        sa.PrimaryKeyConstraint("product_id", name="pk_laptop_specs"),
        _ck("laptop_specs", "category_is_laptop", "category = 'laptop'"),
        _ck("laptop_specs", "ram_gb_positive", "ram_gb IS NULL OR ram_gb > 0"),
        _ck("laptop_specs", "storage_gb_positive", "storage_gb IS NULL OR storage_gb > 0"),
        _ck(
            "laptop_specs",
            "storage_type_valid",
            "storage_type IS NULL OR storage_type IN ('SSD', 'HDD')",
        ),
        _ck(
            "laptop_specs",
            "storage_interface_valid",
            "storage_interface IS NULL OR storage_interface IN ('NVME', 'SATA')",
        ),
        _ck(
            "laptop_specs",
            "nvme_requires_ssd",
            "storage_interface IS DISTINCT FROM 'NVME' OR COALESCE(storage_type = 'SSD', false)",
        ),
        _ck(
            "laptop_specs",
            "screen_size_positive",
            "screen_size_inches IS NULL OR screen_size_inches > 0",
        ),
        _ck("laptop_specs", "weight_kg_positive", "weight_kg IS NULL OR weight_kg > 0"),
    )

    op.create_table(
        "phone_specs",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("ram_gb", sa.SmallInteger(), nullable=True),
        sa.Column("storage_gb", sa.Integer(), nullable=True),
        sa.Column("camera", sa.Text(), nullable=True),
        sa.Column("battery_mah", sa.Integer(), nullable=True),
        sa.Column("screen_size_inches", sa.Numeric(4, 1), nullable=True),
        sa.Column("operating_system", sa.Text(), nullable=True),
        _spec_fk("phone_specs"),
        sa.PrimaryKeyConstraint("product_id", name="pk_phone_specs"),
        _ck("phone_specs", "category_is_phone", "category = 'phone'"),
        _ck("phone_specs", "ram_gb_positive", "ram_gb IS NULL OR ram_gb > 0"),
        _ck("phone_specs", "storage_gb_positive", "storage_gb IS NULL OR storage_gb > 0"),
        _ck("phone_specs", "battery_mah_positive", "battery_mah IS NULL OR battery_mah > 0"),
        _ck(
            "phone_specs",
            "screen_size_positive",
            "screen_size_inches IS NULL OR screen_size_inches > 0",
        ),
    )

    op.create_table(
        "shoe_specs",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("size", sa.Numeric(3, 1), nullable=True),
        sa.Column("size_system", sa.Text(), nullable=True),
        sa.Column("color", sa.Text(), nullable=True),
        sa.Column("material", sa.Text(), nullable=True),
        sa.Column("gender", sa.Text(), nullable=True),
        _spec_fk("shoe_specs"),
        sa.PrimaryKeyConstraint("product_id", name="pk_shoe_specs"),
        _ck("shoe_specs", "category_is_shoes", "category = 'shoes'"),
        _ck("shoe_specs", "size_positive", "size IS NULL OR size > 0"),
        _ck(
            "shoe_specs",
            "size_system_valid",
            "size_system IS NULL OR size_system IN ('UK', 'US', 'EU')",
        ),
        _ck("shoe_specs", "size_and_system_together", "(size IS NULL) = (size_system IS NULL)"),
        _ck(
            "shoe_specs",
            "gender_valid",
            "gender IS NULL OR gender IN ('men', 'women', 'unisex', 'kids')",
        ),
    )

    op.create_table(
        "headphone_specs",
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("wireless", sa.Boolean(), nullable=True),
        sa.Column("anc", sa.Boolean(), nullable=True),
        sa.Column("battery_life_hours", sa.Numeric(5, 1), nullable=True),
        sa.Column("connectivity", sa.Text(), nullable=True),
        _spec_fk("headphone_specs"),
        sa.PrimaryKeyConstraint("product_id", name="pk_headphone_specs"),
        _ck("headphone_specs", "category_is_headphones", "category = 'headphones'"),
        _ck(
            "headphone_specs",
            "battery_hours_positive",
            "battery_life_hours IS NULL OR battery_life_hours > 0",
        ),
        _ck(
            "headphone_specs",
            "connectivity_valid",
            "connectivity IS NULL OR "
            "connectivity IN ('bluetooth', 'wired', 'usb', 'wireless_2_4ghz')",
        ),
        _ck(
            "headphone_specs",
            "wired_not_wireless",
            "NOT (connectivity = 'wired' AND wireless IS TRUE)",
        ),
        _ck(
            "headphone_specs",
            "wireless_connectivity_consistent",
            "NOT (connectivity IN ('bluetooth', 'wireless_2_4ghz') AND wireless IS FALSE)",
        ),
    )

    op.create_table(
        "catalog_reviews",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("review_batch_id", sa.Text(), nullable=False),
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("dataset_pk", sa.BigInteger(), nullable=False),
        sa.Column("product_content_sha256", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("issue_fields", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.Text(), nullable=False),
        sa.Column("sample_manifest_sha256", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_pk"], ["catalog_datasets.id"], name="fk_catalog_reviews_dataset_pk_datasets"
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.product_id"], name="fk_catalog_reviews_product_id_products"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_catalog_reviews"),
        sa.UniqueConstraint(
            "review_batch_id",
            "product_id",
            "reviewer",
            name="uq_catalog_reviews_batch_product_reviewer",
        ),
        _ck(
            "catalog_reviews",
            "verdict_valid",
            "verdict IN ('accept', 'needs_correction', 'unsure')",
        ),
        _ck("catalog_reviews", "reviewer_not_blank", "btrim(reviewer) <> ''"),
        _ck("catalog_reviews", "manifest_sha256_format", f"sample_manifest_sha256 {SHA}"),
        _ck(
            "catalog_reviews",
            "product_content_sha256_format",
            f"product_content_sha256 {SHA}",
        ),
    )

    # Human review outcomes are append-only: block UPDATE, DELETE and TRUNCATE.
    op.execute(
        """
        CREATE FUNCTION catalog_reviews_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'catalog_reviews is append-only';
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_catalog_reviews_no_update_delete "
        "BEFORE UPDATE OR DELETE ON catalog_reviews "
        "FOR EACH ROW EXECUTE FUNCTION catalog_reviews_append_only()"
    )
    op.execute(
        "CREATE TRIGGER trg_catalog_reviews_no_truncate "
        "BEFORE TRUNCATE ON catalog_reviews "
        "FOR EACH STATEMENT EXECUTE FUNCTION catalog_reviews_append_only()"
    )


def downgrade() -> None:
    # Destroys all catalog data. Dropping the table also drops its triggers.
    op.drop_table("catalog_reviews")
    op.execute("DROP FUNCTION catalog_reviews_append_only()")
    op.drop_table("headphone_specs")
    op.drop_table("shoe_specs")
    op.drop_table("phone_specs")
    op.drop_table("laptop_specs")
    op.drop_table("products")
    op.drop_table("raw_catalog_records")
    op.drop_table("catalog_datasets")
