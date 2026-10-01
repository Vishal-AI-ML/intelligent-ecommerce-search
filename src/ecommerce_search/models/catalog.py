"""Catalog tables (Milestone 2).

Raw source lines, normalized products, per-category attribute tables, dataset provenance and
append-only human review outcomes are separate tables. There are no search, embedding or
inferred-decision columns.

Enumerated values are TEXT + named CHECK constraints (not native ENUMs); the values come from
`ecommerce_search.catalog.taxonomy` and a test keeps them equal to the migration.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.db.base import Base


def in_values(column: str, enum_cls: type[Enum]) -> str:
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return f"{column} IN ({values})"


SHA256_FORMAT = "~ '^[0-9a-f]{64}$'"


class CatalogDataset(Base):
    """Provenance of one dataset version. Rights explanation is separate from any licence id."""

    __tablename__ = "catalog_datasets"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "dataset_version", name="uq_catalog_datasets_dataset_id_dataset_version"
        ),
        CheckConstraint(f"checksum_sha256 {SHA256_FORMAT}", name="checksum_sha256_format"),
        CheckConstraint("record_count >= 0", name="record_count_nonneg"),
        # Versions are positive integers without leading zeros, so ordering is numeric.
        CheckConstraint("dataset_version ~ '^[1-9][0-9]{0,8}$'", name="dataset_version_numeric"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_version: Mapped[str] = mapped_column(Text, nullable=False)
    source_description: Mapped[str] = mapped_column(Text, nullable=False)
    rights_note: Mapped[str] = mapped_column(Text, nullable=False)
    license_identifier: Mapped[str | None] = mapped_column(Text, nullable=True)
    authored_on: Mapped[date] = mapped_column(Date, nullable=False)
    generator: Mapped[str] = mapped_column(Text, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    transform_version: Mapped[str] = mapped_column(Text, nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(Text, nullable=False)
    rules_version: Mapped[str] = mapped_column(Text, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RawCatalogRecord(Base):
    """The exact source line, never updated."""

    __tablename__ = "raw_catalog_records"
    __table_args__ = (
        UniqueConstraint("dataset_pk", "source_key", name="uq_raw_catalog_records_dataset_source"),
        CheckConstraint("line_number >= 1", name="line_number_positive"),
        CheckConstraint(f"raw_sha256 {SHA256_FORMAT}", name="raw_sha256_format"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    dataset_pk: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("catalog_datasets.id", name="fk_raw_catalog_records_dataset_pk_datasets"),
        nullable=False,
    )
    source_key: Mapped[str] = mapped_column(Text, nullable=False)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_line: Mapped[str] = mapped_column(Text, nullable=False)
    raw_sha256: Mapped[str] = mapped_column(Text, nullable=False)


class Product(Base):
    """Normalized product. `dataset_pk` is the dataset version that last changed the row."""

    __tablename__ = "products"
    __table_args__ = (
        UniqueConstraint("product_id", "category", name="uq_products_product_id_category"),
        UniqueConstraint("raw_record_id", name="uq_products_raw_record_id"),
        CheckConstraint(
            r"product_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'", name="product_id_format"
        ),
        CheckConstraint("btrim(title) <> ''", name="title_not_blank"),
        CheckConstraint("btrim(brand) <> ''", name="brand_not_blank"),
        CheckConstraint(in_values("category", tx.Category), name="category_valid"),
        CheckConstraint("price > 0", name="price_positive"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_format"),
        CheckConstraint("rating IS NULL OR (rating >= 0 AND rating <= 5)", name="rating_range"),
        CheckConstraint("review_count IS NULL OR review_count >= 0", name="review_count_nonneg"),
        CheckConstraint(in_values("availability", tx.Availability), name="availability_valid"),
        CheckConstraint(in_values("source_type", tx.SourceType), name="source_type_valid"),
        CheckConstraint(
            "is_synthetic = (source_type = 'synthetic')", name="synthetic_flag_consistent"
        ),
        CheckConstraint(f"content_sha256 {SHA256_FORMAT}", name="content_sha256_format"),
        CheckConstraint("updated_at >= created_at", name="updated_not_before_created"),
    )

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset_pk: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("catalog_datasets.id", name="fk_products_dataset_pk_catalog_datasets"),
        nullable=False,
    )
    raw_record_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("raw_catalog_records.id", name="fk_products_raw_record_id_raw_records"),
        nullable=False,
    )
    seller_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    subcategory: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[Decimal | None] = mapped_column(Numeric(2, 1), nullable=True)
    review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    availability: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def _spec_fk(table: str) -> ForeignKeyConstraint:
    # The composite FK (product_id, category) -> products makes a spec row attachable only to a
    # product of the matching category (the CHECK pins the category value).
    return ForeignKeyConstraint(
        ["product_id", "category"],
        ["products.product_id", "products.category"],
        name=f"fk_{table}_product_id_category_products",
    )


class LaptopSpec(Base):
    __tablename__ = "laptop_specs"
    __table_args__ = (
        _spec_fk("laptop_specs"),
        CheckConstraint("category = 'laptop'", name="category_is_laptop"),
        CheckConstraint("ram_gb IS NULL OR ram_gb > 0", name="ram_gb_positive"),
        CheckConstraint("storage_gb IS NULL OR storage_gb > 0", name="storage_gb_positive"),
        CheckConstraint(
            "storage_type IS NULL OR " + in_values("storage_type", tx.StorageType),
            name="storage_type_valid",
        ),
        CheckConstraint(
            "storage_interface IS NULL OR " + in_values("storage_interface", tx.StorageInterface),
            name="storage_interface_valid",
        ),
        CheckConstraint(
            "storage_interface IS DISTINCT FROM 'NVME' OR COALESCE(storage_type = 'SSD', false)",
            name="nvme_requires_ssd",
        ),
        CheckConstraint(
            "screen_size_inches IS NULL OR screen_size_inches > 0", name="screen_size_positive"
        ),
        CheckConstraint("weight_kg IS NULL OR weight_kg > 0", name="weight_kg_positive"),
    )

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    ram_gb: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    storage_gb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_interface: Mapped[str | None] = mapped_column(Text, nullable=True)
    processor: Mapped[str | None] = mapped_column(Text, nullable=True)
    gpu: Mapped[str | None] = mapped_column(Text, nullable=True)
    screen_size_inches: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    operating_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)


class PhoneSpec(Base):
    __tablename__ = "phone_specs"
    __table_args__ = (
        _spec_fk("phone_specs"),
        CheckConstraint("category = 'phone'", name="category_is_phone"),
        CheckConstraint("ram_gb IS NULL OR ram_gb > 0", name="ram_gb_positive"),
        CheckConstraint("storage_gb IS NULL OR storage_gb > 0", name="storage_gb_positive"),
        CheckConstraint("battery_mah IS NULL OR battery_mah > 0", name="battery_mah_positive"),
        CheckConstraint(
            "screen_size_inches IS NULL OR screen_size_inches > 0", name="screen_size_positive"
        ),
    )

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    ram_gb: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    storage_gb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera: Mapped[str | None] = mapped_column(Text, nullable=True)
    battery_mah: Mapped[int | None] = mapped_column(Integer, nullable=True)
    screen_size_inches: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    operating_system: Mapped[str | None] = mapped_column(Text, nullable=True)


class ShoeSpec(Base):
    __tablename__ = "shoe_specs"
    __table_args__ = (
        _spec_fk("shoe_specs"),
        CheckConstraint("category = 'shoes'", name="category_is_shoes"),
        CheckConstraint("size IS NULL OR size > 0", name="size_positive"),
        CheckConstraint(
            "size_system IS NULL OR " + in_values("size_system", tx.SizeSystem),
            name="size_system_valid",
        ),
        CheckConstraint("(size IS NULL) = (size_system IS NULL)", name="size_and_system_together"),
        CheckConstraint("gender IS NULL OR " + in_values("gender", tx.Gender), name="gender_valid"),
    )

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    size: Mapped[Decimal | None] = mapped_column(Numeric(3, 1), nullable=True)
    size_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(Text, nullable=True)
    material: Mapped[str | None] = mapped_column(Text, nullable=True)
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)


class HeadphoneSpec(Base):
    __tablename__ = "headphone_specs"
    __table_args__ = (
        _spec_fk("headphone_specs"),
        CheckConstraint("category = 'headphones'", name="category_is_headphones"),
        CheckConstraint(
            "battery_life_hours IS NULL OR battery_life_hours > 0", name="battery_hours_positive"
        ),
        CheckConstraint(
            "connectivity IS NULL OR " + in_values("connectivity", tx.Connectivity),
            name="connectivity_valid",
        ),
        CheckConstraint(
            "NOT (connectivity = 'wired' AND wireless IS TRUE)", name="wired_not_wireless"
        ),
        CheckConstraint(
            "NOT (connectivity IN ('bluetooth', 'wireless_2_4ghz') AND wireless IS FALSE)",
            name="wireless_connectivity_consistent",
        ),
    )

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    wireless: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anc: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    battery_life_hours: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    connectivity: Mapped[str | None] = mapped_column(Text, nullable=True)


SPEC_TABLES: dict[tx.Category, type[Base]] = {
    tx.Category.LAPTOP: LaptopSpec,
    tx.Category.PHONE: PhoneSpec,
    tx.Category.SHOES: ShoeSpec,
    tx.Category.HEADPHONES: HeadphoneSpec,
}


class CatalogReview(Base):
    """Append-only human review outcomes (UPDATE/DELETE/TRUNCATE are blocked by triggers)."""

    __tablename__ = "catalog_reviews"
    __table_args__ = (
        UniqueConstraint(
            "review_batch_id",
            "product_id",
            "reviewer",
            name="uq_catalog_reviews_batch_product_reviewer",
        ),
        CheckConstraint(in_values("verdict", tx.ReviewVerdict), name="verdict_valid"),
        CheckConstraint("btrim(reviewer) <> ''", name="reviewer_not_blank"),
        CheckConstraint(f"sample_manifest_sha256 {SHA256_FORMAT}", name="manifest_sha256_format"),
        CheckConstraint(
            f"product_content_sha256 {SHA256_FORMAT}", name="product_content_sha256_format"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    review_batch_id: Mapped[str] = mapped_column(Text, nullable=False)
    product_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("products.product_id", name="fk_catalog_reviews_product_id_products"),
        nullable=False,
    )
    dataset_pk: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("catalog_datasets.id", name="fk_catalog_reviews_dataset_pk_datasets"),
        nullable=False,
    )
    # content_sha256 of the product version that was reviewed (checked before recording).
    product_content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    issue_fields: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(Text, nullable=False)
    sample_manifest_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
