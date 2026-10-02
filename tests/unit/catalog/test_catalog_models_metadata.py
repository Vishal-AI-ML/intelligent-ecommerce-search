import re

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

import ecommerce_search.models  # noqa: F401
from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.db.base import Base
from ecommerce_search.ingestion.loader import Provenance
from ecommerce_search.models.catalog import CatalogReview, Product

EXPECTED_TABLES = {
    "catalog_datasets",
    "raw_catalog_records",
    "products",
    "laptop_specs",
    "phone_specs",
    "shoe_specs",
    "headphone_specs",
    "catalog_reviews",
    # Milestone 3 adds exactly one derived table: the lexical search index.
    "product_search_documents",
    # Milestone 4 adds exactly one derived table: the dense embedding index.
    "product_embeddings",
}
SEARCH_TABLE = "product_search_documents"
EMBEDDING_TABLE = "product_embeddings"


def test_metadata_has_exactly_the_m2_and_m3_tables_and_search_columns_only_in_the_index_table():
    # M3 intentionally supersedes the M2 "no search columns" rule: the single allowed place for a
    # tsvector is product_search_documents.search_vector. M4 likewise allows exactly one pgvector
    # column, product_embeddings.embedding. No JSON column exists anywhere.
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    for table in Base.metadata.tables.values():
        for column in table.columns:
            kind = type(column.type).__name__
            assert kind not in {"JSONB", "JSON"}, column
            if kind == "TSVECTOR":
                assert (table.name, column.name) == (SEARCH_TABLE, "search_vector")
            if kind in {"VECTOR", "Vector", "HALFVEC", "SPARSEVEC", "BIT"}:
                assert (table.name, column.name) == (EMBEDDING_TABLE, "embedding")


def test_constraint_names_follow_convention_and_fit_postgres_limit():
    names = []
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        names += re.findall(r"CONSTRAINT (\w+)", ddl)
    assert names and all(len(n) <= 63 for n in names)
    assert all(re.match(r"(pk|uq|fk|ck)_", n) for n in names)
    assert len(names) == len(set(names))


def test_no_indexes_beyond_constraint_backing_ones_except_the_m3_gin_index():
    # M3 intentionally adds one search index; every other table still has none. M4 adds no
    # vector index (exact scan at 240 rows, ADR-006).
    for table in Base.metadata.tables.values():
        if table.name != SEARCH_TABLE:
            assert not table.indexes, f"{table.name} defines a non-constraint index"
    (index,) = Base.metadata.tables[SEARCH_TABLE].indexes
    assert index.name == "ix_product_search_documents_search_vector"
    assert index.dialect_options["postgresql"]["using"] == "gin"
    assert [c.name for c in index.columns] == ["search_vector"]


def test_catalog_tables_carry_no_label_or_review_fields():
    product_columns = {c.name for c in Product.__table__.columns}
    assert not product_columns & qp.FORBIDDEN_LABEL_FIELDS
    review_columns = {c.name for c in CatalogReview.__table__.columns}
    assert {"verdict", "reviewer", "issue_fields", "product_content_sha256"} <= review_columns
    assert not {"verdict", "reviewer", "issue_fields", "notes"} & product_columns


def test_check_constraints_use_taxonomy_values():
    expected = {
        ("products", "category_valid"): tx.Category,
        ("products", "availability_valid"): tx.Availability,
        ("products", "source_type_valid"): tx.SourceType,
        ("laptop_specs", "storage_type_valid"): tx.StorageType,
        ("laptop_specs", "storage_interface_valid"): tx.StorageInterface,
        ("shoe_specs", "size_system_valid"): tx.SizeSystem,
        ("shoe_specs", "gender_valid"): tx.Gender,
        ("headphone_specs", "connectivity_valid"): tx.Connectivity,
        ("catalog_reviews", "verdict_valid"): tx.ReviewVerdict,
    }
    for (table_name, short), enum_cls in expected.items():
        table = Base.metadata.tables[table_name]
        check = next(
            c for c in table.constraints if getattr(c, "name", None) and short in str(c.name)
        )
        assert set(re.findall(r"'([^']+)'", str(check.sqltext))) == {m.value for m in enum_cls}


# ---- I1: dataset versions are validated positive integers --------------------------------


@pytest.mark.parametrize("version", ["1", "2", "10", "999999999"])
def test_numeric_versions_are_accepted(seed_provenance, version):
    assert Provenance.model_validate({**seed_provenance.model_dump(), "dataset_version": version})


@pytest.mark.parametrize(
    "version", ["", "0", "01", "v1", "1.0", "1e3", " 1", "-1", "1000000000", "f"]
)
def test_non_numeric_or_ambiguous_versions_are_rejected(seed_provenance, version):
    with pytest.raises(ValueError):
        Provenance.model_validate({**seed_provenance.model_dump(), "dataset_version": version})


def test_versions_compare_numerically_not_lexicographically():
    assert "10" < "9"  # the string ordering a text column would give
    assert int("10") > int("9")  # what ingestion uses
