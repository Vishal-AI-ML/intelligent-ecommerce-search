"""Migration 0004 on throwaway databases (never the development database)."""

import re

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from catalog_support import PROVENANCE, SEED
from dense_support import SPEC, TABLE, sql
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.schema import CreateTable

import ecommerce_search.models  # noqa: F401
from ecommerce_search.db.base import Base
from ecommerce_search.ingestion.service import ingest_file

pytestmark = pytest.mark.integration

EXPECTED_CONSTRAINTS = {
    "pk_product_embeddings",
    "fk_product_embeddings_product_id_products",
    "ck_product_embeddings_model_id_not_blank",
    "ck_product_embeddings_model_revision_format",
    "ck_product_embeddings_dimension_positive",
    "ck_product_embeddings_dimension_matches_vector",
    "ck_product_embeddings_embedding_nonzero",
    "ck_product_embeddings_embedding_normalized",
    "ck_product_embeddings_embedding_text_version_not_blank",
    "ck_product_embeddings_embedding_config_sha256_format",
    "ck_product_embeddings_source_content_sha256_format",
    "ck_product_embeddings_embedding_text_sha256_format",
}
M3_AND_EARLIER = {
    "catalog_datasets",
    "catalog_reviews",
    "headphone_specs",
    "laptop_specs",
    "phone_specs",
    "products",
    "raw_catalog_records",
    "shoe_specs",
    "product_search_documents",
}


def tables(engine) -> set[str]:
    with engine.connect() as conn:
        return set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            ).scalars()
        )


def revision(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def test_round_trip_0003_0004_0003_head_and_head_is_a_no_op(settings, scratch_database, migrator):
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        migrator.upgrade(scratch_database, "0003")
        assert TABLE not in tables(engine) and revision(engine) == "0003"
        ingest_file(engine, SEED, PROVENANCE)  # data in M2/M3 tables must survive
        migrator.upgrade(scratch_database, "0004")
        assert TABLE in tables(engine) and revision(engine) == "0004"
        migrator.downgrade(scratch_database, "0003")
        assert TABLE not in tables(engine) and revision(engine) == "0003"
        assert tables(engine) >= M3_AND_EARLIER
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM products")).scalar_one() == 240
            assert (
                conn.execute(text("SELECT count(*) FROM product_search_documents")).scalar_one()
                == 240
            )
            assert (
                conn.execute(
                    text("SELECT count(*) FROM pg_extension WHERE extname='vector'")
                ).scalar_one()
                == 1
            )
        migrator.upgrade(scratch_database)
        assert TABLE in tables(engine) and revision(engine) == "0004"
        before = tables(engine)
        migrator.upgrade(scratch_database)  # head again: no-op
        assert tables(engine) == before and revision(engine) == "0004"
    finally:
        engine.dispose()


def test_schema_constraints_type_and_no_vector_index(migrated_engine):
    with migrated_engine.connect() as conn:
        constraints = set(
            conn.execute(
                text(
                    "SELECT c.conname FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
                    "WHERE t.relname = :t"
                ),
                {"t": TABLE},
            ).scalars()
        )
        indexes = set(
            conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": TABLE}
            ).scalars()
        )
        vector_type = conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'product_embeddings'::regclass AND attname = 'embedding'"
            )
        ).scalar_one()
        ann = conn.execute(
            text(
                "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_am a ON a.oid = c.relam WHERE a.amname IN ('hnsw', 'ivfflat')"
            )
        ).scalar_one()
    assert constraints == EXPECTED_CONSTRAINTS
    assert indexes == {"pk_product_embeddings"}  # exact scan: no HNSW/IVFFlat (ADR-006)
    assert vector_type == "vector(384)" and ann == 0
    assert all(len(name) <= 63 for name in constraints)


def test_model_and_migration_parity(migrated_engine):
    with migrated_engine.connect() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "compare_type": True,
                "include_object": lambda obj, name, type_, reflected, compare_to: (
                    not (type_ == "table" and name == "alembic_version")
                ),
            },
        )
        assert compare_metadata(context, Base.metadata) == []
    table = Base.metadata.tables[TABLE]
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert set(re.findall(r"CONSTRAINT (\w+)", ddl)) == EXPECTED_CONSTRAINTS
    assert "embedding VECTOR(384) NOT NULL" in ddl
    assert not table.indexes
    assert SPEC.dimension == table.c.embedding.type.dim == 384


# ---- constraints, verified against real pgvector ---------------------------------------------------


@pytest.fixture
def one_product(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    return migrated_engine


def insert(engine, vector: str, **overrides):
    values = {
        "product_id": "SYN-LAP-0001",
        "model_id": SPEC.model_id,
        "model_revision": SPEC.revision,
        "dimension": 384,
        "normalized": True,
        "text_version": "1",
        "config": "a" * 64,
        "source": "b" * 64,
        "text_sha": "c" * 64,
    }
    values.update(overrides)
    sql(
        engine,
        "INSERT INTO product_embeddings (product_id, embedding, model_id, model_revision, "
        "dimension, normalized, embedding_text_version, embedding_config_sha256, "
        "source_content_sha256, embedding_text_sha256) VALUES (:product_id, CAST(:v AS vector), "
        ":model_id, :model_revision, :dimension, :normalized, :text_version, :config, :source, "
        ":text_sha)",
        v=vector,
        **values,
    )


def unit(dim: int = 384, scale: float = 1.0) -> str:
    return "[" + ",".join([str(scale)] + ["0"] * (dim - 1)) + "]"


def test_a_well_formed_row_is_accepted(one_product):
    insert(one_product, unit())
    insert(one_product, unit(scale=2.0), product_id="SYN-LAP-0002", normalized=False)


@pytest.mark.parametrize(
    ("vector", "overrides", "error", "constraint"),
    [
        (unit(383), {}, DataError, None),  # pgvector: expected 384 dimensions
        (unit(385), {}, DataError, None),
        ("[" + ",".join(["NaN"] + ["0"] * 383) + "]", {}, DataError, None),
        ("[" + ",".join(["Infinity"] + ["0"] * 383) + "]", {}, DataError, None),
        (unit(scale=0.0), {}, IntegrityError, "embedding_nonzero"),
        (unit(scale=2.0), {}, IntegrityError, "embedding_normalized"),
        (unit(scale=0.99), {}, IntegrityError, "embedding_normalized"),
        (unit(), {"dimension": 768}, IntegrityError, "dimension_matches_vector"),
        # 0 also contradicts the vector's 384 dimensions: either check may report it
        (unit(), {"dimension": 0}, IntegrityError, ("dimension_positive", "dimension_matches")),
        (unit(), {"model_revision": "main"}, IntegrityError, "model_revision_format"),
        (unit(), {"model_id": "  "}, IntegrityError, "model_id_not_blank"),
        (unit(), {"text_version": ""}, IntegrityError, "embedding_text_version_not_blank"),
        (unit(), {"config": "XYZ"}, IntegrityError, "embedding_config_sha256_format"),
        (unit(), {"source": "b" * 63}, IntegrityError, "source_content_sha256_format"),
        (unit(), {"text_sha": "C" * 64}, IntegrityError, "embedding_text_sha256_format"),
        (unit(), {"product_id": "NO-SUCH"}, IntegrityError, "product_id_products"),
    ],
)
def test_invalid_rows_are_rejected_by_the_database(
    one_product, vector, overrides, error, constraint
):
    with pytest.raises(error) as info:
        insert(one_product, vector, **overrides)
    if constraint:
        name = str(info.value.orig.diag.constraint_name)
        options = constraint if isinstance(constraint, tuple) else (constraint,)
        assert any(option in name for option in options), name
    with one_product.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM product_embeddings")).scalar_one() == 0


def test_cosine_distance_against_a_zero_vector_is_nan_which_the_check_prevents(one_product):
    # Why `embedding_nonzero` matters: pgvector returns NaN for cosine distance with a zero vector.
    with one_product.connect() as conn:
        value = conn.execute(text("SELECT '[0,0]'::vector <=> '[1,0]'::vector")).scalar_one()
    assert value != value  # NaN
