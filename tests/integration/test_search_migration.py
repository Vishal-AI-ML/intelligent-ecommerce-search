"""Migration 0003 on throwaway databases (never the development database)."""

import io
import re

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from catalog_support import PROVENANCE, SEED
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

import ecommerce_search.models  # noqa: F401
from ecommerce_search.catalog.audit import audit_database
from ecommerce_search.db.base import Base
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.search.cli import main
from ecommerce_search.search.indexing import index_status

pytestmark = pytest.mark.integration

TABLE = "product_search_documents"
GIN = "ix_product_search_documents_search_vector"
EXPECTED_CONSTRAINTS = {
    "pk_product_search_documents",
    "fk_product_search_documents_product_id_products",
    "ck_product_search_documents_document_version_not_blank",
    "ck_product_search_documents_source_content_sha256_format",
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


def test_round_trip_0002_0003_0002_head_and_head_is_a_no_op(settings, scratch_database, migrator):
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        migrator.upgrade(scratch_database, "0002")
        assert TABLE not in tables(engine) and revision(engine) == "0002"
        migrator.upgrade(scratch_database, "0003")
        assert TABLE in tables(engine) and revision(engine) == "0003"
        migrator.downgrade(scratch_database, "0002")
        assert TABLE not in tables(engine) and revision(engine) == "0002"
        assert {"products", "laptop_specs", "catalog_reviews"} <= tables(engine)  # M2 intact
        with engine.connect() as conn:  # the 0001 extension is untouched
            assert (
                conn.execute(
                    text("SELECT count(*) FROM pg_extension WHERE extname='vector'")
                ).scalar_one()
                == 1
            )
        migrator.upgrade(scratch_database)  # M4: head is now 0004 (adds product_embeddings)
        assert TABLE in tables(engine) and revision(engine) == "0004"
        before = tables(engine)
        migrator.upgrade(scratch_database)  # head again: no-op
        assert tables(engine) == before and revision(engine) == "0004"
    finally:
        engine.dispose()


def test_schema_constraints_and_gin_index_definition(migrated_engine):
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
        index = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = :t AND indexname = :i"),
            {"t": TABLE, "i": GIN},
        ).scalar_one()
        indexes = set(
            conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": TABLE}
            )
            .scalars()
            .all()
        )
        columns = dict(
            conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = :t"
                ),
                {"t": TABLE},
            ).all()
        )
        udt = set(
            conn.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name = :t"
                ),
                {"t": TABLE},
            ).scalars()
        )
    assert constraints == EXPECTED_CONSTRAINTS
    assert indexes == {GIN, "pk_product_search_documents"}
    assert re.fullmatch(
        r"CREATE INDEX ix_product_search_documents_search_vector ON public\.product_search_documents "
        r"USING gin \(search_vector\)",
        index,
    )
    assert columns == {
        "product_id": "text",
        "document_version": "text",
        "source_content_sha256": "text",
        "search_vector": "tsvector",
        "built_at": "timestamp with time zone",
    }
    # No pgvector column in the lexical table. (M4 adds one only in product_embeddings.)
    assert "vector" not in udt
    assert all(len(name) <= 63 for name in constraints | indexes)


def test_model_and_migration_parity_for_the_search_table(migrated_engine):
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
    model_names = set(
        re.findall(
            r"CONSTRAINT (\w+)",
            str(CreateTable(Base.metadata.tables[TABLE]).compile(dialect=postgresql.dialect())),
        )
    )
    assert model_names == EXPECTED_CONSTRAINTS
    (index,) = Base.metadata.tables[TABLE].indexes
    assert index.name == GIN


def test_upgrade_sql_is_schema_only_and_creates_no_vector_objects(settings):
    config = Config("alembic.ini")
    config.attributes["database"] = "offline_sql_only"  # offline mode never connects
    config.output_buffer = io.StringIO()
    command.upgrade(config, "0002:0003", sql=True)
    sql = config.output_buffer.getvalue()
    assert "CREATE TABLE product_search_documents" in sql
    assert "CREATE INDEX ix_product_search_documents_search_vector" in sql and "USING gin" in sql
    assert not re.search(r"\bvector\(|CREATE EXTENSION|INSERT INTO", sql)


def test_populated_0002_database_upgrades_to_0003_and_an_explicit_reindex_builds_documents(
    settings, scratch_database, migrator, capsys
):
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        migrator.upgrade(scratch_database)
        assert ingest_file(engine, SEED, PROVENANCE).result.inserted == 240
        migrator.downgrade(scratch_database, "0002")  # drops only the derived search table
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM products")).scalar_one() == 240
        with Session(engine) as session:  # a 0002 audit still works and says why it is n/a
            report = audit_database(session)
        check = {c["name"]: c for c in report.checks}["embedding_search_text_leakage"]
        assert check["status"] == "not_applicable" and "0003" in check["note"]
        assert report.error_count == 0

        migrator.upgrade(scratch_database)  # schema only: the migration builds no documents
        with Session(engine) as session:
            status = index_status(session)
        assert (status.total_products, status.total_documents, status.missing) == (240, 0, 240)

        assert main(["reindex", "--database", scratch_database]) == 0
        out = capsys.readouterr().out
        assert f"target database: {scratch_database}" in out
        assert "inserted=240 updated=0 unchanged=0" in out
        with Session(engine) as session:
            assert index_status(session).current
    finally:
        engine.dispose()


# ---- I6 B: GIN index health ------------------------------------------------------------------


def test_gin_index_is_valid_ready_and_on_the_search_vector(migrated_engine):
    with migrated_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT i.indisvalid, i.indisready, i.indislive, am.amname, "
                "pg_get_indexdef(i.indexrelid), t.relname, "
                "array_agg(a.attname::text ORDER BY k.ord) "
                "FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_class t ON t.oid = i.indrelid "
                "JOIN pg_am am ON am.oid = c.relam "
                "CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) "
                "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum "
                "WHERE c.relname = :name "
                "GROUP BY 1, 2, 3, 4, 5, 6"
            ),
            {"name": GIN},
        ).one()
    valid, ready, live, method, definition, table, columns = row
    assert (valid, ready, live) == (True, True, True)
    assert (method, table, columns) == ("gin", TABLE, ["search_vector"])
    assert definition.endswith("USING gin (search_vector)")


# ---- I6 A: new code against a database still at revision 0002 -----------------------------------

LEAKS = ("product_search_documents", "INSERT", "relation ", "postgresql://", "psycopg", "Traceback")


ROW_COUNT_SQL = text(
    "SELECT 'catalog_datasets', count(*) FROM catalog_datasets UNION ALL "
    "SELECT 'raw_catalog_records', count(*) FROM raw_catalog_records UNION ALL "
    "SELECT 'products', count(*) FROM products UNION ALL "
    "SELECT 'laptop_specs', count(*) FROM laptop_specs UNION ALL "
    "SELECT 'phone_specs', count(*) FROM phone_specs UNION ALL "
    "SELECT 'shoe_specs', count(*) FROM shoe_specs UNION ALL "
    "SELECT 'headphone_specs', count(*) FROM headphone_specs"
)


def row_counts(engine) -> dict[str, int]:
    with engine.connect() as conn:
        return dict(conn.execute(ROW_COUNT_SQL).all())


def test_m3_code_refuses_cleanly_against_a_database_at_0002(
    settings, scratch_database, migrator, capsys
):
    from ecommerce_search.catalog.cli import main as catalog_main
    from ecommerce_search.ingestion.service import IngestRefused

    migrator.upgrade(scratch_database, "0002")
    engine = create_engine(settings.database_url(database=scratch_database))
    password = settings.postgres_password.get_secret_value()
    try:
        before_tables = tables(engine)
        with pytest.raises(IngestRefused) as excinfo:
            ingest_file(engine, SEED, PROVENANCE)
        message = str(excinfo.value)
        assert "alembic upgrade head" in message and "Nothing was written" in message
        for leaked in (*LEAKS, password, settings.postgres_host, settings.postgres_user):
            assert leaked not in message
        # Nothing was written: no dataset, raw, product, spec or search-document row exists.
        assert set(row_counts(engine).values()) == {0}
        assert tables(engine) == before_tables and TABLE not in tables(engine)

        # The catalog CLI exits non-zero with the same sanitized instruction.
        assert catalog_main(["ingest", "--database", scratch_database]) == 1
        captured = capsys.readouterr()
        assert "alembic upgrade head" in captured.err
        for leaked in (*LEAKS, password, settings.postgres_host, "Details:"):
            assert leaked not in captured.out + captured.err
        assert set(row_counts(engine).values()) == {0}

        # The search commands refuse in the same way (no table to read or write).
        for command_name in ("reindex", "status"):
            assert main([command_name, "--database", scratch_database]) == 1
            captured = capsys.readouterr()
            assert "alembic upgrade head" in captured.err
            for leaked in (*LEAKS, password):
                assert leaked not in captured.out + captured.err
    finally:
        engine.dispose()
