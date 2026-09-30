import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.integration


def _extension_version(settings, database):
    engine = create_engine(settings.database_url(database=database))
    try:
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    finally:
        engine.dispose()


def _vector_distance(settings, database):
    engine = create_engine(settings.database_url(database=database))
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")).scalar_one()
    finally:
        engine.dispose()


def test_upgrade_enables_working_pgvector(settings, scratch_database, migrator):
    assert _extension_version(settings, scratch_database) is None
    migrator.upgrade(scratch_database)
    assert _extension_version(settings, scratch_database) is not None
    assert _vector_distance(settings, scratch_database) == pytest.approx(1.0)


def test_upgrade_is_idempotent(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database)
    migrator.upgrade(scratch_database)
    assert _extension_version(settings, scratch_database) is not None


def test_downgrade_then_upgrade(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database)
    migrator.downgrade(scratch_database)
    assert _extension_version(settings, scratch_database) is None
    migrator.upgrade(scratch_database)
    assert _extension_version(settings, scratch_database) is not None


def test_migration_creates_no_tables(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database)
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        with engine.connect() as conn:
            tables = (
                conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
                .scalars()
                .all()
            )
    finally:
        engine.dispose()
    assert tables == ["alembic_version"]
