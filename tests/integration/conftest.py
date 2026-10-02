import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from ecommerce_search.config import Settings, get_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings from the real environment/.env (docker compose db on localhost)."""
    return get_settings()


@pytest.fixture
def scratch_database(settings):
    """A throwaway database on the dev server. The dev database itself is never touched."""
    name = f"ecommerce_search_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


class Migrator:
    def __init__(self, config: Config) -> None:
        self._config = config

    def upgrade(self, database: str, revision: str = "head") -> None:
        self._config.attributes["database"] = database
        command.upgrade(self._config, revision)

    def downgrade(self, database: str, revision: str = "base") -> None:
        self._config.attributes["database"] = database
        command.downgrade(self._config, revision)


@pytest.fixture
def migrator() -> Migrator:
    return Migrator(Config("alembic.ini"))


@pytest.fixture
def migrated_engine(settings, scratch_database, migrator):
    """Engine on a throwaway database migrated to head. Never the development database."""
    migrator.upgrade(scratch_database)
    engine = create_engine(settings.database_url(database=scratch_database))
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def seeded_engine(settings):
    """A throwaway database at head with the committed seed ingested (documents included).

    Module-scoped for speed: tests using it must treat the database as READ-ONLY. Tests that
    change data use `migrated_engine` instead. The development database is never touched."""
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    name = f"ecommerce_search_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = None
    try:
        Migrator(Config("alembic.ini")).upgrade(name)
        engine = create_engine(settings.database_url(database=name))
        outcome = ingest_file(engine, SEED, PROVENANCE)
        assert outcome.result is not None and outcome.result.inserted == 240
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def writable_seeded_engine(settings):
    """Like `seeded_engine`, but owned by one test module, which may change it (Milestone 4)."""
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    name = f"ecommerce_search_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = None
    try:
        Migrator(Config("alembic.ini")).upgrade(name)
        engine = create_engine(settings.database_url(database=name))
        outcome = ingest_file(engine, SEED, PROVENANCE)
        assert outcome.result is not None and outcome.result.inserted == 240
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
