import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ecommerce_search.db.checks import check_database
from ecommerce_search.db.engine import create_db_engine, create_session_factory

pytestmark = pytest.mark.integration


def test_connects_and_reports_postgres_16(settings):
    engine = create_db_engine(settings)
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar_one() == 1
            assert conn.execute(text("SHOW server_version_num")).scalar_one().startswith("16")
    finally:
        engine.dispose()


def test_check_database_ok(settings):
    engine = create_db_engine(settings)
    try:
        with create_session_factory(engine)() as session:
            assert check_database(session).ok
    finally:
        engine.dispose()


def test_statement_timeout_is_applied_and_enforced(settings):
    engine = create_db_engine(settings.model_copy(update={"db_statement_timeout_ms": 1500}))
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar_one() == "1500ms"
            with pytest.raises(OperationalError) as excinfo:
                conn.execute(text("SELECT pg_sleep(10)"))
            assert "statement timeout" in str(excinfo.value.orig)
    finally:
        engine.dispose()
