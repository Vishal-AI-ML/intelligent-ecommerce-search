"""Hybrid source reads against real PostgreSQL scratch databases (fake model).

Evidence for the Milestone 5 transaction rule: both reads run on one connection, in one
REPEATABLE READ READ ONLY transaction (one snapshot), the setting is transaction-local (the pooled
connection is back to the server defaults afterwards), and the connection is released on success
and on failure. The development database is never touched.
"""

import pytest
from dense_support import embed_fake, sql
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.orm import Session

from ecommerce_search.db.engine import create_db_engine
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.search import hybrid as hybrid_search
from ecommerce_search.search.dense import dense_search
from ecommerce_search.search.lexical import lexical_search
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

SPEC = ALL_MINILM_L6_V2
PID = "SYN-SHO-0001"
QUERY = "Strato running shoes"
PROBE = text(
    "SELECT current_setting('transaction_isolation') AS isolation, "
    "current_setting('transaction_read_only') AS read_only, "
    "pg_backend_pid() AS pid, CAST(pg_current_snapshot() AS text) AS snapshot, "
    "txid_current_if_assigned() AS txid"
)


@pytest.fixture
def catalog(migrated_engine):
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    ingest_file(migrated_engine, SEED, PROVENANCE)
    embed_fake(migrated_engine)
    return migrated_engine


@pytest.fixture
def app_engine(settings, catalog):
    """The production engine factory (pre-ping, timeouts) with a single pooled connection."""
    scratch = settings.model_copy(
        update={"postgres_db": catalog.url.database, "db_pool_size": 1, "db_max_overflow": 0}
    )
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    engine = create_db_engine(scratch)
    yield engine
    engine.dispose()


def probe(session: Session) -> dict:
    return dict(session.execute(PROBE).mappings().one())


def defaults(engine) -> dict:
    with Session(engine) as session:
        return probe(session)


def instrument(monkeypatch, before_lexical=None, after_lexical=None, before_dense=None):
    """Wrap the real source functions inside `read_sources` with probes/hooks."""
    seen = {}

    def lexical(session, query, limit):
        seen["lexical"] = probe(session)
        if before_lexical:
            before_lexical(session)
        result = lexical_search(session, query, limit)
        if after_lexical:
            after_lexical(session)
        return result

    def dense(session, vector, spec, limit):
        seen["dense"] = probe(session)
        if before_dense:
            before_dense(session)
        return dense_search(session, vector, spec, limit)

    monkeypatch.setattr(hybrid_search, "lexical_search", lexical)
    monkeypatch.setattr(hybrid_search, "dense_search", dense)
    return seen


def record_statements(engine) -> list:
    log: list = []
    event.listen(engine, "begin", lambda conn: log.append("BEGIN"))
    event.listen(engine, "commit", lambda conn: log.append("COMMIT"))
    event.listen(engine, "rollback", lambda conn: log.append("ROLLBACK"))
    event.listen(
        engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, params, context, many: log.append(
            " ".join(statement.split())
        ),
    )
    return log


def test_both_reads_share_one_repeatable_read_read_only_snapshot(app_engine, monkeypatch):
    baseline = defaults(app_engine)
    assert (baseline["isolation"], baseline["read_only"]) == ("read committed", "off")
    seen = instrument(monkeypatch)
    log = record_statements(app_engine)
    vector = FakeEmbedder().embed_query(QUERY)
    with Session(app_engine) as session:
        lexical, dense = hybrid_search.read_sources(session, QUERY, vector, SPEC, 50, 50)
        assert not session.in_transaction()
    assert seen["lexical"] == seen["dense"]  # same backend, same snapshot, nothing written
    assert seen["lexical"]["isolation"] == "repeatable read"
    assert seen["lexical"]["read_only"] == "on"
    assert seen["lexical"]["pid"] == baseline["pid"] and seen["lexical"]["txid"] is None
    assert log[0] == "BEGIN"
    assert log[1] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert log[-1] == "COMMIT" and log.count("BEGIN") == 1
    assert app_engine.pool.checkedout() == 0
    assert lexical.hits and dense.hits


def test_setting_is_transaction_local_on_the_same_pooled_connection(app_engine):
    vector = FakeEmbedder().embed_query(QUERY)
    before = defaults(app_engine)
    with Session(app_engine) as session:
        hybrid_search.read_sources(session, QUERY, vector, SPEC, 10, 10)
        after_same_session = probe(session)
    after = defaults(app_engine)
    for state in (after_same_session, after):
        assert state["pid"] == before["pid"]  # pool_size=1: the very same connection
        assert (state["isolation"], state["read_only"]) == ("read committed", "off")


def test_a_commit_between_the_two_reads_is_invisible_to_the_dense_read(
    app_engine, catalog, monkeypatch
):
    def concurrent_change(session):
        # Another connection commits a catalog change after the lexical read has taken the
        # snapshot; content_sha256 is untouched so the embedding stays current.
        sql(catalog, "UPDATE products SET title = 'Changed Title' WHERE product_id = :p", p=PID)

    instrument(monkeypatch, after_lexical=concurrent_change)
    vector = FakeEmbedder().embed_query(QUERY)
    with Session(app_engine) as session:
        lexical, dense = hybrid_search.read_sources(session, QUERY, vector, SPEC, 50, 50)
    lexical_title = next(h.title for h in lexical.hits if h.product_id == PID)
    dense_title = next(h.title for h in dense.hits if h.product_id == PID)
    assert lexical_title == dense_title != "Changed Title"  # one snapshot for both reads
    with Session(app_engine) as session:  # a new transaction sees the committed change
        assert next(
            h.title for h in dense_search(session, vector, SPEC, 50).hits if h.product_id == PID
        ) == ("Changed Title")


def test_the_transaction_rejects_writes(app_engine, monkeypatch):
    def write(session):
        session.execute(text("UPDATE products SET title = title WHERE product_id = :p"), {"p": PID})

    instrument(monkeypatch, before_dense=write)
    vector = FakeEmbedder().embed_query(QUERY)
    with Session(app_engine) as session, pytest.raises(DBAPIError) as raised:
        hybrid_search.read_sources(session, QUERY, vector, SPEC, 10, 10)
    assert raised.value.orig.sqlstate == "25006"  # read_only_sql_transaction
    assert app_engine.pool.checkedout() == 0


@pytest.mark.parametrize("failing", ["lexical", "dense"])
def test_a_failed_read_rolls_back_and_releases_a_clean_connection(app_engine, monkeypatch, failing):
    def fail(session):
        session.execute(text("SELECT * FROM no_such_table"))

    hooks = {"before_lexical": fail} if failing == "lexical" else {"before_dense": fail}
    instrument(monkeypatch, **hooks)
    defaults(app_engine)  # first connect happens before statements are recorded
    log = record_statements(app_engine)
    vector = FakeEmbedder().embed_query(QUERY)
    session = Session(app_engine)
    with pytest.raises(ProgrammingError):
        hybrid_search.read_sources(session, QUERY, vector, SPEC, 10, 10)
    assert not session.in_transaction()
    assert app_engine.pool.checkedout() == 0  # released before the session is even closed
    session.close()
    assert log[0] == "BEGIN" and log[-1] == "ROLLBACK" and "COMMIT" not in log
    state = defaults(app_engine)  # the same connection is usable and back to defaults
    assert (state["isolation"], state["read_only"]) == ("read committed", "off")


def test_sources_equal_the_standalone_reads(app_engine):
    vector = FakeEmbedder().embed_query("wireless headphones")
    with Session(app_engine) as session:
        lexical, dense = hybrid_search.read_sources(
            session, "wireless headphones", vector, SPEC, 50, 50
        )
    with Session(app_engine) as session:
        assert lexical.hits == lexical_search(session, "wireless headphones", 50).hits
        assert dense.hits == dense_search(session, vector, SPEC, 50).hits
