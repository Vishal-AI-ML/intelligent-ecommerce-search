"""Milestone 7 filtered lexical/dense retrieval against real PostgreSQL scratch databases (fake
embedder only; the development database is never touched).

Evidence:

* the predeclared truth table (`filter_support.TRUTH_TABLE`) parses as declared and its
  hand-written include/exclude/empty expectations hold in the independent catalog oracle;
* lexical and dense SQL eligibility each equal the oracle (dense also accounts for embedding
  freshness, recomputed independently);
* for every row, both sources and k in {1, 5, 50}, the filtered list equals the oracle-filtered
  prefix of the unfiltered full-depth ranking, with contiguous ranks and unchanged scores;
* EXPLAIN shows every filter predicate below the Limit node;
* both reads share one REPEATABLE READ READ ONLY snapshot, roll back on error and release
  their connection.
"""

import dataclasses
from decimal import Decimal

import pytest
from dense_support import embed_fake, sql
from filter_support import (
    FULL_DEPTH,
    K_VALUES,
    MISSING_EMBEDDING,
    MISSING_LAPTOP_SPEC,
    MISSING_PHONE_SPEC,
    NOT_CURRENT,
    NULL_RAM_LAPTOP,
    NULL_STORAGE_TYPE_LAPTOP,
    STALE_EMBEDDING,
    TRUTH_TABLE,
    TRUTH_TABLE_IDS,
    apply_mutations,
    catalog_facts,
    current_embedding_ids,
    oracle_eligible,
)
from search_support import oracle_matches, oracle_tokens
from sqlalchemy import event, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from ecommerce_search.catalog.taxonomy import Category, StorageInterface, StorageType
from ecommerce_search.db.engine import create_db_engine
from ecommerce_search.decision import DeterministicDecisionProvider, understand_normalized_query
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.filtering import FilterField, FilterSpec, derive_filters
from ecommerce_search.search import dense, filtered, lexical
from ecommerce_search.search.dense import dense_search
from ecommerce_search.search.filtered import (
    filtered_dense_search,
    filtered_lexical_search,
    read_filtered_sources,
)
from ecommerce_search.search.lexical import lexical_search
from ecommerce_search.search.query import normalize_query
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

SPEC = ALL_MINILM_L6_V2
ROWS = pytest.mark.parametrize("row", TRUTH_TABLE, ids=TRUTH_TABLE_IDS)


@pytest.fixture(scope="module")
def world(writable_seeded_engine):
    """Module-owned scratch catalog: seed + fake embeddings + the deliberate MUTATIONS.

    Read-only after setup. The lexical token oracle is taken before the mutations because the
    FTS documents are not rebuilt (they still describe the ingested catalog)."""
    engine = writable_seeded_engine
    assert engine.url.database.startswith("ecommerce_search_test_")
    embed_fake(engine)
    tokens = oracle_tokens(engine)
    apply_mutations(engine)
    return {
        "engine": engine,
        "tokens": tokens,
        "facts": catalog_facts(engine),
        "current": current_embedding_ids(engine, SPEC),
    }


def vector(query: str) -> list[float]:
    return FakeEmbedder(SPEC).embed_query(query)


def ids(hits) -> list[str]:
    return [hit.product_id for hit in hits]


# ------------------------------------------------------------------- truth table and oracle


def test_mutated_catalog_state(world):
    facts = world["facts"]
    assert len(facts) == 240
    assert facts[MISSING_LAPTOP_SPEC]["laptop"] is None
    assert facts[MISSING_PHONE_SPEC]["phone"] is None
    assert facts[NULL_RAM_LAPTOP]["laptop"]["ram_gb"] is None
    assert facts[NULL_STORAGE_TYPE_LAPTOP]["laptop"]["storage_type"] is None
    assert world["current"] == set(facts) - NOT_CURRENT


@pytest.mark.parametrize("row", [r for r in TRUTH_TABLE if r.parsed], ids=lambda r: r.id)
def test_parsed_rows_derive_the_declared_spec(row):
    query = normalize_query(row.query, 500)
    assert query == row.query  # the table holds normalized text: it reaches SQL unchanged
    decision = understand_normalized_query(query, DeterministicDecisionProvider())
    assert derive_filters(decision).spec == row.spec


@ROWS
def test_oracle_meets_the_predeclared_expectations(world, row):
    eligible = oracle_eligible(world["facts"], row.spec)
    assert (len(eligible) == 0) == (row.expect == "empty")
    assert row.include <= eligible
    assert not row.exclude & eligible
    if row.spec.is_empty():
        assert eligible == set(world["facts"])


def test_truth_table_is_non_vacuous_for_both_sources(world):
    """Guards the parity tests against becoming trivially empty == empty (for example after a
    seed or oracle change): filters must visibly narrow non-empty candidate sets in both
    sources, both to a smaller non-empty set and to nothing. The floors are deliberately low."""
    narrowed = {"lexical": 0, "dense": 0}
    emptied = {"lexical": 0, "dense": 0}
    for row in TRUTH_TABLE:
        eligible = oracle_eligible(world["facts"], row.spec)
        candidates = {
            "lexical": oracle_matches(world["tokens"], row.query),
            "dense": world["current"],
        }
        for source, before in candidates.items():
            after = before & eligible
            narrowed[source] += 0 < len(after) < len(before)
            emptied[source] += len(after) == 0 < len(before)
    for source in ("lexical", "dense"):
        assert narrowed[source] >= 5, (source, narrowed)
        assert emptied[source] >= 3, (source, emptied)


@ROWS
def test_dense_sql_eligibility_equals_the_oracle(world, row):
    with Session(world["engine"]) as session:
        hits = filtered_dense_search(session, vector(row.query), SPEC, row.spec, FULL_DEPTH).hits
    found = ids(hits)
    assert len(found) == len(set(found))
    expected = oracle_eligible(world["facts"], row.spec) & world["current"]
    assert set(found) == expected
    assert not set(found) & NOT_CURRENT  # stale/missing embeddings only shorten the list


@ROWS
def test_lexical_sql_eligibility_equals_the_oracle(world, row):
    with Session(world["engine"]) as session:
        hits = filtered_lexical_search(session, row.query, row.spec, FULL_DEPTH).hits
    found = ids(hits)
    assert len(found) == len(set(found))
    text_matches = oracle_matches(world["tokens"], row.query)
    assert set(found) == oracle_eligible(world["facts"], row.spec) & text_matches


# ------------------------------------------------------------------- retrieval prefix


def _assert_prefix(filtered_hits, unfiltered_hits, eligible, k):
    expected = [hit for hit in unfiltered_hits if hit.product_id in eligible][:k]
    assert [h.rank for h in filtered_hits] == list(range(1, len(filtered_hits) + 1))
    # Same products, order, scores and columns; only the positional rank is renumbered.
    assert filtered_hits == [
        dataclasses.replace(hit, rank=position) for position, hit in enumerate(expected, 1)
    ]


@ROWS
@pytest.mark.parametrize("source", ["lexical", "dense"])
def test_filtered_results_are_the_oracle_filtered_prefix(world, row, source):
    eligible = oracle_eligible(world["facts"], row.spec)
    with Session(world["engine"]) as session:
        if source == "lexical":
            unfiltered = lexical_search(session, row.query, FULL_DEPTH).hits
            assert set(ids(unfiltered)) == oracle_matches(world["tokens"], row.query)
        else:
            qvec = vector(row.query)
            unfiltered = dense_search(session, qvec, SPEC, FULL_DEPTH).hits
            # Freshness accounted explicitly: full-depth dense = every current product.
            assert set(ids(unfiltered)) == world["current"]
        for k in K_VALUES:
            if source == "lexical":
                hits = filtered_lexical_search(session, row.query, row.spec, k).hits
            else:
                hits = filtered_dense_search(session, qvec, SPEC, row.spec, k).hits
            _assert_prefix(hits, unfiltered, eligible, k)


@pytest.mark.parametrize("row", TRUTH_TABLE[:6] + TRUTH_TABLE[-6:], ids=lambda r: r.id)
def test_read_filtered_sources_equals_the_standalone_filtered_reads(world, row):
    qvec = vector(row.query)
    with Session(world["engine"]) as session:
        lexical_result, dense_result = read_filtered_sources(
            session, row.query, qvec, SPEC, row.spec, 50, 50
        )
        assert not session.in_transaction()
    with Session(world["engine"]) as session:
        assert lexical_result.hits == filtered_lexical_search(session, row.query, row.spec, 50).hits
        assert dense_result.hits == filtered_dense_search(session, qvec, SPEC, row.spec, 50).hits


# ------------------------------------------------------------------- spec rows and freshness


SPEC_FIELD_FILTERS = {
    "ram_gb": {"ram_gb": 8},
    "storage_gb": {"storage_gb": 256},
    "storage_type": {"storage_type": StorageType.SSD},
    "storage_interface": {
        "storage_type": StorageType.SSD,
        "storage_interface": StorageInterface.NVME,
    },
}


def _both_sources(world, query, spec) -> tuple[set[str], set[str]]:
    with Session(world["engine"]) as session:
        lex = filtered_lexical_search(session, query, spec, FULL_DEPTH).hits
        den = filtered_dense_search(session, vector(query), SPEC, spec, FULL_DEPTH).hits
    return set(ids(lex)), set(ids(den))


def test_missing_laptop_spec_row_excludes_only_under_spec_filters(world):
    base = {"category": Category.LAPTOP, "brand": "HP"}
    lex, den = _both_sources(world, "hp laptop", FilterSpec(**base))
    assert MISSING_LAPTOP_SPEC in lex and MISSING_LAPTOP_SPEC in den
    for extra in SPEC_FIELD_FILTERS.values():
        lex, den = _both_sources(world, "hp laptop", FilterSpec(**base, **extra))
        assert MISSING_LAPTOP_SPEC not in lex | den
    # Its ingested document still says 8GB RAM / 256GB SSD: text alone never satisfies a filter.
    assert {"8gb", "256gb", "ssd"} <= world["tokens"][MISSING_LAPTOP_SPEC]


def test_missing_phone_spec_row_excludes_only_under_capacity_filters(world):
    base = {"category": Category.PHONE, "brand": "Samsung"}
    lex, den = _both_sources(world, "samsung phone", FilterSpec(**base))
    assert MISSING_PHONE_SPEC in lex and MISSING_PHONE_SPEC in den
    for extra in ({"ram_gb": 8}, {"storage_gb": 256}):
        lex, den = _both_sources(world, "samsung phone", FilterSpec(**base, **extra))
        assert MISSING_PHONE_SPEC not in lex | den


def test_null_spec_values_never_match(world):
    _, den = _both_sources(world, "hp laptop", FilterSpec(ram_gb=8))
    assert NULL_RAM_LAPTOP not in den
    _, den = _both_sources(world, "hp laptop", FilterSpec(storage_gb=512))
    assert NULL_RAM_LAPTOP in den  # only the NULL column fails
    _, den = _both_sources(world, "hp laptop", FilterSpec(storage_type=StorageType.SSD))
    assert NULL_STORAGE_TYPE_LAPTOP not in den
    _, den = _both_sources(world, "hp laptop", FilterSpec(ram_gb=8, storage_gb=512))
    assert NULL_STORAGE_TYPE_LAPTOP in den


def test_stale_and_missing_embeddings_obey_current_dense_semantics(world):
    spec = FilterSpec(category=Category.LAPTOP, ram_gb=8)
    lex, den = _both_sources(world, "laptop 8gb ram", spec)
    assert {STALE_EMBEDDING, MISSING_EMBEDDING} <= oracle_eligible(world["facts"], spec)
    assert {STALE_EMBEDDING, MISSING_EMBEDDING} <= lex  # lexical does not depend on vectors
    assert not {STALE_EMBEDDING, MISSING_EMBEDDING} & den


# ------------------------------------------------------------------- EXPLAIN

COND_KEYS = ("Filter", "Join Filter", "Hash Cond", "Merge Cond", "Index Cond", "Recheck Cond")
FIELD_EVIDENCE = {
    FilterField.CATEGORY: ("category",),
    FilterField.BRAND: ("brand",),
    FilterField.RAM_GB: ("ram_gb",),
    FilterField.STORAGE_GB: ("storage_gb",),
    FilterField.STORAGE_TYPE: ("storage_type",),
    FilterField.STORAGE_INTERFACE: ("storage_interface",),
    FilterField.MIN_PRICE: ("price >=",),
    FilterField.MAX_PRICE: ("price <=",),
}
EXPLAIN_SPECS = {
    "category": FilterSpec(category=Category.LAPTOP),
    "brand": FilterSpec(brand="HP"),
    "ram": FilterSpec(ram_gb=8),
    "storage": FilterSpec(storage_gb=256),
    "nvme": FilterSpec(storage_type=StorageType.SSD, storage_interface=StorageInterface.NVME),
    "price": FilterSpec(min_price=Decimal("30000.00"), max_price=Decimal("90000.00")),
    "all": next(row.spec for row in TRUTH_TABLE if row.id == "full"),
}


def _nodes(node):
    yield node
    for child in node.get("Plans", []):
        yield from _nodes(child)


def _conditions(nodes) -> str:
    return " ".join(str(node.get(key, "")) for node in nodes for key in COND_KEYS)


def _explain(engine, statement, params) -> dict:
    with engine.connect() as conn:
        plan = conn.execute(text("EXPLAIN (FORMAT JSON) " + statement.text), params).scalar_one()
    return plan[0]["Plan"]


@pytest.mark.parametrize("source", ["lexical", "dense"])
@pytest.mark.parametrize("spec", EXPLAIN_SPECS.values(), ids=EXPLAIN_SPECS.keys())
def test_filter_predicates_sit_below_the_limit(world, source, spec):
    fields = filtered.active_fields(spec)
    if source == "lexical":
        statement = filtered.lexical_statement(fields)
        params = lexical.retrieval_params("laptop", 5) | filtered.filter_params(spec)
    else:
        statement = filtered.dense_statement(fields)
        params = dense.retrieval_params(vector("laptop"), SPEC, 5) | filtered.filter_params(spec)
    root = _explain(world["engine"], statement, params)
    limits = [node for node in _nodes(root) if node["Node Type"] == "Limit"]
    assert len(limits) == 1
    limit = limits[0]
    below = list(_nodes(limit))[1:]
    above = [node for node in _nodes(root) if not any(node is n for n in _nodes(limit))]
    assert not _conditions([limit, *above]).strip()  # nothing filters at or above the Limit
    assert not any("Relation Name" in node for node in above)
    conditions = _conditions(below)
    relations = {node.get("Relation Name") for node in below}
    for field in fields:
        for evidence in FIELD_EVIDENCE[field]:
            assert evidence in conditions, (field, conditions)
    if FilterField.RAM_GB in fields or FilterField.STORAGE_GB in fields:
        assert {"laptop_specs", "phone_specs"} <= relations
    if FilterField.STORAGE_TYPE in fields:
        assert "laptop_specs" in relations


# ------------------------------------------------------------------- transaction behaviour

PROBE = text(
    "SELECT current_setting('transaction_isolation') AS isolation, "
    "current_setting('transaction_read_only') AS read_only, "
    "pg_backend_pid() AS pid, CAST(pg_current_snapshot() AS text) AS snapshot, "
    "txid_current_if_assigned() AS txid"
)
SNAPSHOT_PID = "SYN-LAP-0008"  # HP 58999: leaves the window below when repriced
SNAPSHOT_SPEC = FilterSpec(category=Category.LAPTOP, brand="HP", max_price=Decimal("60000.00"))


@pytest.fixture
def catalog(migrated_engine):
    from catalog_support import PROVENANCE, SEED

    from ecommerce_search.ingestion.service import ingest_file

    assert migrated_engine.url.database.startswith("ecommerce_search_test_")
    ingest_file(migrated_engine, SEED, PROVENANCE)
    embed_fake(migrated_engine)
    return migrated_engine


@pytest.fixture
def app_engine(settings, catalog):
    """The production engine factory with a single pooled connection, on the scratch database."""
    scratch = settings.model_copy(
        update={"postgres_db": catalog.url.database, "db_pool_size": 1, "db_max_overflow": 0}
    )
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    engine = create_db_engine(scratch)
    checkouts = {"out": 0, "in": 0}
    event.listen(engine, "checkout", lambda *a: checkouts.__setitem__("out", checkouts["out"] + 1))
    event.listen(engine, "checkin", lambda *a: checkouts.__setitem__("in", checkouts["in"] + 1))
    engine.checkouts = checkouts
    yield engine
    engine.dispose()


def probe(session) -> dict:
    return dict(session.execute(PROBE).mappings().one())


def defaults(engine) -> dict:
    with Session(engine) as session:
        return probe(session)


def instrument(monkeypatch, after_lexical=None, before_dense=None):
    seen = {}

    def lexical_read(session, query, filters, limit):
        seen["lexical"] = probe(session)
        result = filtered_lexical_search(session, query, filters, limit)
        if after_lexical:
            after_lexical(session)
        return result

    def dense_read(session, query_vector, spec, filters, limit):
        seen["dense"] = probe(session)
        if before_dense:
            before_dense(session)
        return filtered_dense_search(session, query_vector, spec, filters, limit)

    monkeypatch.setattr(filtered, "filtered_lexical_search", lexical_read)
    monkeypatch.setattr(filtered, "filtered_dense_search", dense_read)
    return seen


def record(engine) -> list:
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


def test_both_filtered_reads_share_one_read_only_snapshot(app_engine, monkeypatch):
    baseline = defaults(app_engine)
    seen = instrument(monkeypatch)
    log = record(app_engine)
    with Session(app_engine) as session:
        lexical_result, dense_result = read_filtered_sources(
            session, "hp laptop", vector("hp laptop"), SPEC, SNAPSHOT_SPEC, 50, 50
        )
        assert not session.in_transaction()
    assert seen["lexical"] == seen["dense"]
    assert (seen["lexical"]["isolation"], seen["lexical"]["read_only"]) == ("repeatable read", "on")
    assert seen["lexical"]["pid"] == baseline["pid"] and seen["lexical"]["txid"] is None
    assert log[0] == "BEGIN"
    assert log[1] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert log[-1] == "COMMIT" and log.count("BEGIN") == 1
    assert app_engine.pool.checkedout() == 0
    assert app_engine.checkouts["out"] == app_engine.checkouts["in"]
    assert lexical_result.hits and dense_result.hits
    after = defaults(app_engine)
    assert (after["isolation"], after["read_only"]) == ("read committed", "off")


def test_a_commit_between_the_reads_does_not_change_filter_eligibility(
    app_engine, catalog, monkeypatch
):
    def reprice(session):
        # Another connection commits a price outside the filter window after the snapshot.
        sql(catalog, "UPDATE products SET price = 70000 WHERE product_id = :p", p=SNAPSHOT_PID)

    instrument(monkeypatch, after_lexical=reprice)
    qvec = vector("hp laptop")
    with Session(app_engine) as session:
        lexical_result, dense_result = read_filtered_sources(
            session, "hp laptop", qvec, SPEC, SNAPSHOT_SPEC, 50, 50
        )
    assert SNAPSHOT_PID in ids(lexical_result.hits)
    assert SNAPSHOT_PID in ids(dense_result.hits)  # same snapshot: old price still eligible
    with Session(app_engine) as session:  # a new transaction sees the committed price
        fresh = filtered_dense_search(session, qvec, SPEC, SNAPSHOT_SPEC, 50).hits
    assert SNAPSHOT_PID not in ids(fresh)


@pytest.mark.parametrize("failing", ["lexical", "dense"])
def test_a_failed_read_rolls_back_and_releases_the_connection(app_engine, monkeypatch, failing):
    def fail(session):
        session.execute(text("SELECT * FROM no_such_table"))

    if failing == "lexical":
        instrument(monkeypatch, after_lexical=fail)
    else:
        instrument(monkeypatch, before_dense=fail)
    defaults(app_engine)
    log = record(app_engine)
    session = Session(app_engine)
    with pytest.raises(ProgrammingError):
        read_filtered_sources(
            session, "hp laptop", vector("hp laptop"), SPEC, SNAPSHOT_SPEC, 10, 10
        )
    assert not session.in_transaction()
    assert app_engine.pool.checkedout() == 0
    session.close()
    assert log[0] == "BEGIN" and log[-1] == "ROLLBACK" and "COMMIT" not in log
    assert app_engine.checkouts["out"] == app_engine.checkouts["in"]
    state = defaults(app_engine)
    assert (state["isolation"], state["read_only"]) == ("read committed", "off")


def test_an_open_transaction_is_refused_without_database_work(app_engine):
    with Session(app_engine) as session:
        session.execute(text("SELECT 1"))
        assert session.in_transaction()
        log = record(app_engine)
        with pytest.raises(RuntimeError):
            read_filtered_sources(session, "q", vector("q"), SPEC, FilterSpec(), 5, 5)
        assert log == []  # refused before any statement, BEGIN or COMMIT
        session.rollback()
    assert app_engine.pool.checkedout() == 0
