"""Search-document synchronization, reindex and the database audit (scratch databases only)."""

import json
import threading
import time

import pytest
from catalog_support import (
    PROVENANCE,
    SEED,
    counts,
    edit_line,
    make_variant,
    seed_lines,
)
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from ecommerce_search.catalog.audit import audit_database
from ecommerce_search.ingestion import service as service_module
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.models.search import ProductSearchDocument
from ecommerce_search.search.cli import main
from ecommerce_search.search.documents import DOCUMENT_VERSION
from ecommerce_search.search.indexing import search_lock_key

pytestmark = pytest.mark.integration

CHECK = "embedding_search_text_leakage"


def documents(engine) -> dict[str, tuple]:
    """product_id -> (version, source hash, built_at, vector text)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT product_id, document_version, source_content_sha256, built_at, "
                "CAST(search_vector AS text) FROM product_search_documents"
            )
        ).all()
    return {r[0]: tuple(r[1:]) for r in rows}


def check_outcome(engine):
    with Session(engine) as session:
        report = audit_database(session)
    return {c["name"]: c for c in report.checks}[CHECK], report


def sql(engine, statement: str, **params) -> None:
    with engine.begin() as conn:
        conn.execute(text(statement), params)


# ---- ingestion creates and refreshes documents in the same transaction -------------------------


def test_ingestion_creates_one_current_document_per_product(migrated_engine):
    result = ingest_file(migrated_engine, SEED, PROVENANCE).result
    assert result.inserted == 240 and result.search_documents_written == 240
    assert "search_documents_written=240" in result.summary()
    docs = documents(migrated_engine)
    assert len(docs) == 240
    with migrated_engine.connect() as conn:
        hashes = dict(conn.execute(text("SELECT product_id, content_sha256 FROM products")).all())
    assert {pid: d[1] for pid, d in docs.items()} == hashes
    assert {d[0] for d in docs.values()} == {DOCUMENT_VERSION}


def test_same_version_reingest_changes_no_document_and_keeps_built_at(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before = documents(migrated_engine)
    result = ingest_file(migrated_engine, SEED, PROVENANCE).result
    assert result.unchanged == 240 and result.search_documents_written == 0
    assert documents(migrated_engine) == before  # includes every built_at and vector


def test_a_newer_version_refreshes_only_affected_documents_and_keeps_absent_products(
    migrated_engine, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before = documents(migrated_engine)
    lines = seed_lines()
    lines[0] = edit_line(lines[0], title="HP Aurelia Renamed Zebrabook Laptop")  # searchable text
    lines[1] = edit_line(lines[1], price=123456)  # not searchable, but the content hash changes
    dropped = json.loads(lines.pop())["product_id"]
    path, prov = make_variant(tmp_path, lines, version="2")
    result = ingest_file(migrated_engine, path, prov).result
    assert (result.updated, result.unchanged) == (2, 237)
    assert result.search_documents_written == 2 and result.absent_product_ids == [dropped]
    after = documents(migrated_engine)
    changed = {pid for pid in before if before[pid] != after[pid]}
    assert changed == {"SYN-LAP-0001", "SYN-LAP-0002"}
    assert "zebrabook" in after["SYN-LAP-0001"][3] and "zebrabook" not in before["SYN-LAP-0001"][3]
    assert after["SYN-LAP-0002"][3] == before["SYN-LAP-0002"][3]  # same words, new source hash
    assert after["SYN-LAP-0002"][1] != before["SYN-LAP-0002"][1]
    assert after[dropped] == before[dropped]  # absent product keeps its valid document
    assert counts(migrated_engine)["products"] == 240


def test_a_failure_after_documents_are_written_rolls_back_catalog_and_documents(
    migrated_engine, monkeypatch, tmp_path
):
    real = service_module.sync_documents

    def write_then_fail(session, items):
        real(session, items)
        raise RuntimeError("simulated failure after the documents were written")

    monkeypatch.setattr(service_module, "sync_documents", write_then_fail)
    with pytest.raises(RuntimeError, match="simulated"):
        ingest_file(migrated_engine, SEED, PROVENANCE)
    assert counts(migrated_engine)["products"] == 0 and documents(migrated_engine) == {}

    monkeypatch.undo()
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before_docs = documents(migrated_engine)
    with migrated_engine.connect() as conn:
        before_products = conn.execute(
            text("SELECT product_id, content_sha256, updated_at FROM products ORDER BY 1")
        ).all()
    lines = seed_lines()
    lines[0] = edit_line(lines[0], title="HP Rolled Back Laptop")
    path, prov = make_variant(tmp_path, lines, version="2", name="rb")
    monkeypatch.setattr(service_module, "sync_documents", write_then_fail)
    with pytest.raises(RuntimeError, match="simulated"):
        ingest_file(migrated_engine, path, prov)
    assert documents(migrated_engine) == before_docs
    with migrated_engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT product_id, content_sha256, updated_at FROM products ORDER BY 1")
            ).all()
            == before_products
        )
        assert conn.execute(text("SELECT count(*) FROM catalog_datasets")).scalar_one() == 1


# ---- reindex heals missing and stale documents ---------------------------------------------------


def test_reindex_heals_missing_and_stale_documents_and_preserves_current_ones(
    migrated_engine, scratch_database, capsys
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before = documents(migrated_engine)
    ids = sorted(before)
    missing, bad_version, bad_hash = ids[:3], ids[3], ids[4]
    sql(
        migrated_engine,
        "DELETE FROM product_search_documents WHERE product_id = ANY(:ids)",
        ids=missing,
    )
    sql(
        migrated_engine,
        "UPDATE product_search_documents SET document_version = '0' WHERE product_id = :p",
        p=bad_version,
    )
    sql(
        migrated_engine,
        "UPDATE product_search_documents SET source_content_sha256 = :h WHERE product_id = :p",
        p=bad_hash,
        h="f" * 64,
    )
    assert main(["status", "--database", scratch_database, "--require-current"]) == 1
    out = capsys.readouterr().out
    assert "missing=3" in out and "stale_document_version=1" in out
    assert "content_hash_mismatch=1" in out and "INCOMPLETE" in out
    audit, report = check_outcome(migrated_engine)
    assert audit["status"] == "fail" and report.error_count == 5

    assert main(["reindex", "--database", scratch_database]) == 0
    out = capsys.readouterr().out
    assert f"target database: {scratch_database}" in out
    assert "inserted=3 updated=2 unchanged=235 stale_found=2" in out
    assert "document_version=1, content_hash=1" in out
    after = documents(migrated_engine)
    healed = set(missing) | {bad_version, bad_hash}
    for pid in ids:
        if pid in healed:
            assert after[pid][:2] == (DOCUMENT_VERSION, before[pid][1])
            assert after[pid][3] == before[pid][3]  # rebuilt deterministically: identical vector
        else:
            assert after[pid] == before[pid]  # current documents (and built_at) are untouched
    assert main(["status", "--database", scratch_database, "--require-current"]) == 0
    capsys.readouterr()

    assert main(["reindex", "--database", scratch_database]) == 0  # nothing left to do
    assert "inserted=0 updated=0 unchanged=240" in capsys.readouterr().out
    assert documents(migrated_engine) == after
    assert check_outcome(migrated_engine)[0]["status"] == "pass"


def test_reindex_is_all_or_nothing_when_a_product_cannot_be_indexed(
    migrated_engine, scratch_database, capsys
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sql(migrated_engine, "DELETE FROM product_search_documents WHERE product_id LIKE 'SYN-HDP-00%'")
    sql(migrated_engine, "DELETE FROM phone_specs WHERE product_id = 'SYN-PHN-0001'")
    before = documents(migrated_engine)
    assert main(["reindex", "--database", scratch_database]) == 1
    captured = capsys.readouterr()
    assert "SYN-PHN-0001" in captured.err and "nothing was written" in captured.err
    assert documents(migrated_engine) == before  # the healthy headphone documents were not built


def test_search_commands_require_an_explicit_database(capsys):
    for argv in (["reindex"], ["status"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
    assert "--database" in capsys.readouterr().err


def test_search_commands_print_only_the_database_name(
    migrated_engine, scratch_database, settings, capsys
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    for argv in (
        ["status", "--database", scratch_database],
        ["reindex", "--database", scratch_database],
    ):
        assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out.count("target database:") == 2
    for secret in (
        settings.postgres_password.get_secret_value(),
        "postgresql://",
        "@",
        "127.0.0.1",
    ):
        assert secret not in captured.out + captured.err


def test_search_commands_report_database_errors_without_details(capsys):
    assert main(["status", "--database", "no_such_database_for_m3_tests"]) == 1
    err = capsys.readouterr().err
    assert "database error" in err and "postgresql" not in err and "password" not in err.lower()


def test_reindex_waits_for_the_search_index_advisory_lock(migrated_engine, scratch_database):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sql(migrated_engine, "DELETE FROM product_search_documents WHERE product_id = 'SYN-LAP-0001'")
    holder = migrated_engine.connect()
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(search_lock_key())))
    outcome: dict = {}

    def run():
        try:
            outcome["code"] = main(["reindex", "--database", scratch_database])
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()

    def waiting() -> int:
        with migrated_engine.connect() as probe:
            return probe.execute(
                text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted")
            ).scalar_one()

    try:
        deadline = time.monotonic() + 30
        while waiting() != 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert waiting() == 1 and worker.is_alive()
        assert "SYN-LAP-0001" not in documents(migrated_engine)  # nothing written while blocked
    finally:
        transaction.rollback()
        holder.close()
    worker.join(120)
    assert not worker.is_alive() and outcome == {"code": 0}
    assert "SYN-LAP-0001" in documents(migrated_engine)


# ---- database audit: data-quality check 10 ---------------------------------------------------------


def test_clean_index_passes_check_10_and_an_empty_database_is_not_applicable(migrated_engine):
    empty, _ = check_outcome(migrated_engine)
    assert empty["status"] == "not_applicable" and empty["evaluated"] == 0
    ingest_file(migrated_engine, SEED, PROVENANCE)
    clean, report = check_outcome(migrated_engine)
    assert (clean["status"], clean["evaluated"], clean["findings"]) == ("pass", 240, 0)
    assert report.error_count == 0


@pytest.mark.parametrize(
    ("label", "statement", "message"),
    [
        (
            "missing document",
            "DELETE FROM product_search_documents WHERE product_id = 'SYN-LAP-0001'",
            "missing",
        ),
        (
            "stale document version",
            "UPDATE product_search_documents SET document_version = '0' "
            "WHERE product_id = 'SYN-LAP-0001'",
            "stale",
        ),
        (
            "source hash mismatch",
            "UPDATE product_search_documents SET source_content_sha256 = repeat('e', 64) "
            "WHERE product_id = 'SYN-LAP-0001'",
            "hash mismatch",
        ),
        (
            "tampered lexemes",
            "UPDATE product_search_documents SET search_vector = search_vector || "
            "to_tsvector('simple', 'unexpectedword') WHERE product_id = 'SYN-LAP-0001'",
            "differs from the vector rebuilt",
        ),
        (
            "removed lexemes",
            "UPDATE product_search_documents SET search_vector = to_tsvector('simple', 'hp') "
            "WHERE product_id = 'SYN-LAP-0001'",
            "differs from the vector rebuilt",
        ),
    ],
)
def test_check_10_detects_index_damage(migrated_engine, label, statement, message):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sql(migrated_engine, statement)
    outcome, report = check_outcome(migrated_engine)
    assert outcome["status"] == "fail", label
    findings = [f for f in report.findings if f.check == CHECK]
    assert [f.product_id for f in findings] == ["SYN-LAP-0001"], label
    assert any(message in f.message for f in findings), (label, [f.message for f in findings])
    assert report.error_count >= 1


def insert_review_note(engine, product_id: str, note: str) -> None:
    with engine.begin() as conn:
        dataset_pk = conn.execute(text("SELECT id FROM catalog_datasets")).scalar_one()
        content = conn.execute(
            text("SELECT content_sha256 FROM products WHERE product_id = :p"), {"p": product_id}
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO catalog_reviews (review_batch_id, product_id, dataset_pk, "
                "product_content_sha256, verdict, issue_fields, notes, reviewer, "
                "sample_manifest_sha256) VALUES ('test-batch', :p, :d, :h, 'accept', "
                ":note, :note, 'TEST-ONLY-REVIEWER', :h)"
            ),
            {"p": product_id, "d": dataset_pk, "h": content, "note": note},
        )


def matches(engine, query: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT count(*) FROM product_search_documents "
                "WHERE search_vector @@ plainto_tsquery('simple', CAST(:q AS text))"
            ),
            {"q": query},
        ).scalar_one()


def test_review_notes_and_provenance_never_enter_the_search_index(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    insert_review_note(migrated_engine, "SYN-LAP-0001", "zzreviewmarkerqq unsure wording")
    sql(migrated_engine, "DELETE FROM product_search_documents")  # force a full rebuild
    from ecommerce_search.search.indexing import reindex

    with Session(migrated_engine) as session, session.begin():
        assert reindex(session).inserted == 240
    # The bare word "synthetic" is legitimate catalog text (a shoe material), so only review
    # text and dataset/provenance identifiers are checked here.
    for forbidden in ("zzreviewmarkerqq", "synthetic-seed", "public_dataset"):
        assert matches(migrated_engine, forbidden) == 0, forbidden
    outcome, _ = check_outcome(migrated_engine)
    assert outcome["status"] == "pass"


def test_check_10_flags_injected_review_and_provenance_text(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    insert_review_note(migrated_engine, "SYN-LAP-0001", "zzreviewmarkerqq")
    sql(
        migrated_engine,
        "UPDATE product_search_documents SET search_vector = search_vector || "
        "to_tsvector('simple', 'zzreviewmarkerqq') WHERE product_id = 'SYN-LAP-0001'",
    )
    sql(
        migrated_engine,
        "UPDATE product_search_documents SET search_vector = search_vector || "
        "to_tsvector('simple', 'synthetic-seed') WHERE product_id = 'SYN-LAP-0002'",
    )
    outcome, report = check_outcome(migrated_engine)
    assert outcome["status"] == "fail"
    messages = {f.product_id: f.message for f in report.findings if f.check == CHECK}
    assert "forbidden review/provenance text" in messages["SYN-LAP-0001"]
    assert "zzreviewmarkerqq" in messages["SYN-LAP-0001"]
    assert "forbidden review/provenance text" in messages["SYN-LAP-0002"]


def test_documents_table_is_the_only_search_storage(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    with Session(migrated_engine) as session:
        assert session.scalar(select(func.count()).select_from(ProductSearchDocument)) == 240
    with migrated_engine.connect() as conn:
        product_columns = set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = 'products'"
                )
            ).scalars()
        )
    assert not product_columns & {"search_vector", "tsv", "document"}  # nothing on products


# ---- I4: search_index metadata in database quality reports -------------------------------------


def search_index_section(engine) -> dict:
    with Session(engine) as session:
        return audit_database(session).to_dict()["search_index"]


CLEAN_SECTION = {
    "applicable": True,
    "document_version": "1",
    "fts_config": "simple",
    "search_version": "v0_lexical",
    "products": 240,
    "products_audited": 240,
    "search_documents": 240,
    "missing": 0,
    "stale_version": 0,
    "hash_mismatch": 0,
    "tampered_vector": 0,
}


def test_clean_index_reports_exact_search_index_metadata(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    assert search_index_section(migrated_engine) == CLEAN_SECTION


@pytest.mark.parametrize(
    ("label", "statement", "changes"),
    [
        (
            "missing document",
            "DELETE FROM product_search_documents WHERE product_id = 'SYN-LAP-0001'",
            {"search_documents": 239, "missing": 1},
        ),
        (
            "stale version",
            "UPDATE product_search_documents SET document_version = '0' "
            "WHERE product_id = 'SYN-LAP-0001'",
            {"stale_version": 1},
        ),
        (
            "hash mismatch",
            "UPDATE product_search_documents SET source_content_sha256 = repeat('e', 64) "
            "WHERE product_id = 'SYN-LAP-0001'",
            {"hash_mismatch": 1},
        ),
        (
            "tampered vector",
            "UPDATE product_search_documents SET search_vector = search_vector || "
            "to_tsvector('simple', 'unexpectedword') WHERE product_id = 'SYN-LAP-0001'",
            {"tampered_vector": 1},
        ),
    ],
)
def test_damaged_index_reports_exact_search_index_metadata(
    migrated_engine, label, statement, changes
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sql(migrated_engine, statement)
    assert search_index_section(migrated_engine) == {**CLEAN_SECTION, **changes}, label


def test_database_report_files_carry_the_search_index_section(migrated_engine, tmp_path):
    from ecommerce_search.catalog.quality.report import write_report

    ingest_file(migrated_engine, SEED, PROVENANCE)
    with Session(migrated_engine) as session:
        report = audit_database(session)
    json_path, md_path = write_report(report, tmp_path, "audit")
    assert json.loads(json_path.read_text(encoding="utf-8"))["search_index"] == CLEAN_SECTION
    assert "## Search index" in md_path.read_text(encoding="utf-8")
    # The M2 catalog versions block is untouched: no document version is mixed into it.
    assert set(report.to_dict()["versions"]) == {
        "rules_version",
        "taxonomy_version",
        "transform_version",
    }


def test_a_database_without_the_search_table_reports_not_applicable_metadata(
    settings, scratch_database, migrator
):
    migrator.upgrade(scratch_database, "0002")
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        section = search_index_section(engine)
    finally:
        engine.dispose()
    assert section["applicable"] is False and "0003" in section["reason"]
    assert set(section) == {"applicable", "reason"}


# ---- I6 C: concurrent ingest and reindex --------------------------------------------------------


def advisory_locks(engine) -> list[tuple[int, bool, int, int]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT pid, granted, classid::bigint, objid::bigint FROM pg_locks "
                "WHERE locktype = 'advisory'"
            )
        ).all()
    return [tuple(r) for r in rows]


def lock_ids(key: int) -> tuple[int, int]:
    """(classid, objid) as pg_locks shows a 64-bit advisory key."""
    return (key >> 32) & 0xFFFFFFFF, key & 0xFFFFFFFF


def wait_until(predicate, timeout=60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_concurrent_ingest_and_reindex_serialize_without_deadlock_or_stale_documents(
    migrated_engine, scratch_database, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)  # version 1
    sql(
        migrated_engine,
        "DELETE FROM product_search_documents WHERE product_id IN "
        "('SYN-SHO-0001', 'SYN-SHO-0002', 'SYN-SHO-0003')",
    )
    lines = seed_lines()
    lines[0] = edit_line(lines[0], title="HP Aurelia Concurrent Zebrabook Laptop")
    lines[1] = edit_line(lines[1], price=654321)
    v2_path, v2_prov = make_variant(tmp_path, lines, version="2")

    search_ids = lock_ids(search_lock_key())
    dataset_ids = lock_ids(service_module.ingest_lock_key("synthetic-seed"))
    assert search_ids != dataset_ids

    holder = migrated_engine.connect()  # plays an in-flight search-index writer
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(search_lock_key())))
    outcome: dict = {}

    def run_reindex():
        try:
            outcome["reindex"] = main(["reindex", "--database", scratch_database])
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            outcome["reindex_error"] = exc

    def run_ingest():
        try:
            outcome["ingest"] = ingest_file(migrated_engine, v2_path, v2_prov)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            outcome["ingest_error"] = exc

    def waiting_on_search() -> set[int]:
        return {
            pid
            for pid, granted, classid, objid in advisory_locks(migrated_engine)
            if not granted and (classid, objid) == search_ids
        }

    reindex_thread = threading.Thread(target=run_reindex)
    ingest_thread = threading.Thread(target=run_ingest)
    try:
        reindex_thread.start()
        assert wait_until(lambda: len(waiting_on_search()) == 1), "reindex did not wait"
        ingest_thread.start()
        assert wait_until(lambda: len(waiting_on_search()) == 2), "ingest did not wait"

        locks = advisory_locks(migrated_engine)
        waiting = waiting_on_search()
        dataset_holders = {pid for pid, granted, c, o in locks if granted and (c, o) == dataset_ids}
        # Lock order observed: the ingesting backend already HOLDS the dataset lock while it
        # waits for the search lock; the reindexing backend never takes a dataset lock.
        assert len(dataset_holders) == 1 and dataset_holders <= waiting
        (reindex_pid,) = waiting - dataset_holders
        assert all((c, o) == search_ids for pid, _g, c, o in locks if pid == reindex_pid)
        deleted = {"SYN-SHO-0001", "SYN-SHO-0002", "SYN-SHO-0003"}
        assert not deleted & set(documents(migrated_engine))  # nothing written while blocked
    finally:
        transaction.rollback()  # release the held search lock: both writers proceed
        holder.close()
    reindex_thread.join(120)
    ingest_thread.join(120)
    assert not reindex_thread.is_alive() and not ingest_thread.is_alive(), "deadlock"
    assert "reindex_error" not in outcome and "ingest_error" not in outcome, outcome
    assert outcome["reindex"] == 0
    assert (outcome["ingest"].result.updated, outcome["ingest"].result.unchanged) == (2, 238)

    # No partial or stale state: every product has a current document, and the latest valid
    # content (version 2) won, in either interleaving.
    docs = documents(migrated_engine)
    with migrated_engine.connect() as conn:
        products = dict(conn.execute(text("SELECT product_id, content_sha256 FROM products")).all())
        title, dataset_version = conn.execute(
            text(
                "SELECT p.title, d.dataset_version FROM products p "
                "JOIN catalog_datasets d ON d.id = p.dataset_pk WHERE p.product_id = 'SYN-LAP-0001'"
            )
        ).one()
        dataset_count = conn.execute(text("SELECT count(*) FROM catalog_datasets")).scalar_one()
    assert len(products) == len(docs) == 240
    assert {pid: d[1] for pid, d in docs.items()} == products
    assert {d[0] for d in docs.values()} == {DOCUMENT_VERSION}
    assert (title, dataset_version, dataset_count) == (
        "HP Aurelia Concurrent Zebrabook Laptop",
        "2",
        2,
    )
    assert "zebrabook" in docs["SYN-LAP-0001"][3]
    assert check_outcome(migrated_engine)[0]["status"] == "pass"
    assert search_index_section(migrated_engine) == CLEAN_SECTION
