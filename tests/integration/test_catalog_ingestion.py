"""Ingestion and audit tests on throwaway databases (never the development database)."""

import hashlib
import json
import threading
import time

import pytest
from catalog_support import (
    EMPTY_COUNTS,
    PROVENANCE,
    SEED,
    counts,
    edit_line,
    make_variant,
    seed_lines,
    snapshot,
)
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ecommerce_search.catalog.audit import audit_database
from ecommerce_search.catalog.cli import main
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.ingestion import loader as loader_module
from ecommerce_search.ingestion import service as service_module
from ecommerce_search.ingestion.service import IngestRefused, ingest_file, ingest_lock_key
from ecommerce_search.models.catalog import CatalogDataset, Product, RawCatalogRecord

pytestmark = pytest.mark.integration


# ---- full ingestion and idempotency -----------------------------------------------------


def test_full_seed_ingestion_then_idempotent_rerun(migrated_engine):
    first = ingest_file(migrated_engine, SEED, PROVENANCE)
    assert first.report.error_count == 0 and first.result is not None
    r1 = first.result
    assert (r1.inserted, r1.updated, r1.unchanged, r1.raw_records_inserted) == (240, 0, 0, 240)
    assert r1.dataset_created and r1.absent_product_ids == []
    assert counts(migrated_engine) == {
        "datasets": 1,
        "raw": 240,
        "products": 240,
        "specs": 240,
        "reviews": 0,
    }

    with Session(migrated_engine) as s:
        by_category = dict(
            s.execute(select(Product.category, func.count()).group_by(Product.category)).all()
        )
        assert by_category == {"laptop": 80, "phone": 60, "shoes": 50, "headphones": 50}
        dataset = s.scalars(select(CatalogDataset)).one()
        assert dataset.is_synthetic and dataset.license_identifier is None
        assert dataset.checksum_sha256 == hashlib.sha256(SEED.read_bytes()).hexdigest()
        assert dataset.record_count == 240
        raw = {r.source_key: r.raw_line for r in s.scalars(select(RawCatalogRecord))}
        file_lines = {json.loads(line)["product_id"]: line for line in seed_lines()}
        assert raw == file_lines  # raw lines preserved exactly, separate from normalized values
        assert all(
            p.is_synthetic and p.source_type == "synthetic" for p in s.scalars(select(Product))
        )
    before = snapshot(migrated_engine)

    second = ingest_file(migrated_engine, SEED, PROVENANCE)
    r2 = second.result
    assert (r2.inserted, r2.updated, r2.unchanged, r2.raw_records_inserted) == (0, 0, 240, 0)
    assert not r2.dataset_created
    assert snapshot(migrated_engine) == before  # includes every created_at/updated_at


def test_same_version_with_changed_checksum_is_refused_and_changes_nothing(
    migrated_engine, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before = snapshot(migrated_engine)
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=123456)
    path, prov = make_variant(tmp_path, lines, version="1")  # same version, different bytes
    with pytest.raises(IngestRefused, match="checksum"):
        ingest_file(migrated_engine, path, prov)
    assert snapshot(migrated_engine) == before


# ---- I1: versions are ordered numerically; older versions never roll data back -----------


def test_new_version_upserts_changed_products_and_never_deletes_absent_ones(
    migrated_engine, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    with Session(migrated_engine) as s:
        before = {p.product_id: p.updated_at for p in s.scalars(select(Product))}
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=123456)
    dropped = json.loads(lines.pop())["product_id"]
    path, prov = make_variant(tmp_path, lines, version="2")
    result = ingest_file(migrated_engine, path, prov).result
    assert (result.inserted, result.updated, result.unchanged) == (0, 1, 238)
    assert result.dataset_created and result.raw_records_inserted == 239
    assert result.absent_product_ids == [dropped]  # reported explicitly
    with Session(migrated_engine) as s:
        assert s.get(Product, dropped) is not None  # never deleted
        changed = s.get(Product, "SYN-LAP-0001")
        assert changed.price == 123456 and changed.updated_at > before["SYN-LAP-0001"]
        assert all(
            p.updated_at == before[p.product_id]
            for p in s.scalars(select(Product))
            if p.product_id != "SYN-LAP-0001"
        )
        assert s.scalar(select(func.count()).select_from(CatalogDataset)) == 2


def test_older_version_after_newer_is_refused_and_latest_data_is_preserved(
    migrated_engine, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)  # v1
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=123456)
    v2_path, v2_prov = make_variant(tmp_path, lines, version="2")
    ingest_file(migrated_engine, v2_path, v2_prov)  # v2
    after_v2 = snapshot(migrated_engine)

    with pytest.raises(IngestRefused, match="older than the already ingested version 2"):
        ingest_file(migrated_engine, SEED, PROVENANCE)  # v1 again: would revert the price
    assert snapshot(migrated_engine) == after_v2
    with Session(migrated_engine) as s:
        assert s.get(Product, "SYN-LAP-0001").price == 123456


def test_versions_are_compared_numerically_not_as_strings(migrated_engine, tmp_path):
    lines = seed_lines()
    nine, nine_prov = make_variant(tmp_path, lines, version="9", name="v9")
    lines[0] = edit_line(lines[0], price=222222)
    ten, ten_prov = make_variant(tmp_path, lines, version="10", name="v10")
    assert ingest_file(migrated_engine, nine, nine_prov).result is not None
    assert ingest_file(migrated_engine, ten, ten_prov).result.updated == 1  # "10" > "9"
    before = snapshot(migrated_engine)
    with pytest.raises(IngestRefused, match="older"):
        ingest_file(migrated_engine, nine, nine_prov)
    assert snapshot(migrated_engine) == before


def test_re_ingesting_the_latest_version_stays_idempotent_after_older_ones(
    migrated_engine, tmp_path
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=123456)
    v2_path, v2_prov = make_variant(tmp_path, lines, version="2")
    ingest_file(migrated_engine, v2_path, v2_prov)
    before = snapshot(migrated_engine)
    result = ingest_file(migrated_engine, v2_path, v2_prov).result
    assert (result.inserted, result.updated, result.unchanged) == (0, 0, 240)
    assert snapshot(migrated_engine) == before


def test_category_change_for_an_existing_product_is_refused_atomically(
    migrated_engine, tmp_path, raw_record
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    before = snapshot(migrated_engine)
    lines = seed_lines()
    lines[0] = json.dumps(raw_record("shoes", product_id="SYN-LAP-0001"))
    lines[1] = edit_line(lines[1], price=99999)  # would be updated before the refusal
    path, prov = make_variant(tmp_path, lines, version="2")
    with pytest.raises(IngestRefused, match="category"):
        ingest_file(migrated_engine, path, prov)
    assert snapshot(migrated_engine) == before


# ---- I9: provenance / version drift under a reused dataset version -----------------------


@pytest.fixture
def ingested(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    return snapshot(migrated_engine)


@pytest.mark.parametrize(
    ("field", "patch"),
    [
        ("transform_version", "transform"),
        ("taxonomy_version", "taxonomy"),
        ("rules_version", "rules"),
    ],
)
def test_changed_code_versions_require_a_new_dataset_version(
    migrated_engine, ingested, monkeypatch, tmp_path, field, patch
):
    changes = {}
    if patch == "transform":
        monkeypatch.setattr(loader_module, "TRANSFORM_VERSION", "2")
        monkeypatch.setattr(service_module, "TRANSFORM_VERSION", "2")
        changes["transform_version"] = "2"
    elif patch == "taxonomy":
        monkeypatch.setattr(loader_module, "TAXONOMY_VERSION", "2")
        monkeypatch.setattr(service_module, "TAXONOMY_VERSION", "2")
        changes["taxonomy_version"] = "2"
    else:
        monkeypatch.setattr(qp, "RULES_VERSION", "2")
    path, prov = make_variant(tmp_path, seed_lines(), version="1", **changes)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == hashlib.sha256(SEED.read_bytes()).hexdigest()
    )
    with pytest.raises(IngestRefused, match=field):
        ingest_file(migrated_engine, path, prov)
    assert snapshot(migrated_engine) == ingested  # nothing rewritten under old provenance


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rights_note", "A different rights explanation."),
        ("source_description", "A different source description."),
        ("generator", "scripts/generate_seed_catalog.py@2"),
        ("authored_on", "2026-10-01"),
        ("notes", "different notes"),
        ("license_identifier", "TEST-ONLY-LICENCE-ID"),
    ],
)
def test_changed_provenance_fields_require_a_new_dataset_version(
    migrated_engine, ingested, tmp_path, field, value
):
    path, prov = make_variant(tmp_path, seed_lines(), version="1", **{field: value})
    with pytest.raises(IngestRefused, match=field):
        ingest_file(migrated_engine, path, prov)
    assert snapshot(migrated_engine) == ingested


# ---- error-level input persists nothing --------------------------------------------------


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("unknown unit", lambda raw: raw | {"ram_gb": "16 MB"}),
        ("nvme with hdd", lambda raw: raw | {"storage_type": "HDD", "storage_interface": "NVME"}),
        ("non-positive price", lambda raw: raw | {"price": 0}),
        ("cross-category attribute", lambda raw: raw | {"size": 9}),
        ("invalid taxonomy", lambda raw: raw | {"availability": "gone"}),
        ("label field", lambda raw: raw | {"relevance": "relevant"}),
        ("inconsistent synthetic flag", lambda raw: raw | {"is_synthetic": False}),
    ],
)
def test_error_level_input_persists_nothing(migrated_engine, tmp_path, label, mutate):
    lines = seed_lines()
    lines[0] = json.dumps(mutate(json.loads(lines[0])), ensure_ascii=False)
    path, prov = make_variant(tmp_path, lines, version="9")
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.result is None, label
    assert outcome.report.error_count >= 1
    assert counts(migrated_engine) == EMPTY_COUNTS


def test_duplicate_ids_and_identical_content_persist_nothing(migrated_engine, tmp_path):
    lines = seed_lines()
    duplicate_id = edit_line(lines[1], product_id=json.loads(lines[0])["product_id"])
    path, prov = make_variant(
        tmp_path, [*lines[:1], duplicate_id, *lines[2:]], version="9", name="dupid"
    )
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.result is None
    assert "unique_product_id" in {f.check for f in outcome.report.findings}

    clone = edit_line(lines[0], product_id="SYN-LAP-9999")  # identical but for the id
    path, prov = make_variant(tmp_path, [*lines, clone], version="9", name="clone")
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.result is None
    assert "exact_duplicate" in {f.check for f in outcome.report.findings}
    assert counts(migrated_engine)["products"] == 0


def test_same_title_from_another_seller_is_only_a_warning_and_is_persisted(
    migrated_engine, tmp_path
):
    lines = seed_lines()
    other_seller = edit_line(
        lines[0], product_id="SYN-LAP-9999", seller_id="another-seller", price=44444
    )
    path, prov = make_variant(tmp_path, [*lines, other_seller], version="9", name="multiseller")
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.report.error_count == 0
    assert {"same_title_listing"} <= {f.check for f in outcome.report.findings}
    assert outcome.result is not None and outcome.result.inserted == 241


def test_a_warning_only_catalog_is_persisted_and_reported(migrated_engine, tmp_path):
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=5)  # far below the laptop sanity band -> warning only
    path, prov = make_variant(tmp_path, lines, version="3", name="warn")
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.report.error_count == 0 and outcome.report.warning_count >= 1
    assert outcome.result is not None and outcome.result.inserted == 240


# ---- I3: empty catalogs ------------------------------------------------------------------


def test_empty_catalog_is_an_error_and_writes_nothing_even_if_provenance_claims_zero(
    migrated_engine, tmp_path
):
    path, prov = make_variant(tmp_path, [], version="1", name="empty")
    assert json.loads(prov.read_text(encoding="utf-8"))["record_count"] == 0
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.result is None
    assert [f.check for f in outcome.report.findings if f.severity.value == "error"] == [
        "catalog_non_empty"
    ]
    assert counts(migrated_engine) == EMPTY_COUNTS


def test_database_audit_of_an_empty_catalog_is_not_a_success(
    migrated_engine, scratch_database, tmp_path, capsys
):
    with Session(migrated_engine) as s:
        report = audit_database(s)
    statuses = {c["name"]: c["status"] for c in report.checks}
    assert report.error_count == 1 and statuses["catalog_non_empty"] == "fail"
    assert statuses["unique_product_id"] == "not_applicable"  # distinct from the error
    assert main(["check", "--database", scratch_database, "--output-dir", str(tmp_path)]) == 1
    assert "EMPTY" in capsys.readouterr().err


# ---- I8: serialization per dataset family ------------------------------------------------


def test_lock_key_is_deterministic_and_family_specific():
    assert ingest_lock_key("synthetic-seed") == 5132559333090862849  # SHA-256 based, stable
    assert ingest_lock_key("other-dataset") == -2079185854484325726
    assert ingest_lock_key("synthetic-seed") != ingest_lock_key("other-dataset")
    assert -(2**63) <= ingest_lock_key("synthetic-seed") < 2**63


def _wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ingestion_blocks_on_the_dataset_family_lock_until_it_is_released(migrated_engine):
    holder = migrated_engine.connect()
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(ingest_lock_key("synthetic-seed"))))
    results = {}

    def run():
        try:
            results["outcome"] = ingest_file(migrated_engine, SEED, PROVENANCE)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            results["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()

    def waiting_locks():
        with migrated_engine.connect() as probe:
            return probe.execute(
                text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted")
            ).scalar_one()

    try:
        assert _wait_until(lambda: waiting_locks() == 1), "ingest did not wait for the lock"
        assert worker.is_alive() and "outcome" not in results
        assert counts(migrated_engine)["products"] == 0  # nothing written while blocked
    finally:
        transaction.rollback()
        holder.close()
    worker.join(120)
    assert not worker.is_alive() and "error" not in results
    assert results["outcome"].result.inserted == 240


def test_a_different_dataset_family_is_not_blocked_by_the_lock(migrated_engine, tmp_path):
    holder = migrated_engine.connect()
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(ingest_lock_key("some-other-family"))))
    try:
        assert ingest_file(migrated_engine, SEED, PROVENANCE).result.inserted == 240
    finally:
        transaction.rollback()
        holder.close()


def test_concurrent_ingests_of_one_family_cannot_leave_last_writer_wins_corruption(
    migrated_engine, tmp_path
):
    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=777777)
    v2_path, v2_prov = make_variant(tmp_path, lines, version="2", name="v2")
    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}

    def run(name, path, prov):
        barrier.wait()
        try:
            outcomes[name] = ingest_file(migrated_engine, path, prov)
        except IngestRefused as exc:
            outcomes[name] = exc
        except BaseException as exc:  # noqa: BLE001
            outcomes[name] = exc

    threads = [
        threading.Thread(target=run, args=("v1", SEED, PROVENANCE)),
        threading.Thread(target=run, args=("v2", v2_path, v2_prov)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert not any(t.is_alive() for t in threads)
    # v2 always succeeds; v1 either ran first, or was refused as older. Nothing else may happen.
    assert hasattr(outcomes["v2"], "result") and outcomes["v2"].result is not None
    assert hasattr(outcomes["v1"], "result") or isinstance(outcomes["v1"], IngestRefused)
    with Session(migrated_engine) as s:
        latest = s.get(Product, "SYN-LAP-0001")
        assert latest.price == 777777  # the newest version's content wins in either order
        dataset = s.get(CatalogDataset, latest.dataset_pk)
        assert dataset.dataset_version == "2"
        assert s.scalar(select(func.count()).select_from(Product)) == 240
    assert counts(migrated_engine)["specs"] == 240


def test_integrity_races_become_sanitized_domain_errors(migrated_engine, monkeypatch):
    def boom(self, *args, **kwargs):
        raise IntegrityError(
            "INSERT INTO catalog_datasets VALUES (:secret)",
            {"secret": "hunter2"},
            Exception("duplicate key value violates unique constraint; hunter2"),
        )

    monkeypatch.setattr(Session, "flush", boom)
    with pytest.raises(IngestRefused) as excinfo:
        ingest_file(migrated_engine, SEED, PROVENANCE)
    message = str(excinfo.value)
    assert "nothing was written" in message
    for leaked in ("hunter2", "INSERT", "catalog_datasets", "secret"):
        assert leaked not in message
    monkeypatch.undo()
    assert counts(migrated_engine) == EMPTY_COUNTS


# ---- file side vs database side ---------------------------------------------------------


def flawed_lines(raw_record):
    """Valid-but-imperfect records that exercise most warning and info rules."""
    return [
        json.dumps(r)
        for r in (
            raw_record(
                "laptop", product_id="F-1", price=500, title="HP Alpha Laptop, 16GB RAM, 512GB SSD"
            ),
            raw_record(
                "laptop", product_id="F-2", ram_gb=48, title="HP 48GB RAM Laptop, 512GB SSD"
            ),
            raw_record(
                "laptop", product_id="F-3", ram_gb=8, title="HP Beta Laptop, 16GB RAM, 512GB SSD"
            ),
            raw_record("phone", product_id="F-4", rating=4.0, review_count=0, seller_id=None),
            raw_record(
                "phone", product_id="F-5", brand="Zorbo", title="Zorbo (8GB RAM, 128GB Storage) X"
            ),
            raw_record("shoes", product_id="F-6", brand="Sony", title="Sony Shoes", gender=None),
            raw_record(
                "headphones", product_id="F-7", title="Sony Wired Headphones", wireless=True
            ),
            raw_record(
                "laptop",
                product_id="F-8",
                processor=None,
                screen_size_inches=40.0,
                title="HP Gamma Laptop",
            ),
            # same title as F-4 from another seller: a warning, never an error
            raw_record("phone", product_id="F-9", seller_id="other-seller", price=22999),
        )
    ]


def comparable(report):
    return (
        [
            c
            for c in report.checks
            if c["name"] not in ("spec_linkage", "raw_normalized_consistency")
        ],
        [f.to_dict() | {"line": None} for f in report.findings],
        report.statistics,
        report.total_records,
    )


def test_file_side_and_database_side_quality_checks_agree(migrated_engine, tmp_path, raw_record):
    path, prov = make_variant(tmp_path, flawed_lines(raw_record), version="3", name="flawed")
    outcome = ingest_file(migrated_engine, path, prov)
    assert outcome.result is not None and outcome.report.warning_count >= 6
    assert "same_title_listing" in {f.check for f in outcome.report.findings}
    with Session(migrated_engine) as s:
        audit = audit_database(s)
    assert audit.error_count == 0 and audit.source == "database"
    file_checks, file_findings, file_stats, file_total = comparable(outcome.report)
    db_checks, db_findings, db_stats, db_total = comparable(audit)

    def strip(checks):
        return [{k: v for k, v in c.items() if k != "note"} for c in checks]

    assert strip(db_checks) == strip(file_checks)
    assert db_findings == file_findings
    assert (db_stats, db_total) == (file_stats, file_total)
    by_name = {c["name"]: c for c in audit.checks}
    assert by_name["spec_linkage"]["status"] == "pass" and by_name["spec_linkage"]["evaluated"] == 9
    assert by_name["raw_normalized_consistency"]["status"] == "pass"


def test_audit_of_the_ingested_seed_matches_the_file_report(migrated_engine):
    file_report = ingest_file(migrated_engine, SEED, PROVENANCE).report
    with Session(migrated_engine) as s:
        audit = audit_database(s)
    assert audit.error_count == file_report.error_count == 0
    assert comparable(audit)[1:] == comparable(file_report)[1:]


def test_audit_detects_tampering_instead_of_reporting_a_pass(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    with migrated_engine.begin() as conn:
        conn.execute(
            text("UPDATE products SET title = 'HP tampered' WHERE product_id = 'SYN-LAP-0001'")
        )
        conn.execute(text("DELETE FROM phone_specs WHERE product_id = 'SYN-PHN-0001'"))
    with Session(migrated_engine) as s:
        audit = audit_database(s)
    statuses = {c["name"]: c["status"] for c in audit.checks}
    assert statuses["raw_normalized_consistency"] == "fail"
    assert statuses["spec_linkage"] == "fail"
    assert audit.error_count >= 2
