"""Human-review recording, binding and status against real PostgreSQL (scratch databases only).

Every reviewer name and verdict here is explicit TEST data created by the tests; no real
human review is simulated, recorded or written to data/labels.
"""

import csv
import json
from datetime import UTC, datetime

import pytest
from catalog_support import (
    PROVENANCE,
    ROOT,
    SEED,
    TEST_REVIEWER,
    counts,
    edit_line,
    make_variant,
    seed_lines,
)
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ecommerce_search.catalog import cli
from ecommerce_search.catalog.cli import main
from ecommerce_search.catalog.evidence import EvidenceError
from ecommerce_search.catalog.review import (
    CSV_COLUMNS,
    ReviewError,
    database_review_status,
    prepare_sample,
    read_review_csv,
    record_reviews,
    validate_import,
)
from ecommerce_search.config import get_settings
from ecommerce_search.ingestion.service import dataset_info, ingest_file, prepare_ingest
from ecommerce_search.models.catalog import CatalogDataset, CatalogReview, Product

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 1, 9, 0, 0, tzinfo=UTC)


def build_sample():
    prepared = prepare_ingest(SEED, PROVENANCE)
    return prepare_sample(
        prepared.catalog,
        prepared.provenance,
        prepared.report,
        dataset_info(prepared.provenance, prepared.catalog.checksum_sha256),
    )


def completed(sample, verdict="accept"):
    rows = [{**r, "verdict": verdict} for r in sample.rows]
    return validate_import(
        manifest=sample.manifest, rows=rows, reviewer=TEST_REVIEWER, confirmed=True
    )


def record(engine, sample, reviews=None, reviewer=TEST_REVIEWER):
    with Session(engine) as s, s.begin():
        return record_reviews(
            s,
            manifest=sample.manifest,
            reviews=reviews if reviews is not None else completed(sample),
            reviewer=reviewer,
            recorded_at=NOW,
        )


def status(engine, sample):
    with Session(engine) as s:
        return database_review_status(s, sample.manifest)


def insert_review_sql(engine, sample, product_id, *, reviewer, content_hash, manifest_hash=None):
    with engine.begin() as conn:
        dataset_pk = conn.execute(text("SELECT id FROM catalog_datasets")).scalar_one()
        conn.execute(
            text(
                "INSERT INTO catalog_reviews (review_batch_id, product_id, dataset_pk, "
                "product_content_sha256, verdict, reviewer, sample_manifest_sha256) "
                "VALUES (:b, :p, :d, :h, 'accept', :r, :m)"
            ),
            {
                "b": sample.manifest["review_batch_id"],
                "p": product_id,
                "d": dataset_pk,
                "h": content_hash,
                "r": reviewer,
                "m": manifest_hash or sample.manifest["manifest_sha256"],
            },
        )


# ---- recording binds the review to exact content (B2) ------------------------------------


def test_recording_stores_hashes_and_never_touches_product_content(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    assert status(migrated_engine, sample).state == "PENDING"
    with Session(migrated_engine) as s:
        before = {
            p.product_id: (p.title, p.description, p.content_sha256, p.updated_at)
            for p in s.scalars(select(Product))
        }

    assert record(migrated_engine, sample) == 36

    with Session(migrated_engine) as s:
        rows = s.scalars(select(CatalogReview).order_by(CatalogReview.id)).all()
        live = {p.product_id: p.content_sha256 for p in s.scalars(select(Product))}
        dataset = s.scalars(select(CatalogDataset)).one()
        assert len(rows) == 36
        expected = {p["product_id"]: p["content_sha256"] for p in sample.manifest["products"]}
        for row in rows:
            assert row.product_content_sha256 == expected[row.product_id] == live[row.product_id]
            assert row.sample_manifest_sha256 == sample.manifest["manifest_sha256"]
            assert row.review_batch_id == sample.manifest["review_batch_id"]
            assert row.dataset_pk == dataset.id and row.reviewer == TEST_REVIEWER
            assert row.recorded_at == NOW
        after = {
            p.product_id: (p.title, p.description, p.content_sha256, p.updated_at)
            for p in s.scalars(select(Product))
        }
    assert after == before  # review outcomes never leak into product text or content
    result = status(migrated_engine, sample)
    assert (result.state, result.reviewed, result.total) == ("RECORDED", 36, 36)


def test_a_product_changed_after_sampling_cannot_be_reviewed(migrated_engine, tmp_path):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    sampled = sample.manifest["product_ids"][0]
    index = next(i for i, line in enumerate(seed_lines()) if f'"{sampled}"' in line)
    lines = seed_lines()
    lines[index] = edit_line(lines[index], price=987654)
    path, prov = make_variant(tmp_path, lines, version="2")
    assert ingest_file(migrated_engine, path, prov).result.updated == 1

    with pytest.raises(ReviewError, match="changed after the sample was made"):
        record(migrated_engine, sample)
    assert counts(migrated_engine)["reviews"] == 0


def test_review_cannot_attach_to_a_different_dataset_or_missing_data(migrated_engine, tmp_path):
    sample = build_sample()
    with pytest.raises(ReviewError, match="not in the database"):  # nothing ingested
        record(migrated_engine, sample)

    lines = seed_lines()
    lines[0] = edit_line(lines[0], price=11111)
    path, prov = make_variant(tmp_path, lines, version="1", name="other_v1")
    ingest_file(migrated_engine, path, prov)  # dataset "synthetic-seed" v1 with other bytes
    with pytest.raises(ReviewError, match="not in the database"):
        record(migrated_engine, sample)
    assert counts(migrated_engine)["reviews"] == 0


def test_recording_twice_is_a_clean_review_error(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    record(migrated_engine, sample)
    with pytest.raises(ReviewError, match="could not be recorded"):
        record(migrated_engine, sample)
    assert counts(migrated_engine)["reviews"] == 36


# ---- status is manifest-exact (I7) -------------------------------------------------------


def test_partial_batches_stay_pending_and_foreign_or_mixed_rows_are_errors(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    reviews = completed(sample)
    record(migrated_engine, sample, reviews[:10])
    partial = status(migrated_engine, sample)
    assert (partial.state, partial.reviewed, partial.total) == ("PENDING", 10, 36)

    # a second reviewer's row for another sampled product: one consistent reviewer is required
    insert_review_sql(
        migrated_engine,
        sample,
        reviews[10].product_id,
        reviewer="SOMEONE-ELSE",
        content_hash=reviews[10].product_content_sha256,
    )
    assert status(migrated_engine, sample).state == "ERROR"


def test_a_product_outside_the_sample_makes_the_batch_an_error_even_when_36_are_present(
    migrated_engine,
):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    record(migrated_engine, sample)
    assert status(migrated_engine, sample).state == "RECORDED"
    outsider = next(
        pid
        for pid in (f"SYN-LAP-{n:04d}" for n in range(1, 81))
        if pid not in sample.manifest["product_ids"]
    )
    with migrated_engine.connect() as conn:
        content = conn.execute(
            text("SELECT content_sha256 FROM products WHERE product_id = :p"), {"p": outsider}
        ).scalar_one()
    insert_review_sql(
        migrated_engine, sample, outsider, reviewer=TEST_REVIEWER, content_hash=content
    )
    result = status(migrated_engine, sample)
    assert result.state == "ERROR" and any("outside the sample" in p for p in result.problems)


def test_wrong_hashes_never_complete_a_batch(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    sample = build_sample()
    reviews = completed(sample)
    record(migrated_engine, sample, reviews[:35])
    insert_review_sql(
        migrated_engine,
        sample,
        reviews[35].product_id,
        reviewer=TEST_REVIEWER,
        content_hash="0" * 64,
    )
    result = status(migrated_engine, sample)
    assert result.state == "ERROR" and any("product_content_sha256" in p for p in result.problems)

    other = build_sample()
    other.manifest["manifest_sha256"] = "1" * 64  # rows were recorded under a different manifest
    mismatch = status(
        migrated_engine, other
    )  # same batch id, but recorded under another manifest hash
    assert mismatch.state == "ERROR" and any("manifest_sha256" in p for p in mismatch.problems)


# ---- CLI end to end on a scratch database -----------------------------------------------


def fill_all(csv_path, verdict="accept"):
    rows = read_review_csv(csv_path)
    for row in rows:
        row["verdict"] = verdict
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_end_to_end_with_explicit_database_and_offline_evidence(
    migrated_engine, scratch_database, tmp_path, capsys, monkeypatch
):
    labels_before = sorted((ROOT / "data").rglob("catalog_review_*.json"))
    base = ["--file", str(SEED), "--provenance", str(PROVENANCE)]
    assert main(["ingest", *base, "--database", scratch_database]) == 0
    out = capsys.readouterr().out
    assert f"target database: {scratch_database}" in out
    assert "inserted=240 updated=0 unchanged=0" in out
    assert main(["ingest", *base, "--database", scratch_database]) == 0
    assert "inserted=0 updated=0 unchanged=240" in capsys.readouterr().out

    quality = tmp_path / "quality"
    assert main(["check", "--database", scratch_database, "--output-dir", str(quality)]) == 0
    assert "errors=0" in capsys.readouterr().out

    review_root = tmp_path / "review"
    labels = tmp_path / "labels"  # test-only evidence directory, never data/labels
    assert main(["review-sample", *base, "--output-dir", str(review_root)]) == 0
    capsys.readouterr()
    (batch_dir,) = review_root.iterdir()
    manifest, csv_path = batch_dir / "review_manifest.json", batch_dir / "review_sample.csv"

    status_args = ["review-status", *base, "--evidence-dir", str(labels)]
    assert main(status_args) == 0
    assert "PENDING (0/36" in capsys.readouterr().out

    import_args = [
        "review-import",
        "--manifest",
        str(manifest),
        "--csv",
        str(csv_path),
        "--database",
        scratch_database,
        "--evidence-dir",
        str(labels),
        *base,
    ]
    assert main([*import_args, "--reviewer", TEST_REVIEWER]) == 1  # no confirmation flag
    assert main([*import_args, "--reviewer", TEST_REVIEWER, "--confirm-human-review"]) == 1  # blank
    capsys.readouterr()
    assert counts(migrated_engine)["reviews"] == 0 and not labels.exists()

    fill_all(csv_path)  # TEST-ONLY verdicts written by this test
    assert main([*import_args, "--reviewer", TEST_REVIEWER, "--confirm-human-review"]) == 0
    out = capsys.readouterr().out
    assert f"target database: {scratch_database}" in out and "recorded 36 review rows" in out
    (evidence_file,) = labels.glob("catalog_review_synthetic-seed_v1_*.json")
    assert counts(migrated_engine)["reviews"] == 36

    assert main([*status_args, "--require-complete", "--database", scratch_database]) == 0
    out = capsys.readouterr().out
    assert "(evidence): RECORDED (36/36" in out and "(database): RECORDED (36/36" in out
    assert main(["review-verify", *base, "--evidence-dir", str(labels)]) == 0
    capsys.readouterr()

    # evidence is never overwritten and a second import cannot double-record
    before = evidence_file.read_bytes()
    assert main([*import_args, "--reviewer", TEST_REVIEWER, "--confirm-human-review"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert evidence_file.read_bytes() == before and counts(migrated_engine)["reviews"] == 36

    # N1: historical evidence stays RECORDED after the CURRENT sampling rules change, and the
    # database cross-check uses the embedded (reviewed) manifest, not today's code
    from ecommerce_search.catalog import review as review_module
    from ecommerce_search.catalog.quality import parameters as qp

    monkeypatch.setattr(qp, "RULES_VERSION", "99")
    monkeypatch.setattr(review_module, "SELECTION_METHOD", "reworded after the review")
    assert main([*status_args, "--require-complete", "--database", scratch_database]) == 0
    out = capsys.readouterr().out
    assert "(evidence): RECORDED (36/36" in out and "(database): RECORDED (36/36" in out
    assert "matches_current_sampling_rules: false" in out
    monkeypatch.undo()

    # the repository's own data/ tree gained no evidence file
    assert sorted((ROOT / "data").rglob("catalog_review_*.json")) == labels_before


def test_evidence_write_failure_rolls_the_database_back(
    migrated_engine, scratch_database, tmp_path, monkeypatch, capsys
):
    base = ["--file", str(SEED), "--provenance", str(PROVENANCE)]
    assert main(["ingest", *base, "--database", scratch_database]) == 0
    assert main(["review-sample", *base, "--output-dir", str(tmp_path / "r")]) == 0
    (batch_dir,) = (tmp_path / "r").iterdir()
    fill_all(batch_dir / "review_sample.csv")
    capsys.readouterr()

    def failing_write(*args, **kwargs):
        raise EvidenceError("simulated evidence write failure")

    monkeypatch.setattr(cli, "write_evidence", failing_write)
    labels = tmp_path / "labels"
    code = main(
        [
            "review-import",
            "--manifest",
            str(batch_dir / "review_manifest.json"),
            "--csv",
            str(batch_dir / "review_sample.csv"),
            "--database",
            scratch_database,
            "--reviewer",
            TEST_REVIEWER,
            "--confirm-human-review",
            "--evidence-dir",
            str(labels),
        ]
    )
    assert code == 1
    assert counts(migrated_engine)["reviews"] == 0  # no DB rows without durable evidence
    assert not labels.exists() or not list(labels.glob("*.json"))


# ---- sanitized database failures against a real server -----------------------------------


@pytest.fixture
def fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_database_is_reported_without_connection_details(fresh_settings, tmp_path, capsys):
    code = main(
        ["check", "--database", "no_such_database_for_m2_tests", "--output-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert code == 1 and "database error (OperationalError)" in captured.err
    settings = get_settings()
    for leaked in (
        settings.postgres_host,
        settings.postgres_user,
        settings.postgres_password.get_secret_value(),
        "postgresql",
        "Traceback",
    ):
        assert leaked not in captured.err + captured.out


def test_unreachable_server_is_reported_without_connection_details(
    fresh_settings, monkeypatch, tmp_path, capsys
):
    password = get_settings().postgres_password.get_secret_value()
    monkeypatch.setenv("POSTGRES_PORT", "1")
    get_settings.cache_clear()
    code = main(["check", "--database", "anything", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1 and "database error" in captured.err
    for leaked in (password, "127.0.0.1", "Traceback", "port 1"):
        assert leaked not in captured.err + captured.out


def test_a_forged_manifest_is_refused_before_any_database_or_evidence_write(
    migrated_engine, scratch_database, tmp_path, capsys
):
    from ecommerce_search.catalog.manifest import batch_id, manifest_digest

    base = ["--file", str(SEED), "--provenance", str(PROVENANCE)]
    assert main(["ingest", *base, "--database", scratch_database]) == 0
    assert main(["review-sample", *base, "--output-dir", str(tmp_path / "r")]) == 0
    (batch_dir,) = (tmp_path / "r").iterdir()
    csv_path = batch_dir / "review_sample.csv"
    fill_all(csv_path)
    capsys.readouterr()

    genuine = json.loads((batch_dir / "review_manifest.json").read_text(encoding="utf-8"))
    forged = {**genuine, "quotas": {**genuine["quotas"], "laptop": 11, "phone": 9}}
    forged["manifest_sha256"] = manifest_digest(forged)  # self-consistent on purpose
    forged["review_batch_id"] = batch_id(
        forged["dataset_id"], forged["dataset_version"], forged["manifest_sha256"]
    )
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    labels = tmp_path / "labels"
    code = main(
        [
            "review-import",
            "--manifest",
            str(forged_path),
            "--csv",
            str(csv_path),
            "--database",
            scratch_database,
            "--reviewer",
            TEST_REVIEWER,
            "--confirm-human-review",
            "--evidence-dir",
            str(labels),
            *base,
        ]
    )
    captured = capsys.readouterr()
    assert code == 1 and "not the review sample generated" in captured.err
    assert counts(migrated_engine)["reviews"] == 0  # no catalog_reviews row
    assert not labels.exists()  # no evidence file
