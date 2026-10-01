import csv
import json
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.exc import OperationalError

from ecommerce_search.catalog import cli
from ecommerce_search.catalog.review import CSV_COLUMNS, read_review_csv, write_review_artifacts
from ecommerce_search.ingestion.service import IngestResult

ROOT = Path(__file__).resolve().parents[3]
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"
FILE_ARGS = ["--file", str(SEED), "--provenance", str(PROVENANCE)]
SECRETS = ("hunter2", "10.0.0.5", "INSERT INTO", "Traceback", "postgresql://", "127.0.0.1")


class DummyEngine:
    disposed = False

    def dispose(self):
        self.disposed = True


@pytest.fixture
def engine_calls(monkeypatch):
    """Replaces the engine factory: records requested database names, never connects."""
    calls = []

    def fake_engine(database):
        calls.append(database)
        return DummyEngine()

    monkeypatch.setattr(cli, "_engine", fake_engine)
    return calls


@pytest.fixture
def no_engine(monkeypatch):
    def forbidden(database):  # pragma: no cover - failing is the point
        raise AssertionError("an engine was created before validation finished")

    monkeypatch.setattr(cli, "_engine", forbidden)


# ---- I5: explicit targeting of writes --------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["ingest", *FILE_ARGS],
        ["review-import", "--manifest", "m.json", "--csv", "r.csv", "--reviewer", "TEST-ONLY"],
        ["check"],
    ],
)
def test_write_and_audit_commands_require_an_explicit_database(argv, no_engine, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2  # argparse usage error, before any connection
    assert "--database" in capsys.readouterr().err or "check needs" in capsys.readouterr().err


def test_offline_commands_do_not_need_a_database(no_engine, tmp_path, capsys):
    assert cli.main(["review-sample", *FILE_ARGS, "--output-dir", str(tmp_path)]) == 0
    assert cli.main(["review-status", *FILE_ARGS, "--evidence-dir", str(tmp_path)]) == 0
    assert cli.main(["check", *FILE_ARGS, "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()


def test_ingest_prints_exactly_the_selected_database_and_no_credentials(
    engine_calls, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "persist", lambda *a, **k: IngestResult(inserted=240))
    assert cli.main(["ingest", *FILE_ARGS, "--database", "scratch_db_1"]) == 0
    out = capsys.readouterr()
    assert engine_calls == ["scratch_db_1"]
    assert "target database: scratch_db_1" in out.out.splitlines()
    for secret in ("127.0.0.1", "localhost", "password", "postgresql://", "@"):
        assert secret not in out.out and secret not in out.err


def test_ingest_validates_before_creating_an_engine_or_reading_settings(
    no_engine, tmp_path, capsys
):
    altered = tmp_path / "seed.jsonl"
    altered.write_bytes(SEED.read_bytes() + b"\n")
    assert (
        cli.main(
            [
                "ingest",
                "--database",
                "scratch_x",
                "--file",
                str(altered),
                "--provenance",
                str(PROVENANCE),
            ]
        )
        == 1
    )
    assert "mismatch" in capsys.readouterr().err


def test_ingest_blocked_by_errors_prints_no_target_and_makes_no_engine(no_engine, tmp_path, capsys):
    lines = SEED.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "price": 0})
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    import hashlib

    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prov["checksum_sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    prov_path = tmp_path / "p.json"
    prov_path.write_text(json.dumps(prov), encoding="utf-8")
    assert (
        cli.main(
            [
                "ingest",
                "--database",
                "scratch_x",
                "--file",
                str(bad),
                "--provenance",
                str(prov_path),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "target database" not in captured.out and "BLOCKED" in captured.err


# ---- I6: clean handling of expected failures --------------------------------------------


@pytest.fixture
def review_files(seed_sample, tmp_path):
    csv_path, manifest_path, _ = write_review_artifacts(tmp_path / "review", seed_sample)
    return manifest_path, csv_path


def rewrite_csv(csv_path, **cells):
    rows = read_review_csv(csv_path)
    for row in rows:
        row.update(cells)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def assert_clean_failure(capsys, *needles):
    captured = capsys.readouterr()
    assert (
        captured.err.startswith("error:") or "\nerror:" in captured.err or "error:" in captured.err
    )
    for secret in SECRETS:
        assert secret not in captured.err + captured.out
    for needle in needles:
        assert needle in captured.err, (needle, captured.err)


def test_malformed_and_missing_inputs_fail_cleanly(review_files, tmp_path, engine_calls, capsys):
    manifest, csv_path = review_files
    junk = tmp_path / "junk.json"
    junk.write_text("{not json", encoding="utf-8")
    not_utf8 = tmp_path / "latin1.csv"
    not_utf8.write_bytes(b"product_id,verdict\r\nSYN-LAP-0001,\x80\r\n")
    no_columns = tmp_path / "nocols.csv"
    no_columns.write_text("a,b\n1,2\n", encoding="utf-8")
    empty_dir_file = tmp_path / "a_file"
    empty_dir_file.write_text("x", encoding="utf-8")
    common = ["--database", "scratch_x", "--reviewer", "TEST-ONLY", "--confirm-human-review"]

    cases = [
        (
            [
                "ingest",
                "--database",
                "x",
                "--file",
                str(tmp_path / "missing.jsonl"),
                "--provenance",
                str(PROVENANCE),
            ],
            "cannot read",
        ),
        (
            ["ingest", "--database", "x", "--file", str(SEED), "--provenance", str(junk)],
            "provenance",
        ),
        (
            [
                "ingest",
                "--database",
                "x",
                "--file",
                str(SEED),
                "--provenance",
                str(tmp_path / "nope.json"),
            ],
            "provenance",
        ),
        (["check", "--file", str(not_utf8), "--provenance", str(PROVENANCE)], "UTF-8"),
        (
            ["review-import", "--manifest", str(junk), "--csv", str(csv_path), *common],
            "not valid JSON",
        ),
        (
            [
                "review-import",
                "--manifest",
                str(tmp_path / "gone.json"),
                "--csv",
                str(csv_path),
                *common,
            ],
            "cannot access",
        ),
        (
            [
                "review-import",
                "--manifest",
                str(manifest),
                "--csv",
                str(tmp_path / "gone.csv"),
                *common,
            ],
            "cannot access",
        ),
        (
            ["review-import", "--manifest", str(manifest), "--csv", str(not_utf8), *common],
            "CSV UTF-8",
        ),
        (
            ["review-import", "--manifest", str(manifest), "--csv", str(no_columns), *common],
            "missing required column",
        ),
        (["review-status", *FILE_ARGS, "--evidence", str(junk)], None),
        (["check", *FILE_ARGS, "--output-dir", str(empty_dir_file)], "cannot access"),
    ]
    for argv, needle in cases:
        code = cli.main(argv)
        assert code == 1, argv
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err + captured.out, argv
        assert captured.err.startswith("error:") or (needle is None and "ERROR" in captured.out), (
            argv
        )
        if needle:
            assert needle.lower() in captured.err.lower(), (argv, captured.err)
    assert engine_calls == []  # none of these reached a database


def test_import_refuses_blank_and_unconfirmed_reviews_before_any_engine(
    review_files, no_engine, capsys
):
    manifest, csv_path = review_files
    base = [
        "review-import",
        "--manifest",
        str(manifest),
        "--csv",
        str(csv_path),
        "--database",
        "scratch_x",
        "--reviewer",
        "TEST-ONLY",
    ]
    assert cli.main(base) == 1
    assert_clean_failure(capsys, "confirm")
    assert cli.main([*base, "--confirm-human-review"]) == 1
    assert_clean_failure(capsys, "verdict is blank")
    rewrite_csv(csv_path, verdict="accept")
    assert cli.main([*base, "--confirm-human-review", "--reviewer", "a@b.co"]) == 1
    assert_clean_failure(capsys, "handle")


def test_import_refuses_when_a_non_review_field_was_edited(review_files, no_engine, capsys):
    manifest, csv_path = review_files
    rows = read_review_csv(csv_path)
    rows[0]["title"] = "EDITED"
    for row in rows:
        row["verdict"] = "accept"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    argv = [
        "review-import",
        "--manifest",
        str(manifest),
        "--csv",
        str(csv_path),
        "--database",
        "scratch_x",
        "--reviewer",
        "TEST-ONLY",
        "--confirm-human-review",
    ]
    assert cli.main(argv) == 1
    assert_clean_failure(capsys, "non-review column")


@pytest.mark.parametrize(
    "error",
    [
        OperationalError(
            "INSERT INTO secret_table VALUES (:p)",
            {"p": "hunter2"},
            Exception("connection to 10.0.0.5 failed password=hunter2"),
        ),
        psycopg.OperationalError("host=10.0.0.5 password=hunter2 refused"),
    ],
)
def test_database_failures_are_sanitized(error, engine_calls, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "persist", boom)
    assert cli.main(["ingest", *FILE_ARGS, "--database", "scratch_x"]) == 1
    captured = capsys.readouterr()
    assert "database error" in captured.err
    for secret in SECRETS + ("secret_table",):
        assert secret not in captured.err + captured.out
    assert engine_calls == ["scratch_x"]


def test_database_audit_failure_is_sanitized(monkeypatch, engine_calls, tmp_path, capsys):
    def boom(session):
        raise OperationalError("SELECT 1", {}, Exception("password=hunter2"))

    monkeypatch.setattr(cli, "audit_database", boom)
    monkeypatch.setattr(cli, "Session", lambda engine: _NullSession())
    assert cli.main(["check", "--database", "x", "--output-dir", str(tmp_path)]) == 1
    assert "hunter2" not in capsys.readouterr().err


class _NullSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_unexpected_programming_errors_are_not_swallowed(engine_calls, monkeypatch):
    def bug(*args, **kwargs):
        raise RuntimeError("a genuine bug")

    monkeypatch.setattr(cli, "persist", bug)
    with pytest.raises(RuntimeError, match="genuine bug"):
        cli.main(["ingest", *FILE_ARGS, "--database", "scratch_x"])


# ---- B3 through the CLI -----------------------------------------------------------------


def test_review_sample_cli_never_overwrites_review_work(tmp_path, capsys):
    argv = ["review-sample", *FILE_ARGS, "--output-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "PENDING" in out and "36 products" in out
    (batch_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert batch_dir.name.startswith("synthetic-seed-v1-")
    csv_path = batch_dir / "review_sample.csv"
    rows = read_review_csv(csv_path)
    assert len(rows) == 36 and all(r["verdict"] == "" for r in rows)

    before = {p.name: p.read_bytes() for p in batch_dir.iterdir()}
    assert cli.main(argv) == 1  # second run refuses
    assert "refusing to overwrite" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in batch_dir.iterdir()} == before

    assert cli.main([*argv, "--force"]) == 0  # provably blank: may be regenerated
    capsys.readouterr()

    rewrite_csv(csv_path, notes="reviewer input")  # now it contains review work
    before = {p.name: p.read_bytes() for p in batch_dir.iterdir()}
    for extra in ([], ["--force"]):
        assert cli.main([*argv, *extra]) == 1
        assert "reviewer input" not in capsys.readouterr().out
        assert {p.name: p.read_bytes() for p in batch_dir.iterdir()} == before


# ---- D. the CLI refuses forged manifests before any database or evidence write ------------


def test_cli_import_accepts_the_genuine_manifest_but_refuses_a_forged_one_before_any_write(
    seed_sample, tmp_path, no_engine, capsys
):
    from test_catalog_review_workflow import seal  # same package directory

    csv_path, manifest_path, _ = write_review_artifacts(tmp_path / "ok", seed_sample)
    rewrite_csv(csv_path, verdict="accept")
    common = [
        "--csv",
        str(csv_path),
        "--database",
        "scratch_x",
        "--reviewer",
        "TEST-ONLY",
        "--confirm-human-review",
        "--evidence-dir",
        str(tmp_path / "labels"),
        *FILE_ARGS,
    ]

    forged = seal(
        {
            **seed_sample.manifest,
            "quotas": {**seed_sample.manifest["quotas"], "laptop": 11, "phone": 9},
        }
    )
    forged_path = tmp_path / "forged_manifest.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    assert cli.main(["review-import", "--manifest", str(forged_path), *common]) == 1
    captured = capsys.readouterr()
    assert (
        "not the review sample generated" in captured.err
        and "quotas" in captured.err.split("differs in:")[1]
    )
    assert "target database" not in captured.out  # refused before anything was attempted
    assert not (tmp_path / "labels").exists()

    # the genuine manifest passes authenticity and only then hits the (blocked) engine step
    with pytest.raises(AssertionError, match="before validation finished"):
        cli.main(["review-import", "--manifest", str(manifest_path), *common])
