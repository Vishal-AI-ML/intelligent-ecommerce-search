"""Catalog command line.

Commands: ingest, check, review-sample, review-import, review-status, review-verify.
Uses only the standard library `argparse`.

* Commands that WRITE database state (`ingest`, `review-import`) require an explicit
  `--database NAME`; there is no silent default to POSTGRES_DB.
* `check` needs `--file` (offline) or `--database` (database audit).
* `review-sample`, `review-status` and `review-verify` are offline (no database).
* Expected operational failures (bad input files, database errors) become one concise line and
  a non-zero exit code; they never show tracebacks, URLs, credentials or SQL parameters.
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.catalog.audit import audit_database
from ecommerce_search.catalog.evidence import (
    EVIDENCE_DIR,
    EvidenceError,
    ReviewStatus,
    SeedFacts,
    build_evidence,
    evidence_path,
    find_evidence,
    load_evidence,
    verify_evidence,
    write_evidence,
)
from ecommerce_search.catalog.quality.report import QualityReport, write_report
from ecommerce_search.catalog.review import (
    SAMPLE_QUOTAS,
    ReviewError,
    ReviewSample,
    database_review_status,
    prepare_sample,
    read_review_csv,
    record_reviews,
    require_genuine_manifest,
    sample_directory,
    validate_import,
    verify_manifest,
    write_review_artifacts,
)
from ecommerce_search.config import get_settings
from ecommerce_search.db.engine import create_db_engine
from ecommerce_search.ingestion.loader import (
    LoaderError,
    load_catalog,
    load_provenance,
    verify_against_provenance,
)
from ecommerce_search.ingestion.service import (
    IngestRefused,
    build_file_report,
    dataset_info,
    persist,
    prepare_ingest,
)

DEFAULT_CATALOG = Path("data/seed/catalog_seed_v1.jsonl")
DEFAULT_PROVENANCE = Path("data/seed/catalog_seed_v1.provenance.json")
DEFAULT_QUALITY_DIR = Path("data/processed/catalog_quality")
DEFAULT_REVIEW_DIR = Path("data/processed/catalog_review")


class CliError(Exception):
    """A user-facing operational error with an already-sanitized message."""


def _engine(database: str) -> Engine:
    try:
        settings = get_settings()
    except ValidationError:
        raise CliError(
            "database settings are incomplete or invalid (is POSTGRES_PASSWORD set?)"
        ) from None
    return create_db_engine(settings.model_copy(update={"postgres_db": database}))


def _print_summary(report: QualityReport) -> None:
    print(
        f"records={report.total_records} errors={report.error_count} "
        f"warnings={report.warning_count} info={report.info_count}"
    )


def _print_errors(report: QualityReport, limit: int = 25) -> None:
    errors = [f for f in report.findings if f.severity.value == "error"]
    for f in errors[:limit]:
        where = f.product_id or (f"line {f.line}" if f.line else "catalog")
        print(f"  error [{f.check}] {where}: {f.message}", file=sys.stderr)
    if len(errors) > limit:
        print(f"  ... and {len(errors) - limit} more", file=sys.stderr)


def _load_file(args: argparse.Namespace):
    provenance = load_provenance(args.provenance)
    catalog = load_catalog(args.file)
    verify_against_provenance(catalog, provenance)
    return catalog, provenance


def _expected_sample(args: argparse.Namespace) -> ReviewSample:
    """Regenerate the deterministic sample from the committed seed (offline)."""
    catalog, provenance = _load_file(args)
    report = build_file_report(catalog, provenance)
    if report.error_count:
        _print_errors(report)
        raise CliError("the catalog has error-level findings; cannot derive the review sample")
    return prepare_sample(
        catalog, provenance, report, dataset_info(provenance, catalog.checksum_sha256)
    )


# ---- commands -------------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    prepared = prepare_ingest(args.file, args.provenance)  # no database / settings needed
    if args.output_dir:
        write_report(prepared.report, args.output_dir, "catalog_quality_file")
    _print_summary(prepared.report)
    if prepared.blocked:
        print("ingestion BLOCKED by error-level findings; nothing was written", file=sys.stderr)
        _print_errors(prepared.report)
        return 1
    print(f"target database: {args.database}")
    engine = _engine(args.database)
    try:
        result = persist(engine, prepared.catalog, prepared.provenance)
    finally:
        engine.dispose()
    print(f"ingest: {result.summary()}")
    if result.absent_product_ids:
        print(
            "products in the database but absent from this file (not deleted): "
            + ", ".join(result.absent_product_ids)
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if args.file is not None:
        catalog, provenance = _load_file(args)
        report = build_file_report(catalog, provenance)
        stem = "catalog_quality_file"
    else:
        engine = _engine(args.database)
        try:
            with Session(engine) as session:
                report = audit_database(session)
        finally:
            engine.dispose()
        stem = "catalog_quality_database"
    json_path, md_path = write_report(report, args.output_dir, stem)
    _print_summary(report)
    print(f"reports: {json_path} , {md_path}")
    if report.error_count:
        _print_errors(report)
        return 1
    return 0


def cmd_review_sample(args: argparse.Namespace) -> int:
    sample = _expected_sample(args)
    directory = sample_directory(args.output_dir, sample.manifest)
    csv_path, manifest_path, _ = write_review_artifacts(directory, sample, force=args.force)
    counts: dict[str, int] = {}
    for row in sample.rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"sample: {len(sample.rows)} products {counts}")
    print(f"batch: {sample.manifest['review_batch_id']}")
    print(f"review artifacts: {csv_path} , {manifest_path}")
    print("review status: PENDING (no human verdicts have been entered or recorded)")
    return 0


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewError(f"{path.name} is not valid JSON ({exc.msg})") from None
    except UnicodeDecodeError:
        raise ReviewError(f"{path.name} is not valid UTF-8") from None
    if not isinstance(manifest, dict):
        raise ReviewError(f"{path.name} is not a manifest object")
    verify_manifest(manifest)
    return manifest


def cmd_review_import(args: argparse.Namespace) -> int:
    manifest = _read_manifest(args.manifest)
    # Authenticity: the manifest must be exactly the sample the committed seed and the current
    # sampling code generate; a hand-crafted (even self-consistent) manifest is refused here,
    # before any database or evidence write.
    require_genuine_manifest(manifest, _expected_sample(args).manifest)
    rows = read_review_csv(args.csv)
    reviews = validate_import(
        manifest=manifest, rows=rows, reviewer=args.reviewer, confirmed=args.confirm_human_review
    )
    reviewer = args.reviewer.strip()
    recorded_at = datetime.now(UTC)
    evidence = build_evidence(manifest, reviews, reviewer_handle=reviewer, recorded_at=recorded_at)
    target = evidence_path(args.evidence_dir, manifest)
    if target.exists():
        raise EvidenceError(f"evidence file {target.name} already exists; it is never overwritten")

    print(f"target database: {args.database}")
    engine = _engine(args.database)
    written: Path | None = None
    try:
        with Session(engine) as session, session.begin():
            n = record_reviews(
                session,
                manifest=manifest,
                reviews=reviews,
                reviewer=reviewer,
                recorded_at=recorded_at,
            )
            written = write_evidence(args.evidence_dir, evidence, manifest)
    except BaseException:
        if written is not None:  # the transaction did not commit: do not leave orphan evidence
            written.unlink(missing_ok=True)
        raise
    finally:
        engine.dispose()
    print(f"recorded {n} review rows for batch {manifest['review_batch_id']} by {reviewer!r}")
    print(f"evidence file: {written}")
    print("commit this evidence file to keep a durable audit trail (not proof of identity)")
    return 0


def _seed_facts(args: argparse.Namespace):
    """Facts from the committed seed (offline). Historical verification uses only these."""
    catalog, provenance = _load_file(args)
    facts = SeedFacts(
        dataset_id=provenance.dataset_id,
        dataset_version=provenance.dataset_version,
        dataset_checksum_sha256=catalog.checksum_sha256,
        content_hashes={line.record.product_id: line.content_sha256 for line in catalog.lines},
    )
    return facts, catalog, provenance


def _current_manifest(catalog, provenance) -> dict | None:
    """Today's sampling-rule output, for the informational compatibility flag only."""
    report = build_file_report(catalog, provenance)
    if report.error_count:
        return None
    try:
        sample = prepare_sample(
            catalog, provenance, report, dataset_info(provenance, catalog.checksum_sha256)
        )
    except ReviewError:
        return None
    return sample.manifest


def _evidence_candidates(args: argparse.Namespace, facts: SeedFacts) -> list[Path]:
    if args.evidence:
        return [args.evidence]
    return find_evidence(args.evidence_dir, facts.dataset_id, facts.dataset_version)


def _compat_line(matches: bool | None) -> str:
    shown = "unknown" if matches is None else str(matches).lower()
    return (
        f"matches_current_sampling_rules: {shown} "
        "(informational; never changes the historical status)"
    )


def cmd_review_status(args: argparse.Namespace) -> int:
    facts, catalog, provenance = _seed_facts(args)
    current = _current_manifest(catalog, provenance)
    total = len(current["product_ids"]) if current else sum(SAMPLE_QUOTAS.values())
    candidates = _evidence_candidates(args, facts)
    cross_check = current
    note = None
    if not candidates or not candidates[0].exists():
        status = ReviewStatus("PENDING", 0, total, problems=("no evidence file found",))
    elif len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        status = ReviewStatus(
            "ERROR", 0, total, problems=(f"more than one evidence file ({names}); use --evidence",)
        )
    else:
        try:
            evidence = load_evidence(candidates[0])
            result = verify_evidence(evidence, facts, current)
            status = ReviewStatus(
                "RECORDED",
                len(result.manifest["product_ids"]),
                len(result.manifest["product_ids"]),
                (evidence.reviewer_handle,),
            )
            cross_check = dict(result.manifest)  # the reviewed (historical) sample
            note = _compat_line(result.matches_current_sampling_rules)
        except EvidenceError as exc:
            status = ReviewStatus("ERROR", 0, total, problems=(str(exc),))
    print(status.describe())
    if note:
        print(note)
    if args.database:  # optional read-only cross-check of the local append-only table
        if cross_check is None:
            print("database cross-check skipped: no review sample could be derived")
        else:
            engine = _engine(args.database)
            try:
                with Session(engine) as session:
                    print(database_review_status(session, cross_check).describe())
            finally:
                engine.dispose()
    if status.state == "ERROR":
        return 1
    return 1 if (args.require_complete and status.state != "RECORDED") else 0


def cmd_review_verify(args: argparse.Namespace) -> int:
    facts, catalog, provenance = _seed_facts(args)
    candidates = _evidence_candidates(args, facts)
    if not candidates:
        raise EvidenceError("no evidence file found")
    if len(candidates) > 1:
        raise EvidenceError("more than one evidence file found; use --evidence")
    evidence = load_evidence(candidates[0])  # raises EvidenceError if absent or malformed
    result = verify_evidence(evidence, facts, _current_manifest(catalog, provenance))
    print(
        f"OK: {evidence.product_count} reviews by {evidence.reviewer_handle!r} verified offline "
        f"against the committed seed and the embedded reviewed manifest (batch "
        f"{evidence.review_batch_id}, records_sha256 {evidence.records_sha256[:16]}...)"
    )
    print(_compat_line(result.matches_current_sampling_rules))
    return 0


# ---- parser ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ecommerce_search.catalog", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p: argparse.ArgumentParser, *, default_file: bool) -> None:
        p.add_argument(
            "--file",
            type=Path,
            default=DEFAULT_CATALOG if default_file else None,
            help="catalog JSONL file",
        )
        p.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)

    def add_db(p: argparse.ArgumentParser, *, required: bool, why: str) -> None:
        p.add_argument(
            "--database",
            required=required,
            help=f"database name on the configured server ({why}); no default",
        )

    def add_evidence(p: argparse.ArgumentParser) -> None:
        p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
        p.add_argument(
            "--evidence", type=Path, help="explicit evidence file (default: found by dataset)"
        )

    p = sub.add_parser("ingest", help="validate, then persist a catalog file atomically")
    add_source(p, default_file=True)
    add_db(p, required=True, why="REQUIRED: the database that will be written")
    p.add_argument("--output-dir", type=Path, help="also write the file-side quality report here")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("check", help="quality report for a file (--file) or a database audit")
    add_source(p, default_file=False)
    add_db(p, required=False, why="required for a database audit when --file is not given")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("review-sample", help="write the deterministic, blank human-review sample")
    add_source(p, default_file=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    p.add_argument(
        "--force",
        action="store_true",
        help="replace an existing sample, but only if its CSV is provably blank",
    )
    p.set_defaults(func=cmd_review_sample)

    p = sub.add_parser("review-import", help="record a COMPLETED human review (human use only)")
    add_source(p, default_file=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    add_db(p, required=True, why="REQUIRED: the database that will be written")
    p.add_argument("--reviewer", required=True, help="the reviewer's handle (never an email)")
    p.add_argument("--confirm-human-review", action="store_true")
    p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    p.set_defaults(func=cmd_review_import)

    p = sub.add_parser("review-status", help="offline: PENDING / RECORDED / ERROR from evidence")
    add_source(p, default_file=True)
    add_evidence(p)
    add_db(p, required=False, why="optional read-only cross-check of the database")
    p.add_argument("--require-complete", action="store_true")
    p.set_defaults(func=cmd_review_status)

    p = sub.add_parser("review-verify", help="offline: verify the evidence file against the seed")
    add_source(p, default_file=True)
    add_evidence(p)
    p.set_defaults(func=cmd_review_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check" and args.file is None and not args.database:
        parser.error("check needs --file (offline) or --database (database audit)")
    try:
        return args.func(args)
    except (LoaderError, IngestRefused, ReviewError, EvidenceError, CliError) as exc:
        return _fail(str(exc))
    except UnicodeDecodeError:
        return _fail("a file is not valid UTF-8 (save CSV files as 'CSV UTF-8')")
    except json.JSONDecodeError as exc:
        return _fail(f"a JSON file is malformed ({exc.msg} at line {exc.lineno})")
    except csv.Error:
        return _fail("a CSV file is malformed")
    except OSError as exc:
        name = Path(exc.filename).name if exc.filename else "a file"
        return _fail(f"cannot access {name}: {exc.strerror or type(exc).__name__}")
    except (SQLAlchemyError, psycopg.Error) as exc:
        return _fail(
            f"database error ({type(exc).__name__}); check that the database exists, is reachable "
            "and has been migrated to head. Details are intentionally not shown"
        )


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
