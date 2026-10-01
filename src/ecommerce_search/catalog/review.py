"""Human-review workflow: deterministic sample, blank review CSV, guarded import.

The tooling never fills in a verdict and never marks a review complete on its own. Review
results are recorded only by `record_reviews`, which needs a named reviewer handle, an explicit
confirmation flag and a fully completed CSV whose non-review columns still match the sample
manifest exactly. Outcomes live in `catalog_reviews` (and the evidence file); they are never
written into product text.
"""

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ecommerce_search.catalog.evidence import (
    EvidenceError,
    evaluate_rows,
    validate_free_text,
    validate_handle,
)
from ecommerce_search.catalog.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    batch_id,
    manifest_digest,
    parse_manifest,
)
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.findings import Finding, Severity
from ecommerce_search.catalog.quality.report import QualityReport
from ecommerce_search.catalog.schemas import CatalogRecord
from ecommerce_search.catalog.taxonomy import (
    TAXONOMY_VERSION,
    TRANSFORM_VERSION,
    Category,
    ReviewVerdict,
)
from ecommerce_search.ingestion.loader import LoadedCatalog, Provenance
from ecommerce_search.models.catalog import CatalogDataset, CatalogReview, Product

SELECTION_VERSION = "1"  # bump when the selection algorithm below changes
SAMPLE_SEED = "m2-review-sample-v1"
SAMPLE_QUOTAS: dict[Category, int] = {
    Category.LAPTOP: 10,
    Category.PHONE: 10,
    Category.SHOES: 8,
    Category.HEADPHONES: 8,
}
SELECTION_METHOD = (
    "per category: (1) products with warning findings, ordered by sha256(seed:product_id), up to "
    "half the quota; (2) then one product per brand not yet represented, brands ordered by "
    "sha256(seed:brand), product with the smallest hash; (3) then fill by smallest "
    "sha256(seed:product_id)"
)
REVIEWER_COLUMNS = ("verdict", "issue_fields", "notes")
CSV_COLUMNS = (
    "product_id",
    "category",
    "subcategory",
    "title",
    "description",
    "brand",
    "price",
    "currency",
    "rating",
    "review_count",
    "availability",
    "seller_id",
    "is_synthetic",
    "dataset",
    "attributes",
    "quality_findings",
    "raw_line",
    "row_sha256",
    "sample_manifest_sha256",
    *REVIEWER_COLUMNS,
)
# Every non-review column except the two hash columns is covered by the row hash.
HASHED_COLUMNS = tuple(
    c for c in CSV_COLUMNS if c not in (*REVIEWER_COLUMNS, "row_sha256", "sample_manifest_sha256")
)
NUMERIC_COLUMNS = ("price", "rating", "review_count")


class ReviewError(Exception):
    """A review-sample or review-import guard clause failed."""


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def select_sample(
    records: Sequence[CatalogRecord],
    findings: Sequence[Finding],
    *,
    seed: str = SAMPLE_SEED,
    quotas: Mapping[Category, int] = SAMPLE_QUOTAS,
) -> list[str]:
    warned = {f.product_id for f in findings if f.severity is Severity.WARNING and f.product_id}
    selected: list[str] = []
    for category, quota in quotas.items():
        members = sorted(
            (r for r in records if r.category is category), key=lambda r: _rank(seed, r.product_id)
        )
        if len(members) < quota:
            raise ReviewError(f"only {len(members)} {category.value} products; quota is {quota}")
        chosen: list[CatalogRecord] = [r for r in members if r.product_id in warned][: quota // 2]
        seen_brands = {r.brand for r in chosen}
        for brand in sorted({r.brand for r in members}, key=lambda b: _rank(seed, b)):
            if len(chosen) >= quota:
                break
            if brand not in seen_brands:
                chosen.append(next(r for r in members if r.brand == brand))
                seen_brands.add(brand)
        for r in members:
            if len(chosen) >= quota:
                break
            if r not in chosen:
                chosen.append(r)
        selected.extend(sorted(r.product_id for r in chosen))
    return selected


# ---- row hash (binds the review to exactly what the reviewer saw) -------------------------


def _canon_cell(column: str, value: object) -> str:
    """Canonical form of a CSV cell. Numeric columns and the boolean are normalized so a
    spreadsheet round trip ('33999.00' -> '33999', 'true' -> 'TRUE') does not break the hash;
    text columns are compared exactly."""
    text = "" if value is None else str(value)
    if column in NUMERIC_COLUMNS:
        text = text.strip()
        if text:
            try:
                return format(Decimal(text).normalize(), "f")
            except InvalidOperation:
                return text
        return text
    if column == "is_synthetic":
        return text.strip().lower()
    return text


def row_hash(row: Mapping[str, object]) -> str:
    payload = {c: _canon_cell(c, row.get(c)) for c in HASHED_COLUMNS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    *, dataset: Mapping, products: Sequence[Mapping[str, str]], seed: str = SAMPLE_SEED
) -> dict:
    """`products`: [{product_id, content_sha256, row_sha256}] in sample order.

    The manifest records the sampling context (seed, selection version/method, quotas and the
    rules/taxonomy/transform versions) so completed evidence can embed it as a historical
    record that no later code change can alter."""
    body = {
        "manifest_kind": "catalog-review-sample",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_checksum_sha256": dataset["checksum_sha256"],
        "is_synthetic": dataset["is_synthetic"],
        "sample_seed": seed,
        "selection_version": SELECTION_VERSION,
        "selection_method": SELECTION_METHOD,
        "rules_version": qp.RULES_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "quotas": {c.value: n for c, n in SAMPLE_QUOTAS.items()},
        "product_ids": [p["product_id"] for p in products],
        "products": [
            {
                "product_id": p["product_id"],
                "content_sha256": p["content_sha256"],
                "row_sha256": p["row_sha256"],
            }
            for p in products
        ],
    }
    digest = manifest_digest(body)
    return {
        **body,
        "manifest_sha256": digest,
        "review_batch_id": batch_id(body["dataset_id"], body["dataset_version"], digest),
    }


def verify_manifest(manifest: Mapping) -> None:
    """Self-consistency only (schema, digest, batch id, product list). See
    `require_genuine_manifest` for the check against the current committed seed."""
    try:
        parse_manifest(manifest)
    except ManifestError as exc:
        raise ReviewError(str(exc)) from None


def require_genuine_manifest(supplied: Mapping, current: Mapping) -> None:
    """Refuse a manifest that is not exactly the sample the committed seed and the current
    sampling code generate, even if it is internally self-consistent (import-time check)."""
    verify_manifest(supplied)
    if dict(supplied) == dict(current):
        return
    differing = sorted(k for k in set(supplied) | set(current) if supplied.get(k) != current.get(k))
    raise ReviewError(
        "the manifest is not the review sample generated from the committed seed and the "
        f"current sampling code (differs in: {', '.join(differing)}). Regenerate it with "
        "review-sample; a hand-edited or stale manifest is never accepted for import"
    )


# ---- sample construction ------------------------------------------------------------------


def _base_row(
    record: CatalogRecord, raw_line: str, findings: Sequence[Finding], dataset: Mapping
) -> dict[str, str]:
    attributes = json.dumps(record.spec.model_dump(mode="json", exclude_none=True), sort_keys=True)
    return {
        "product_id": record.product_id,
        "category": record.category.value,
        "subcategory": record.subcategory or "",
        "title": record.title,
        "description": record.description or "",
        "brand": record.brand,
        "price": str(record.price),
        "currency": record.currency,
        "rating": "" if record.rating is None else str(record.rating),
        "review_count": "" if record.review_count is None else str(record.review_count),
        "availability": record.availability.value,
        "seller_id": record.seller_id or "",
        "is_synthetic": str(record.is_synthetic).lower(),
        "dataset": f"{dataset['dataset_id']} v{dataset['dataset_version']}",
        "attributes": attributes,
        "quality_findings": " | ".join(
            f"{f.severity.value}:{f.check}:{f.message}" for f in findings
        ),
        "raw_line": raw_line,
    }


@dataclass(frozen=True)
class ReviewSample:
    manifest: dict
    rows: list[dict[str, str]]


def prepare_sample(
    catalog: LoadedCatalog,
    provenance: Provenance,
    report: QualityReport,
    dataset: Mapping,
) -> ReviewSample:
    """Deterministic sample, manifest and blank CSV rows. Pure; needs no database."""
    ids = select_sample(catalog.records, report.findings)
    lines = {line.record.product_id: line for line in catalog.lines}
    by_product: dict[str, list[Finding]] = defaultdict(list)
    for f in report.findings:
        if f.product_id:
            by_product[f.product_id].append(f)
    base = {
        pid: _base_row(lines[pid].record, lines[pid].raw_line, by_product.get(pid, []), dataset)
        for pid in ids
    }
    manifest = build_manifest(
        dataset=dataset,
        products=[
            {
                "product_id": pid,
                "content_sha256": lines[pid].content_sha256,
                "row_sha256": row_hash(base[pid]),
            }
            for pid in ids
        ],
    )
    rows = [
        {
            **base[pid],
            "row_sha256": row_hash(base[pid]),
            "sample_manifest_sha256": manifest["manifest_sha256"],
            "verdict": "",
            "issue_fields": "",
            "notes": "",
        }
        for pid in ids
    ]
    return ReviewSample(manifest=manifest, rows=rows)


INSTRUCTIONS = """\
# Catalog human-review instructions

All records are SYNTHETIC (project-generated); this is not marketplace data.

1. **Back up first.** Copy `review_sample.csv` somewhere safe before you edit it. This tool never
   overwrites a CSV that contains any reviewer input; if you regenerate the sample into a new
   directory your edits are not carried over.
2. Open `review_sample.csv` in a plain text editor or a spreadsheet. Every row is one product.
   Read the title, description, price, brand, attributes and `quality_findings`, and compare
   them with `raw_line`.
3. Fill in the `verdict` column for EVERY row: `accept`, `needs_correction` or `unsure`.
   `issue_fields` (which fields look wrong) and `notes` are optional. **Edit only those three
   columns.** Every other column is checked against the sample manifest and the import is
   refused if any of them changed.
4. **Save as "CSV UTF-8"** (UTF-8 with or without BOM, comma separated). Other encodings, such as
   the default Excel "CSV (Comma delimited)", are rejected.
5. **Privacy:** never put an email address, personal information, secrets or machine paths in
   `issue_fields` or `notes`. Use a short handle or pseudonym as your reviewer name (never an
   email).
6. Record the review yourself, naming a target database explicitly:

       uv run python -m ecommerce_search.catalog review-import --manifest review_manifest.json \\
           --csv review_sample.csv --database <database> --reviewer "<handle>" \\
           --confirm-human-review

   This writes the append-only database rows and a versioned evidence file under
   `data/labels/`. Commit that evidence file to keep a durable audit trail. It is an audit
   trail (Git history plus a reviewer handle), not cryptographic proof of identity.
7. Check progress offline with `review-status` and re-check the committed evidence with
   `review-verify`. The status stays PENDING until a verified evidence file covers every
   sampled product. Nothing here is filled in or approved automatically.
"""


def sample_directory(output_dir: Path, manifest: Mapping) -> Path:
    return output_dir / manifest["review_batch_id"]


def _render_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return ("﻿" + buffer.getvalue()).encode("utf-8")


def ensure_blank_sample(csv_path: Path) -> None:
    """Raise unless `csv_path` is a readable review CSV with every reviewer cell blank."""
    try:
        rows = read_review_csv(csv_path)
    except ReviewError as exc:
        raise ReviewError(f"{csv_path.name} cannot be proven blank, so it is kept: {exc}") from None
    if not rows:
        raise ReviewError(f"{csv_path.name} has no rows, so it cannot be proven blank")
    for row in rows:
        if any((row.get(c) or "").strip() for c in REVIEWER_COLUMNS):
            raise ReviewError(
                f"{csv_path.name} contains reviewer input and is never overwritten "
                "(move or back it up yourself if you really want to discard it)"
            )


def _write_bytes(path: Path, data: bytes, *, replace: bool) -> None:
    if replace:
        temp = path.with_name(path.name + ".tmp")
        temp.write_bytes(data)
        os.replace(temp, path)
        return
    try:
        with open(path, "xb") as handle:  # noqa: PTH123 - exclusive create
            handle.write(data)
    except FileExistsError:
        raise ReviewError(f"{path.name} already exists; refusing to overwrite it") from None


def write_review_artifacts(
    directory: Path, sample: ReviewSample, *, force: bool = False
) -> tuple[Path, Path, Path]:
    """Write the CSV, manifest and instructions without ever destroying review work.

    Refuses if any target exists. `force` replaces only a sample whose CSV is provably blank.
    """
    csv_path = directory / "review_sample.csv"
    manifest_path = directory / "review_manifest.json"
    readme_path = directory / "REVIEW_INSTRUCTIONS.md"
    existing = [p.name for p in (csv_path, manifest_path, readme_path) if p.exists()]
    if existing:
        if not force:
            raise ReviewError(
                f"refusing to overwrite existing review files ({', '.join(existing)}). Nothing "
                "was changed. --force only replaces a provably blank, untouched sample"
            )
        if csv_path.exists():
            ensure_blank_sample(csv_path)
    contents = (
        (csv_path, _render_csv(sample.rows)),
        (
            manifest_path,
            (json.dumps(sample.manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        ),
        (readme_path, INSTRUCTIONS.encode("utf-8")),
    )
    directory.mkdir(parents=True, exist_ok=True)
    for path, data in contents:
        _write_bytes(path, data, replace=force and path.exists())
    return csv_path, manifest_path, readme_path


def read_review_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames or []
    except UnicodeDecodeError:
        raise ReviewError(
            f"{path.name} is not valid UTF-8. Save the file as 'CSV UTF-8' and try again"
        ) from None
    except csv.Error as exc:
        raise ReviewError(f"{path.name} is not a well-formed CSV file ({exc})") from None
    missing = [c for c in CSV_COLUMNS if c not in fields]
    if missing:
        raise ReviewError(
            f"{path.name} is missing required column(s): {', '.join(missing)}. "
            "Do not rename or remove columns; save as 'CSV UTF-8'"
        )
    return rows


# ---- import ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRow:
    product_id: str
    product_content_sha256: str
    verdict: str
    issue_fields: str | None
    notes: str | None


def validate_import(
    *,
    manifest: Mapping,
    rows: Sequence[Mapping[str, str]],
    reviewer: str | None,
    confirmed: bool,
) -> list[ReviewRow]:
    """Guard clauses. Raises ReviewError; never fills in or defaults a verdict."""
    if not confirmed:
        raise ReviewError("refusing to record: pass --confirm-human-review (a human must review)")
    try:
        validate_handle(reviewer)
    except EvidenceError as exc:
        raise ReviewError(str(exc)) from None
    verify_manifest(manifest)
    if any(col not in row for row in rows for col in ("product_id", *REVIEWER_COLUMNS)):
        raise ReviewError("CSV rows are missing required columns")

    ids = [r["product_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ReviewError("CSV contains duplicated product ids")
    if sorted(ids) != sorted(manifest["product_ids"]):
        raise ReviewError("CSV product ids do not match the manifest exactly")
    by_id = {r["product_id"]: r for r in rows}
    expected = {p["product_id"]: p for p in manifest["products"]}
    allowed = {v.value for v in ReviewVerdict}
    out = []
    for pid in manifest["product_ids"]:  # manifest order
        row = by_id[pid]
        if row.get("sample_manifest_sha256") != manifest["manifest_sha256"]:
            raise ReviewError("CSV rows were produced from a different manifest")
        if (
            row_hash(row) != expected[pid]["row_sha256"]
            or row.get("row_sha256") != (expected[pid]["row_sha256"])
        ):
            raise ReviewError(
                f"{pid}: a non-review column was modified (row hash mismatch). Edit only "
                "verdict, issue_fields and notes"
            )
        verdict = (row["verdict"] or "").strip()
        if not verdict:
            raise ReviewError(f"{pid}: verdict is blank; every row must be reviewed")
        if verdict not in allowed:
            raise ReviewError(f"{pid}: invalid verdict {verdict!r}; allowed {sorted(allowed)}")
        try:
            issue_fields = validate_free_text(
                f"{pid} issue_fields", (row["issue_fields"] or "").strip() or None
            )
            notes = validate_free_text(f"{pid} notes", (row["notes"] or "").strip() or None)
        except EvidenceError as exc:
            raise ReviewError(str(exc)) from None
        out.append(ReviewRow(pid, expected[pid]["content_sha256"], verdict, issue_fields, notes))
    return out


def record_reviews(
    session: Session,
    *,
    manifest: Mapping,
    reviews: Sequence[ReviewRow],
    reviewer: str,
    recorded_at: datetime,
) -> int:
    """Append review rows in the caller's transaction. Refuses unless the live products still
    have exactly the content that was sampled and reviewed."""
    dataset = session.scalar(
        select(CatalogDataset).where(
            CatalogDataset.dataset_id == manifest["dataset_id"],
            CatalogDataset.dataset_version == manifest["dataset_version"],
        )
    )
    if dataset is None or dataset.checksum_sha256 != manifest["dataset_checksum_sha256"]:
        raise ReviewError("the manifest's dataset version/checksum is not in the database")
    live = dict(
        session.execute(
            select(Product.product_id, Product.content_sha256).where(
                Product.product_id.in_([r.product_id for r in reviews])
            )
        ).all()
    )
    missing = sorted({r.product_id for r in reviews} - set(live))
    if missing:
        raise ReviewError(f"products not in the database: {', '.join(missing[:5])}")
    changed = sorted(
        r.product_id for r in reviews if live[r.product_id] != r.product_content_sha256
    )
    if changed:
        raise ReviewError(
            f"{len(changed)} product(s) changed after the sample was made "
            f"(e.g. {', '.join(changed[:3])}); regenerate the sample and review again"
        )
    try:
        for review in reviews:
            session.add(
                CatalogReview(
                    review_batch_id=manifest["review_batch_id"],
                    product_id=review.product_id,
                    dataset_pk=dataset.id,
                    product_content_sha256=review.product_content_sha256,
                    verdict=review.verdict,
                    issue_fields=review.issue_fields,
                    notes=review.notes,
                    reviewer=reviewer.strip(),
                    sample_manifest_sha256=manifest["manifest_sha256"],
                    recorded_at=recorded_at,
                )
            )
        session.flush()
    except IntegrityError:
        raise ReviewError(
            "these reviews could not be recorded (already recorded for this batch and reviewer?)"
        ) from None
    return len(reviews)


def database_review_rows(session: Session, manifest: Mapping) -> list[dict[str, str]]:
    stmt = (
        select(CatalogReview, CatalogDataset)
        .join(CatalogDataset, CatalogDataset.id == CatalogReview.dataset_pk)
        .where(CatalogReview.review_batch_id == manifest["review_batch_id"])
        .order_by(CatalogReview.id)
    )
    return [
        {
            "product_id": review.product_id,
            "verdict": review.verdict,
            "reviewer": review.reviewer,
            "manifest_sha256": review.sample_manifest_sha256,
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.dataset_version,
            "dataset_checksum_sha256": dataset.checksum_sha256,
            "product_content_sha256": review.product_content_sha256,
        }
        for review, dataset in session.execute(stmt).all()
    ]


def database_review_status(session: Session, manifest: Mapping):
    return evaluate_rows(manifest, database_review_rows(session, manifest), source="database")
