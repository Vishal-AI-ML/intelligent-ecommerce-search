"""Validate, then persist a catalog file atomically and idempotently.

Nothing is written unless the whole file passes the error-level checks. Persistence is a single
transaction that first takes a transaction-scoped advisory lock for the dataset family, so
concurrent ingests of the same dataset id are serialized.

Semantics:
* same dataset version and checksum: idempotent (unchanged products are not touched);
* same version, different checksum: refused;
* an older version than one already ingested: refused (no rollback of newer data);
* an existing version whose recorded provenance/versions differ from the current code or file:
  refused (a new dataset version is required);
* a newer version: changed products are upserted, absent products are reported, never deleted.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ecommerce_search.catalog.mapping import core_values, spec_values
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.report import QualityReport, build_report
from ecommerce_search.catalog.taxonomy import TAXONOMY_VERSION, TRANSFORM_VERSION, Category
from ecommerce_search.ingestion.loader import (
    LoadedCatalog,
    Provenance,
    load_catalog,
    load_provenance,
    verify_against_provenance,
)
from ecommerce_search.models.catalog import (
    SPEC_TABLES,
    CatalogDataset,
    Product,
    RawCatalogRecord,
)


class IngestRefused(Exception):
    """Ingestion was refused before any change was made."""


@dataclass
class IngestResult:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    raw_records_inserted: int = 0
    dataset_created: bool = False
    absent_product_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"inserted={self.inserted} updated={self.updated} unchanged={self.unchanged} "
            f"raw_records_inserted={self.raw_records_inserted} "
            f"dataset_created={self.dataset_created} "
            f"absent_from_file={len(self.absent_product_ids)}"
        )


@dataclass
class IngestOutcome:
    report: QualityReport
    result: IngestResult | None  # None when validation errors blocked persistence


@dataclass
class PreparedIngest:
    """A loaded, provenance-verified and quality-checked file. No database access yet."""

    catalog: LoadedCatalog
    provenance: Provenance
    report: QualityReport

    @property
    def blocked(self) -> bool:
        return self.report.error_count > 0


def ingest_lock_key(dataset_id: str) -> int:
    """Deterministic signed 64-bit key for the per-dataset-family advisory lock.

    Derived from SHA-256 (never Python's process-randomized hash())."""
    digest = hashlib.sha256(f"catalog-ingest:{dataset_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def dataset_info(provenance: Provenance, checksum: str) -> dict:
    return {
        "dataset_id": provenance.dataset_id,
        "dataset_version": provenance.dataset_version,
        "is_synthetic": provenance.is_synthetic,
        "checksum_sha256": checksum,
        "source_description": provenance.source_description,
        "rights_note": provenance.rights_note,
        "license_identifier": provenance.license_identifier,
        "generator": provenance.generator,
        "authored_on": provenance.authored_on.isoformat(),
    }


def build_file_report(catalog: LoadedCatalog, provenance: Provenance) -> QualityReport:
    return build_report(
        source="file",
        dataset=dataset_info(provenance, catalog.checksum_sha256),
        records=catalog.records,
        total_records=catalog.total_lines,
        dataset_synthetic={r.product_id: provenance.is_synthetic for r in catalog.records},
        external=catalog.outcomes,
    )


def prepare_ingest(catalog_path: Path, provenance_path: Path) -> PreparedIngest:
    """Load, verify and check the input. Needs no database and no settings."""
    provenance = load_provenance(provenance_path)
    catalog = load_catalog(catalog_path)
    verify_against_provenance(catalog, provenance)
    return PreparedIngest(catalog, provenance, build_file_report(catalog, provenance))


def ingest_file(engine: Engine, catalog_path: Path, provenance_path: Path) -> IngestOutcome:
    prepared = prepare_ingest(catalog_path, provenance_path)
    if prepared.blocked:
        return IngestOutcome(report=prepared.report, result=None)
    result = persist(engine, prepared.catalog, prepared.provenance)
    return IngestOutcome(report=prepared.report, result=result)


def _provenance_drift(
    dataset: CatalogDataset, catalog: LoadedCatalog, provenance: Provenance
) -> list[str]:
    """Names of recorded fields that differ from what the current code/file would record."""
    expected = {
        "transform_version": TRANSFORM_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "rules_version": qp.RULES_VERSION,
        "source_description": provenance.source_description,
        "rights_note": provenance.rights_note,
        "license_identifier": provenance.license_identifier,
        "authored_on": provenance.authored_on,
        "generator": provenance.generator,
        "is_synthetic": provenance.is_synthetic,
        "record_count": catalog.total_lines,
        "notes": provenance.notes,
    }
    return [name for name, value in expected.items() if getattr(dataset, name) != value]


def persist(engine: Engine, catalog: LoadedCatalog, provenance: Provenance) -> IngestResult:
    result = IngestResult()
    try:
        with Session(engine) as session, session.begin():
            # Serialize every read-then-write decision for this dataset family.
            session.execute(
                select(func.pg_advisory_xact_lock(ingest_lock_key(provenance.dataset_id)))
            )

            family = list(
                session.scalars(
                    select(CatalogDataset).where(CatalogDataset.dataset_id == provenance.dataset_id)
                )
            )
            wanted = int(provenance.dataset_version)
            newest = max((int(d.dataset_version) for d in family), default=None)
            if newest is not None and wanted < newest:
                raise IngestRefused(
                    f"dataset {provenance.dataset_id} version {wanted} is older than the already "
                    f"ingested version {newest}; refusing to roll catalog data back"
                )
            dataset = next(
                (d for d in family if d.dataset_version == provenance.dataset_version), None
            )
            if dataset is not None:
                if dataset.checksum_sha256 != catalog.checksum_sha256:
                    raise IngestRefused(
                        f"dataset {provenance.dataset_id} version {provenance.dataset_version} was "
                        f"already ingested with checksum {dataset.checksum_sha256}; the file has "
                        f"{catalog.checksum_sha256}. Bump the dataset version instead of changing "
                        "a published version."
                    )
                drift = _provenance_drift(dataset, catalog, provenance)
                if drift:
                    raise IngestRefused(
                        f"dataset {provenance.dataset_id} version {provenance.dataset_version} was "
                        f"ingested with different {', '.join(drift)}; a new dataset version is "
                        "required (products are never rewritten under old provenance)"
                    )
            else:
                dataset = CatalogDataset(
                    dataset_id=provenance.dataset_id,
                    dataset_version=provenance.dataset_version,
                    source_description=provenance.source_description,
                    rights_note=provenance.rights_note,
                    license_identifier=provenance.license_identifier,
                    authored_on=provenance.authored_on,
                    generator=provenance.generator,
                    is_synthetic=provenance.is_synthetic,
                    transform_version=TRANSFORM_VERSION,
                    taxonomy_version=TAXONOMY_VERSION,
                    rules_version=qp.RULES_VERSION,
                    checksum_sha256=catalog.checksum_sha256,
                    record_count=catalog.total_lines,
                    notes=provenance.notes,
                )
                session.add(dataset)
                session.flush()
                family.append(dataset)
                result.dataset_created = True

            existing_raw = dict(
                session.execute(
                    select(RawCatalogRecord.source_key, RawCatalogRecord.id).where(
                        RawCatalogRecord.dataset_pk == dataset.id
                    )
                ).all()
            )
            raw_ids: dict[str, int] = {}
            for line in catalog.lines:
                key = line.record.product_id
                if key in existing_raw:
                    raw_ids[key] = existing_raw[key]
                    continue
                raw = RawCatalogRecord(
                    dataset_pk=dataset.id,
                    source_key=key,
                    line_number=line.line_number,
                    raw_line=line.raw_line,
                    raw_sha256=line.raw_sha256,
                )
                session.add(raw)
                session.flush()
                raw_ids[key] = raw.id
                result.raw_records_inserted += 1

            for line in catalog.lines:
                record = line.record
                product = session.get(Product, record.product_id)
                if product is None:
                    session.add(
                        Product(
                            product_id=record.product_id,
                            dataset_pk=dataset.id,
                            raw_record_id=raw_ids[record.product_id],
                            content_sha256=line.content_sha256,
                            **core_values(record),
                        )
                    )
                    session.flush()
                    session.add(
                        SPEC_TABLES[record.category](
                            product_id=record.product_id,
                            category=record.category.value,
                            **spec_values(record),
                        )
                    )
                    result.inserted += 1
                elif product.content_sha256 == line.content_sha256:
                    result.unchanged += 1
                else:
                    if product.category != record.category.value:
                        raise IngestRefused(
                            f"{record.product_id} exists with category {product.category!r}; "
                            f"refusing to change it to {record.category.value!r}"
                        )
                    for name, value in core_values(record).items():
                        setattr(product, name, value)
                    product.dataset_pk = dataset.id
                    product.raw_record_id = raw_ids[record.product_id]
                    product.content_sha256 = line.content_sha256
                    product.updated_at = func.now()
                    spec = session.get(SPEC_TABLES[record.category], record.product_id)
                    values = spec_values(record)
                    if spec is None:
                        session.add(
                            SPEC_TABLES[record.category](
                                product_id=record.product_id,
                                category=record.category.value,
                                **values,
                            )
                        )
                    else:
                        for name, value in values.items():
                            setattr(spec, name, value)
                    result.updated += 1
                session.flush()

            in_file = {line.record.product_id for line in catalog.lines}
            known = session.scalars(
                select(Product.product_id).where(Product.dataset_pk.in_([d.id for d in family]))
            )
            result.absent_product_ids = sorted(set(known) - in_file)
    except IntegrityError as exc:
        # Expected only if something bypassed the advisory lock; never echo SQL or parameters.
        name = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        detail = f" ({name})" if name else ""
        raise IngestRefused(
            f"the database rejected the write{detail}; nothing was written. "
            "Another ingest may have run at the same time: retry"
        ) from None
    return result


def category_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(Product.category, func.count()).group_by(Product.category)).all()
    return {Category(c).value: n for c, n in sorted(rows)}
