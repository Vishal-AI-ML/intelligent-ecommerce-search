"""Database-side quality audit: rebuild records from stored rows and run the same rules."""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from ecommerce_search.catalog.mapping import flat_from_rows
from ecommerce_search.catalog.quality.findings import CheckOutcome, Finding, Severity
from ecommerce_search.catalog.quality.report import QualityReport, build_report
from ecommerce_search.catalog.schemas import (
    CatalogRecord,
    RecordValidationError,
    parse_raw,
    record_content_hash,
)
from ecommerce_search.ingestion.loader import scan_label_fields
from ecommerce_search.models.catalog import (
    SPEC_TABLES,
    CatalogDataset,
    Product,
    RawCatalogRecord,
)
from ecommerce_search.search.audit import CHECK_NAME as SEARCH_CHECK_NAME
from ecommerce_search.search.audit import audit_search_index
from ecommerce_search.search.dense_audit import CHECK_NAME as DENSE_CHECK_NAME
from ecommerce_search.search.dense_audit import audit_dense_index


def audit_database(session: Session) -> QualityReport:
    products = session.scalars(select(Product).order_by(Product.product_id)).all()
    datasets = {
        d.id: d for d in session.scalars(select(CatalogDataset).order_by(CatalogDataset.id))
    }
    raw_lines = {r.id: r.raw_line for r in session.scalars(select(RawCatalogRecord))}

    specs = {}
    for category, table in SPEC_TABLES.items():
        for spec in session.scalars(select(table)):
            specs[(category.value, spec.product_id)] = spec

    records: list[CatalogRecord] = []
    synthetic: dict[str, bool] = {}
    schema_findings: list[Finding] = []
    label_findings: list[Finding] = []
    linkage_findings: list[Finding] = []
    consistency_findings: list[Finding] = []

    def error(check: str, pid: str, message: str, field: str | None = None) -> Finding:
        return Finding(check, Severity.ERROR, message, product_id=pid, field=field)

    for product in products:
        spec = specs.get((product.category, product.product_id))
        if spec is None:
            linkage_findings.append(
                error("spec_linkage", product.product_id, f"no {product.category} attribute row")
            )
            continue
        try:
            record = parse_raw(flat_from_rows(product, spec))
        except RecordValidationError as exc:
            schema_findings.extend(
                error("schema_validation", product.product_id, msg, fld) for fld, msg in exc.errors
            )
            continue
        records.append(record)
        synthetic[product.product_id] = datasets[product.dataset_pk].is_synthetic

        raw_line = raw_lines.get(product.raw_record_id)
        try:
            raw = json.loads(raw_line) if raw_line is not None else None
            for name in scan_label_fields(raw) if isinstance(raw, dict) else []:
                label_findings.append(
                    error(
                        "evaluation_label_fields",
                        product.product_id,
                        f"raw line has field {name!r}",
                        name,
                    )
                )
            fresh = record_content_hash(parse_raw(raw))
        except (json.JSONDecodeError, RecordValidationError) as exc:
            consistency_findings.append(
                error(
                    "raw_normalized_consistency",
                    product.product_id,
                    f"stored raw line does not normalize: {exc}",
                )
            )
            continue
        if fresh != product.content_sha256 or record_content_hash(record) != product.content_sha256:
            consistency_findings.append(
                error(
                    "raw_normalized_consistency",
                    product.product_id,
                    "stored normalized values do not match the normalized raw line / stored hash",
                )
            )

    search_audit = audit_search_index(session)
    dense_audit = audit_dense_index(session)
    n = len(products)
    latest = (
        max(datasets.values(), key=lambda d: (int(d.dataset_version), d.id)) if datasets else None
    )
    dataset = {
        "dataset_id": latest.dataset_id if latest else None,
        "dataset_version": latest.dataset_version if latest else None,
        "is_synthetic": bool(latest and all(d.is_synthetic for d in datasets.values())),
        "checksum_sha256": latest.checksum_sha256 if latest else None,
        "datasets_in_database": [
            {
                "dataset_id": d.dataset_id,
                "dataset_version": d.dataset_version,
                "checksum_sha256": d.checksum_sha256,
                "is_synthetic": d.is_synthetic,
            }
            for d in datasets.values()
        ],
    }
    external = {
        "schema_validation": CheckOutcome(n, tuple(schema_findings)),
        "evaluation_label_fields": CheckOutcome(n, tuple(label_findings)),
        "spec_linkage": CheckOutcome(n, tuple(linkage_findings)),
        "raw_normalized_consistency": CheckOutcome(n, tuple(consistency_findings)),
        SEARCH_CHECK_NAME: search_audit.outcome,
        DENSE_CHECK_NAME: dense_audit.outcome,
    }
    return build_report(
        source="database",
        dataset=dataset,
        records=records,
        total_records=n,
        dataset_synthetic=synthetic,
        external=external,
        search_index=search_audit.metadata,
        dense_index=dense_audit.metadata,
    )
