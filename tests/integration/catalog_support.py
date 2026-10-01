"""Shared helpers for the catalog integration tests (scratch databases only)."""

import hashlib
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ecommerce_search.models.catalog import (
    SPEC_TABLES,
    CatalogDataset,
    CatalogReview,
    Product,
    RawCatalogRecord,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"
TEST_REVIEWER = "TEST-ONLY-REVIEWER"
EMPTY_COUNTS = {"datasets": 0, "raw": 0, "products": 0, "specs": 0, "reviews": 0}


def counts(engine):
    with Session(engine) as s:
        return {
            "datasets": s.scalar(select(func.count()).select_from(CatalogDataset)),
            "raw": s.scalar(select(func.count()).select_from(RawCatalogRecord)),
            "products": s.scalar(select(func.count()).select_from(Product)),
            "specs": sum(
                s.scalar(select(func.count()).select_from(t)) for t in SPEC_TABLES.values()
            ),
            "reviews": s.scalar(select(func.count()).select_from(CatalogReview)),
        }


def snapshot(engine):
    """Everything an ingest could change, for before/after equality assertions."""
    with Session(engine) as s:
        products = {
            p.product_id: (p.content_sha256, p.created_at, p.updated_at, p.dataset_pk, p.price)
            for p in s.scalars(select(Product))
        }
        datasets = {
            (d.dataset_id, d.dataset_version): (
                d.id,
                d.checksum_sha256,
                d.transform_version,
                d.taxonomy_version,
                d.rules_version,
                d.rights_note,
                d.record_count,
                d.notes,
                d.generator,
                d.authored_on,
            )
            for d in s.scalars(select(CatalogDataset))
        }
    return {"counts": counts(engine), "products": products, "datasets": datasets}


def make_variant(tmp_path, lines, *, version, name="variant", **provenance_changes):
    """Write a catalog file plus a matching provenance file for a (modified) dataset file."""
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8") if lines else b"")
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prov.update(
        dataset_version=version,
        record_count=len(lines),
        checksum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    prov.update(provenance_changes)
    prov_path = tmp_path / f"{name}.provenance.json"
    prov_path.write_text(json.dumps(prov), encoding="utf-8")
    return path, prov_path


def seed_lines():
    return SEED.read_text(encoding="utf-8").splitlines()


def edit_line(line, **changes):
    return json.dumps({**json.loads(line), **changes}, ensure_ascii=False, separators=(",", ":"))
