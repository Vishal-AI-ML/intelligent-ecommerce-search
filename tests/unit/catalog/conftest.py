import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Catalog unit tests must never touch the network (or a database)."""

    def guard(*args, **kwargs):
        raise RuntimeError("network access is not allowed in catalog unit tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


from pathlib import Path  # noqa: E402

from ecommerce_search.catalog.review import prepare_sample  # noqa: E402
from ecommerce_search.ingestion.loader import load_catalog, load_provenance  # noqa: E402
from ecommerce_search.ingestion.service import build_file_report, dataset_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"


@pytest.fixture(scope="session")
def seed_catalog():
    return load_catalog(SEED)


@pytest.fixture(scope="session")
def seed_provenance():
    return load_provenance(PROVENANCE)


@pytest.fixture(scope="session")
def seed_report(seed_catalog, seed_provenance):
    return build_file_report(seed_catalog, seed_provenance)


@pytest.fixture(scope="session")
def seed_sample(seed_catalog, seed_provenance, seed_report):
    """The deterministic review sample derived from the committed seed (never written)."""
    return prepare_sample(
        seed_catalog,
        seed_provenance,
        seed_report,
        dataset_info(seed_provenance, seed_catalog.checksum_sha256),
    )


from ecommerce_search.catalog.evidence import SeedFacts  # noqa: E402


@pytest.fixture(scope="session")
def seed_facts(seed_catalog, seed_provenance):
    """What historical evidence verification may know about the committed seed."""
    return SeedFacts(
        dataset_id=seed_provenance.dataset_id,
        dataset_version=seed_provenance.dataset_version,
        dataset_checksum_sha256=seed_catalog.checksum_sha256,
        content_hashes={line.record.product_id: line.content_sha256 for line in seed_catalog.lines},
    )
