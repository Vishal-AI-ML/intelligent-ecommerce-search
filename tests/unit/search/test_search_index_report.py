"""File-side quality reports must state that the search-index audit is not applicable."""

import json
from pathlib import Path

from ecommerce_search.catalog.quality.report import (
    SEARCH_INDEX_FILE_REASON,
    to_markdown,
    write_report,
)
from ecommerce_search.ingestion.loader import load_catalog, load_provenance
from ecommerce_search.ingestion.service import build_file_report

ROOT = Path(__file__).resolve().parents[3]


def file_report():
    catalog = load_catalog(ROOT / "data" / "seed" / "catalog_seed_v1.jsonl")
    provenance = load_provenance(ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json")
    return build_file_report(catalog, provenance)


def test_file_report_states_that_the_search_index_audit_is_not_applicable(tmp_path):
    report = file_report()
    section = report.to_dict()["search_index"]
    assert section == {"applicable": False, "reason": SEARCH_INDEX_FILE_REASON}
    assert "no stored search index" in section["reason"]
    checks = {c["name"]: c for c in report.checks}
    assert checks["embedding_search_text_leakage"]["status"] == "not_applicable"
    json_path, _md = write_report(report, tmp_path, "file")
    assert json.loads(json_path.read_text(encoding="utf-8"))["search_index"] == section
    assert "## Search index" in to_markdown(report)
    assert "applicable: False" in to_markdown(report)


def test_file_report_never_mixes_search_versions_into_the_catalog_versions_block():
    versions = file_report().to_dict()["versions"]
    assert set(versions) == {"rules_version", "taxonomy_version", "transform_version"}
    assert versions["rules_version"] == "1"  # the derived-index audit does not bump the M2 rules
