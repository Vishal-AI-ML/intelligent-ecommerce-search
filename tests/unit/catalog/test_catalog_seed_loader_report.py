import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.findings import Severity
from ecommerce_search.catalog.quality.report import SYNTHETIC_NOTICE, to_markdown, write_report
from ecommerce_search.ingestion.loader import (
    LoaderError,
    Provenance,
    load_catalog,
    load_provenance,
    verify_against_provenance,
)
from ecommerce_search.ingestion.service import build_file_report

ROOT = Path(__file__).resolve().parents[3]
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "seed_generator", ROOT / "scripts" / "generate_seed_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seed_catalog():
    return load_catalog(SEED)


# ---- deterministic seed -----------------------------------------------------------------


def test_seed_regenerates_byte_for_byte(generator):
    data, provenance = generator.render()
    assert generator.render()[0] == data  # deterministic within a process
    assert SEED.read_bytes() == data
    assert PROVENANCE.read_bytes() == generator.provenance_bytes(provenance)
    assert hashlib.sha256(data).hexdigest() == provenance["checksum_sha256"]


def test_seed_has_no_crlf_and_gitattributes_pins_lf():
    assert b"\r" not in SEED.read_bytes()
    assert "data/seed/* text eol=lf" in (ROOT / ".gitattributes").read_text()


def test_seed_counts_and_brands(seed_catalog):
    counts: dict[str, int] = {}
    for record in seed_catalog.records:
        counts[record.category.value] = counts.get(record.category.value, 0) + 1
    assert counts == {"laptop": 80, "phone": 60, "shoes": 50, "headphones": 50}
    assert len({r.product_id for r in seed_catalog.records}) == 240
    for record in seed_catalog.records:
        assert record.category in tx.BRAND_CATEGORIES[record.brand]


def test_seed_is_entirely_synthetic_and_label_free(seed_catalog):
    assert all(
        r.is_synthetic and r.source_type is tx.SourceType.SYNTHETIC for r in seed_catalog.records
    )
    assert all(
        r.seller_id is None or r.seller_id.startswith("synthetic-seller-")
        for r in seed_catalog.records
    )
    assert all(r.product_id.startswith("SYN-") for r in seed_catalog.records)
    for line in SEED.read_text(encoding="utf-8").splitlines():
        assert not {k.casefold() for k in json.loads(line)} & qp.FORBIDDEN_LABEL_FIELDS


def test_seed_contains_raw_formatting_variants_that_normalize(seed_catalog):
    raw = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines()]
    assert sum(isinstance(r["price"], str) for r in raw) >= 4
    assert sum(r["brand"] != r["brand"].strip() for r in raw) >= 4
    assert sum(isinstance(r.get("storage_gb"), str) for r in raw) >= 2
    # normalized values never keep the variant formatting
    assert all(
        r.brand == r.brand.strip() and r.brand in tx.BRAND_CATEGORIES for r in seed_catalog.records
    )


def test_seed_provenance_is_honest_and_verified(seed_catalog):
    provenance = load_provenance(PROVENANCE)
    verify_against_provenance(seed_catalog, provenance)
    assert provenance.is_synthetic is True
    assert provenance.license_identifier is None
    text = (provenance.source_description + provenance.rights_note).lower()
    assert "no third-party source" in text and "no marketplace source" in text
    assert "no formal licence identifier" in text
    assert not (ROOT / "LICENSE").exists()


def test_seed_passes_error_level_checks_with_real_evaluation(seed_catalog):
    report = build_file_report(seed_catalog, load_provenance(PROVENANCE))
    assert report.error_count == 0
    by_name = {c["name"]: c for c in report.checks}
    assert by_name["unique_product_id"]["evaluated"] == 240
    assert by_name["embedding_search_text_leakage"]["status"] == "not_applicable"
    assert by_name["spec_linkage"]["status"] == "not_applicable"  # database-only check


def test_no_network_libraries_are_imported_by_catalog_code():
    forbidden = re.compile(
        r"^\s*(import|from)\s+(requests|urllib|httpx|socket|http\.client|aiohttp)\b", re.M
    )
    files = [ROOT / "scripts" / "generate_seed_catalog.py"]
    for package in ("catalog", "ingestion", "models"):
        files += list((ROOT / "src" / "ecommerce_search" / package).rglob("*.py"))
    assert files
    assert [str(f) for f in files if forbidden.search(f.read_text(encoding="utf-8"))] == []


# ---- loader -----------------------------------------------------------------------------


def write_lines(tmp_path, lines):
    path = tmp_path / "catalog.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_loader_reports_bad_lines_with_line_numbers(tmp_path, raw_record):
    good = json.dumps(raw_record("laptop"))
    bad_unit = json.dumps(raw_record("phone", ram_gb="4 MB"))
    path = write_lines(tmp_path, [good, "{not json", "", bad_unit])
    catalog = load_catalog(path)
    findings = catalog.outcomes["schema_validation"].findings
    assert catalog.total_lines == 4 and len(catalog.lines) == 1
    assert sorted(f.line for f in findings) == [2, 3, 4]
    assert all(f.severity is Severity.ERROR for f in findings)
    assert any(f.product_id == "T-PHN-1" and f.field == "ram_gb" for f in findings)


def test_loader_flags_label_fields_and_still_rejects_them(tmp_path, raw_record):
    path = write_lines(tmp_path, [json.dumps(raw_record("laptop", relevance="relevant"))])
    catalog = load_catalog(path)
    assert catalog.outcomes["evaluation_label_fields"].findings[0].field == "relevance"
    assert catalog.outcomes["schema_validation"].findings  # unknown attribute also rejected
    assert not catalog.lines


def test_loader_raw_line_and_checksum_are_exact(tmp_path, raw_record):
    line = json.dumps(raw_record("laptop"), separators=(",", ":"))
    path = write_lines(tmp_path, [line])
    catalog = load_catalog(path)
    assert catalog.lines[0].raw_line == line
    assert catalog.checksum_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_rejects_non_utf8(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(LoaderError):
        load_catalog(path)


def provenance_dict(catalog):
    data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    data.update(checksum_sha256=catalog.checksum_sha256, record_count=catalog.total_lines)
    return data


def test_provenance_verification_refuses_mismatches(seed_catalog):
    good = Provenance.model_validate(provenance_dict(seed_catalog))
    verify_against_provenance(seed_catalog, good)
    for change in (
        {"checksum_sha256": "0" * 64},
        {"record_count": 239},
        {"transform_version": "999"},
        {"taxonomy_version": "999"},
    ):
        with pytest.raises(LoaderError):
            verify_against_provenance(seed_catalog, good.model_copy(update=change))


def test_provenance_rejects_unknown_keys_and_allows_null_licence(seed_catalog):
    data = provenance_dict(seed_catalog)
    Provenance.model_validate(data | {"license_identifier": None})
    with pytest.raises(ValueError):
        Provenance.model_validate(data | {"surprise": 1})


# ---- reports ----------------------------------------------------------------------------


def test_report_is_deterministic_and_marks_synthetic(seed_catalog):
    provenance = load_provenance(PROVENANCE)
    first = build_file_report(seed_catalog, provenance)
    second = build_file_report(load_catalog(SEED), provenance)
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    assert to_markdown(first) == to_markdown(second)
    assert first.to_dict()["synthetic_notice"] == SYNTHETIC_NOTICE
    assert "SYNTHETIC" in to_markdown(first)
    assert "not measured marketplace facts" in to_markdown(first)


def test_report_status_reflects_findings(tmp_path, raw_record, seed_catalog):
    bad = write_lines(tmp_path, [json.dumps(raw_record("laptop", price=-1))])
    catalog = load_catalog(bad)
    provenance = Provenance.model_validate(provenance_dict(catalog))
    report = build_file_report(catalog, provenance)
    statuses = {c["name"]: c["status"] for c in report.checks}
    assert statuses["schema_validation"] == "fail" and report.error_count >= 1
    assert statuses["unique_product_id"] == "not_applicable"  # nothing valid to evaluate


def test_write_report_only_adds_timestamp_outside_the_data(tmp_path, seed_catalog):
    report = build_file_report(seed_catalog, load_provenance(PROVENANCE))
    json_path, md_path = write_report(report, tmp_path, "q")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    run = payload.pop("run")
    assert set(run) == {"generated_at"}
    assert payload == json.loads(json.dumps(report.to_dict()))
    assert md_path.read_text(encoding="utf-8").startswith("# Catalog quality report")


# ---- I3: an empty catalog is an error, never a pass or "not applicable" -------------------


def test_empty_catalog_file_is_an_error_not_a_pass(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    catalog = load_catalog(empty)
    # a provenance file that claims zero records must not make an empty ingestion succeed
    provenance = Provenance.model_validate(provenance_dict(catalog))
    assert provenance.record_count == 0
    report = build_file_report(catalog, provenance)
    status = {c["name"]: c for c in report.checks}["catalog_non_empty"]
    assert status["status"] == "fail" and status["status"] != "not_applicable"
    assert report.error_count == 1 and report.to_dict()["summary"]["empty"] is True
    assert "EMPTY" in report.findings[0].message


def test_non_empty_catalogs_pass_the_emptiness_check(seed_catalog):
    report = build_file_report(seed_catalog, load_provenance(PROVENANCE))
    check = {c["name"]: c for c in report.checks}["catalog_non_empty"]
    assert check["status"] == "pass" and report.to_dict()["summary"]["empty"] is False


def test_blank_only_file_is_also_an_error(tmp_path):
    blank = tmp_path / "blank.jsonl"
    blank.write_bytes(b"\n")
    catalog = load_catalog(blank)
    report = build_file_report(catalog, Provenance.model_validate(provenance_dict(catalog)))
    assert report.error_count >= 1  # the blank line is invalid JSON


def test_cli_refuses_an_empty_catalog_without_touching_a_database(tmp_path, capsys):
    from ecommerce_search.catalog.cli import main

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    catalog = load_catalog(empty)
    prov_path = tmp_path / "empty.provenance.json"
    prov_path.write_text(json.dumps(provenance_dict(catalog)), encoding="utf-8")
    assert (
        main(
            [
                "ingest",
                "--database",
                "scratch_x",
                "--file",
                str(empty),
                "--provenance",
                str(prov_path),
            ]
        )
        == 1
    )
    assert "EMPTY" in capsys.readouterr().err
