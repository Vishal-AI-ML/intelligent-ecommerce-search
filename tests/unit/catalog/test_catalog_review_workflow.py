import csv
import json
from pathlib import Path

import pytest

from ecommerce_search.catalog.quality.findings import Finding, Severity
from ecommerce_search.catalog.review import (
    CSV_COLUMNS,
    HASHED_COLUMNS,
    REVIEWER_COLUMNS,
    SAMPLE_QUOTAS,
    ReviewError,
    ReviewSample,
    build_manifest,
    prepare_sample,
    read_review_csv,
    row_hash,
    select_sample,
    validate_import,
    verify_manifest,
    write_review_artifacts,
)

HANDLE = "TEST-ONLY-REVIEWER"


# ---- selection and manifest -------------------------------------------------------------


def test_sample_is_36_products_with_required_stratification(seed_catalog, seed_sample):
    by_id = {r.product_id: r for r in seed_catalog.records}
    ids = seed_sample.manifest["product_ids"]
    counts = {c.value: 0 for c in SAMPLE_QUOTAS}
    for pid in ids:
        counts[by_id[pid].category.value] += 1
    assert counts == {"laptop": 10, "phone": 10, "shoes": 8, "headphones": 8}
    assert len(ids) == len(set(ids)) == 36
    for category in SAMPLE_QUOTAS:  # every brand of every category is represented
        all_brands = {r.brand for r in seed_catalog.records if r.category is category}
        assert {by_id[p].brand for p in ids if by_id[p].category is category} == all_brands


def test_sample_is_deterministic_and_seed_dependent(seed_catalog, seed_report):
    a = select_sample(seed_catalog.records, seed_report.findings)
    assert a == select_sample(seed_catalog.records, seed_report.findings)
    assert a != select_sample(seed_catalog.records, seed_report.findings, seed="another-seed")


def test_prepare_sample_is_reproducible(seed_catalog, seed_provenance, seed_report, seed_sample):
    from ecommerce_search.ingestion.service import dataset_info

    again = prepare_sample(
        seed_catalog,
        seed_provenance,
        seed_report,
        dataset_info(seed_provenance, seed_catalog.checksum_sha256),
    )
    assert again.manifest == seed_sample.manifest and again.rows == seed_sample.rows


def test_warned_products_are_prioritised_when_findings_exist(seed_catalog, seed_report):
    flagged = [r.product_id for r in seed_catalog.records if r.category.value == "laptop"][:20]
    findings = [Finding("price_band", Severity.WARNING, "x", product_id=p) for p in flagged]
    ids = select_sample(seed_catalog.records, findings)
    assert len(set(ids) & set(flagged)) >= 5  # up to half the quota
    # the real seed produces no warnings, so its real sample is stratified by category/brand only
    assert not [f for f in seed_report.findings if f.severity is Severity.WARNING]


def test_sample_needs_enough_products(parsed):
    with pytest.raises(ReviewError):
        select_sample([parsed("laptop")], [])


def test_manifest_binds_every_product_content_and_row_hash(seed_catalog, seed_sample):
    manifest = seed_sample.manifest
    verify_manifest(manifest)
    assert "review_status" not in manifest and "initial_status" not in manifest
    contents = {line.record.product_id: line.content_sha256 for line in seed_catalog.lines}
    assert [p["product_id"] for p in manifest["products"]] == manifest["product_ids"]
    for entry in manifest["products"]:
        assert entry["content_sha256"] == contents[entry["product_id"]]
        assert len(entry["row_sha256"]) == 64
    assert manifest["review_batch_id"].startswith("synthetic-seed-v1-")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda m: {**m, "product_ids": m["product_ids"][:-1]},
        lambda m: {**m, "products": m["products"][:-1]},
        lambda m: {**m, "dataset_checksum_sha256": "0" * 64},
        lambda m: {**m, "review_batch_id": "synthetic-seed-v1-000000000000"},
        lambda m: {
            **m,
            "products": [{**m["products"][0], "content_sha256": "0" * 64}, *m["products"][1:]],
        },
        lambda m: {k: v for k, v in m.items() if k != "products"},
    ],
)
def test_manifest_tampering_is_detected(seed_sample, tamper):
    with pytest.raises(ReviewError):
        verify_manifest(tamper(seed_sample.manifest))


# ---- row hash ---------------------------------------------------------------------------


def test_row_hash_covers_every_non_review_column_and_ignores_reviewer_columns(seed_sample):
    row = seed_sample.rows[0]
    assert set(HASHED_COLUMNS) | set(REVIEWER_COLUMNS) | {
        "row_sha256",
        "sample_manifest_sha256",
    } == set(CSV_COLUMNS)
    base = row_hash(row)
    assert base == row["row_sha256"]
    for column in HASHED_COLUMNS:
        assert row_hash({**row, column: row[column] + "x"}) != base, column
    for column in REVIEWER_COLUMNS:
        assert row_hash({**row, column: "anything"}) == base, column


def test_row_hash_tolerates_spreadsheet_number_and_boolean_reformatting(seed_sample):
    row = seed_sample.rows[0]
    assert row["price"].endswith(".00")
    assert (
        row_hash({**row, "price": row["price"][:-3], "is_synthetic": "TRUE"}) == row["row_sha256"]
    )


# ---- artifacts: never overwrite review work (B3) ----------------------------------------


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def fill_csv(csv_path: Path, **cells):
    rows = read_review_csv(csv_path)
    rows[3].update(cells)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_second_generation_refuses_and_leaves_every_byte_unchanged(seed_sample, tmp_path):
    write_review_artifacts(tmp_path, seed_sample)
    before = snapshot(tmp_path)
    with pytest.raises(ReviewError, match="refusing to overwrite"):
        write_review_artifacts(tmp_path, seed_sample)
    assert snapshot(tmp_path) == before


def test_partial_existing_files_also_refuse(seed_sample, tmp_path):
    (tmp_path / "review_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewError, match="refusing to overwrite"):
        write_review_artifacts(tmp_path, seed_sample)
    assert snapshot(tmp_path) == {"review_manifest.json": b"{}"}


def test_force_replaces_only_a_provably_blank_sample(seed_sample, tmp_path):
    write_review_artifacts(tmp_path, seed_sample)
    (tmp_path / "REVIEW_INSTRUCTIONS.md").write_text("stale", encoding="utf-8")
    write_review_artifacts(tmp_path, seed_sample, force=True)
    assert (tmp_path / "REVIEW_INSTRUCTIONS.md").read_text(encoding="utf-8").startswith("# Catalog")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "cells",
    [
        {"verdict": "accept"},
        {"issue_fields": "title"},
        {"notes": "looks odd"},
        {"verdict": "unsure", "notes": "x"},
    ],
)
def test_force_never_overwrites_partial_or_complete_review_work(seed_sample, tmp_path, cells):
    write_review_artifacts(tmp_path, seed_sample)
    fill_csv(tmp_path / "review_sample.csv", **cells)
    before = snapshot(tmp_path)
    for force in (False, True):
        with pytest.raises(ReviewError):
            write_review_artifacts(tmp_path, seed_sample, force=force)
        assert snapshot(tmp_path) == before


def test_force_refuses_an_unreadable_csv_it_cannot_prove_blank(seed_sample, tmp_path):
    write_review_artifacts(tmp_path, seed_sample)
    (tmp_path / "review_sample.csv").write_bytes(b"\xff\xfe not utf-8 \x80")
    before = snapshot(tmp_path)
    with pytest.raises(ReviewError, match="cannot be proven blank"):
        write_review_artifacts(tmp_path, seed_sample, force=True)
    assert snapshot(tmp_path) == before


def test_generated_csv_has_blank_reviewer_cells_and_matching_hashes(seed_sample, tmp_path):
    csv_path, manifest_path, readme = write_review_artifacts(tmp_path, seed_sample)
    rows = read_review_csv(csv_path)
    assert len(rows) == 36 and tuple(rows[0]) == CSV_COLUMNS
    for row, entry in zip(rows, seed_sample.manifest["products"], strict=True):
        assert all(row[c] == "" for c in REVIEWER_COLUMNS)
        assert row["is_synthetic"] == "true" and row["raw_line"].startswith("{")
        assert row["row_sha256"] == entry["row_sha256"] == row_hash(row)
        assert row["sample_manifest_sha256"] == seed_sample.manifest["manifest_sha256"]
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == seed_sample.manifest
    text = readme.read_text(encoding="utf-8")
    for needle in ("Back up", "CSV UTF-8", "never put an email", "audit trail", "--database"):
        assert needle.lower() in text.lower(), needle


# ---- import guards and content binding (B2) ---------------------------------------------


@pytest.fixture
def filled(seed_sample):
    rows = [{**r, "verdict": "accept"} for r in seed_sample.rows]
    return seed_sample.manifest, rows


def call(manifest, rows, reviewer=HANDLE, confirmed=True):
    return validate_import(manifest=manifest, rows=rows, reviewer=reviewer, confirmed=confirmed)


def test_valid_completed_review_is_accepted_in_manifest_order(filled):
    manifest, rows = filled
    accepted = call(manifest, list(reversed(rows)))
    assert [a.product_id for a in accepted] == manifest["product_ids"]
    assert {a.verdict for a in accepted} == {"accept"}
    assert accepted[0].product_content_sha256 == manifest["products"][0]["content_sha256"]


def test_editing_reviewer_fields_is_allowed_when_valid(filled):
    manifest, rows = filled
    rows[0].update(verdict="needs_correction", issue_fields="price, title", notes="Looks off.")
    accepted = call(manifest, rows)
    assert accepted[0].verdict == "needs_correction" and accepted[0].issue_fields == "price, title"


def test_guard_clauses(filled, seed_sample):
    manifest, rows = filled
    with pytest.raises(ReviewError, match="confirm"):
        call(manifest, rows, confirmed=False)
    for handle in (None, "", "   ", "someone@example.com", "x" * 65, "-leading"):
        with pytest.raises(ReviewError, match="handle|reviewer"):
            call(manifest, rows, reviewer=handle)
    with pytest.raises(ReviewError, match="verdict is blank"):
        call(manifest, list(seed_sample.rows))  # the untouched sample has no human verdicts
    partial = [dict(r) for r in rows]
    partial[5]["verdict"] = " "
    with pytest.raises(ReviewError, match="verdict is blank"):
        call(manifest, partial)
    with pytest.raises(ReviewError, match="invalid verdict"):
        call(manifest, [{**r, "verdict": "approved!"} for r in rows])
    with pytest.raises(ReviewError, match="do not match"):
        call(manifest, rows[:-1])
    with pytest.raises(ReviewError, match="duplicated"):
        call(manifest, [*rows[:-1], rows[0]])
    with pytest.raises(ReviewError, match="different manifest"):
        call(manifest, [{**r, "sample_manifest_sha256": "0" * 64} for r in rows])


@pytest.mark.parametrize("column", [c for c in HASHED_COLUMNS if c != "is_synthetic"])
def test_editing_any_non_review_field_fails_import(filled, column):
    manifest, rows = filled
    rows[7][column] = rows[7][column] + " EDITED"
    # product_id is caught by the id-match guard before the row hash is reached
    with pytest.raises(ReviewError, match="non-review column was modified|do not match"):
        call(manifest, rows)


def test_editing_title_price_attributes_raw_line_and_hash_columns_fail(filled):
    manifest, rows = filled
    for column, value in (
        ("title", "TAMPERED"),
        ("price", "1"),
        ("attributes", "{}"),
        ("raw_line", "{}"),
        ("row_sha256", "0" * 64),
    ):
        edited = [dict(r) for r in rows]
        edited[0][column] = value
        with pytest.raises(ReviewError, match="non-review column was modified"):
            call(manifest, edited)


def test_free_text_privacy_guards(filled):
    manifest, rows = filled
    rows[0]["notes"] = "contact me at someone@example.com"
    with pytest.raises(ReviewError, match="email"):
        call(manifest, rows)
    rows[0]["notes"] = "x" * 1001
    with pytest.raises(ReviewError, match="longer"):
        call(manifest, rows)


def test_build_manifest_is_deterministic():
    products = [{"product_id": "A", "content_sha256": "a" * 64, "row_sha256": "b" * 64}]
    dataset = {
        "dataset_id": "d",
        "dataset_version": "1",
        "checksum_sha256": "c" * 64,
        "is_synthetic": True,
    }
    assert build_manifest(dataset=dataset, products=products) == build_manifest(
        dataset=dataset, products=products
    )


def test_review_sample_type_is_immutable_value_object(seed_sample):
    assert isinstance(seed_sample, ReviewSample)


# ---- D. import-time manifest authenticity ------------------------------------------------


def seal(manifest):
    """Make an edited manifest internally self-consistent again (digest + batch id)."""
    from ecommerce_search.catalog.manifest import batch_id, manifest_digest

    manifest = dict(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    manifest["review_batch_id"] = batch_id(
        manifest["dataset_id"], manifest["dataset_version"], manifest["manifest_sha256"]
    )
    return manifest


def test_manifest_carries_the_full_historical_sampling_context(seed_sample):
    manifest = seed_sample.manifest
    for key in (
        "manifest_schema_version",
        "sample_seed",
        "selection_version",
        "selection_method",
        "rules_version",
        "taxonomy_version",
        "transform_version",
        "quotas",
        "dataset_id",
        "dataset_version",
        "dataset_checksum_sha256",
    ):
        assert manifest[key] not in (None, "", {}), key
    assert manifest["manifest_schema_version"] == 3 and manifest["selection_version"] == "1"
    assert sum(manifest["quotas"].values()) == len(manifest["products"]) == 36


def test_the_genuine_generated_manifest_is_accepted(seed_sample):
    from ecommerce_search.catalog.review import require_genuine_manifest

    require_genuine_manifest(seed_sample.manifest, seed_sample.manifest)
    require_genuine_manifest(json.loads(json.dumps(seed_sample.manifest)), seed_sample.manifest)


def _forgeries(manifest):
    products = [dict(p) for p in manifest["products"]]
    swapped = [dict(p) for p in products]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    renamed = [dict(p) for p in products]
    renamed[0]["product_id"] = "SYN-LAP-9999"
    rehashed = [dict(p) for p in products]
    rehashed[0]["content_sha256"] = "0" * 64
    row_hashed = [dict(p) for p in products]
    row_hashed[0]["row_sha256"] = "0" * 64
    quotas = {**manifest["quotas"], "laptop": 11, "phone": 9}

    def with_products(items):
        return {**manifest, "products": items, "product_ids": [p["product_id"] for p in items]}

    return {
        "edited quotas": {**manifest, "quotas": quotas},
        "product order": with_products(swapped),
        "product ids": with_products(renamed),
        "product content hashes": with_products(rehashed),
        "row hashes": with_products(row_hashed),
        "selection method": {**manifest, "selection_method": "hand written"},
        "selection version": {**manifest, "selection_version": "2"},
        "sample seed": {**manifest, "sample_seed": "another-seed"},
        "rules version": {**manifest, "rules_version": "99"},
        "dataset checksum": {**manifest, "dataset_checksum_sha256": "3" * 64},
    }


def test_hand_crafted_self_consistent_manifests_are_rejected(seed_sample):
    from ecommerce_search.catalog.review import require_genuine_manifest, verify_manifest

    for label, forged in _forgeries(seed_sample.manifest).items():
        sealed = seal(forged)
        verify_manifest(sealed)  # internally self-consistent: this alone would accept it
        with pytest.raises(ReviewError, match="not the review sample generated"):
            require_genuine_manifest(sealed, seed_sample.manifest)
        assert label  # (label shown on failure via pytest's locals)


def test_an_unsealed_forgery_is_rejected_by_the_digest(seed_sample):
    from ecommerce_search.catalog.review import require_genuine_manifest

    forged = {
        **seed_sample.manifest,
        "quotas": {**seed_sample.manifest["quotas"], "laptop": 11, "phone": 9},
    }
    with pytest.raises(ReviewError, match="manifest_sha256"):
        require_genuine_manifest(forged, seed_sample.manifest)
    with pytest.raises(ReviewError, match="differs in"):
        require_genuine_manifest(seal(forged), seed_sample.manifest)
