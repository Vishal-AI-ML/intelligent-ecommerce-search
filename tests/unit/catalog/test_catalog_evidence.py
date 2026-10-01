import copy
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecommerce_search.catalog import review as review_module
from ecommerce_search.catalog.cli import main
from ecommerce_search.catalog.evidence import (
    EvidenceError,
    build_evidence,
    canonical_hash,
    evaluate_rows,
    evidence_filename,
    find_evidence,
    load_evidence,
    validate_handle,
    verify_evidence,
    write_evidence,
)
from ecommerce_search.catalog.manifest import batch_id, manifest_digest
from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.review import build_manifest, prepare_sample, validate_import

HANDLE = "TEST-ONLY-REVIEWER"
WHEN = datetime(2026, 10, 1, 12, 30, 5, tzinfo=UTC)
SEED = Path(__file__).resolve().parents[3] / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = SEED.with_name("catalog_seed_v1.provenance.json")
BASE = ["--file", str(SEED), "--provenance", str(PROVENANCE)]


@pytest.fixture
def evidence(seed_sample):
    """Test-only evidence built in memory from verdicts the TEST supplies (never a real review)."""
    rows = [{**r, "verdict": "accept"} for r in seed_sample.rows]
    rows[1].update(verdict="needs_correction", issue_fields="price", notes="test note")
    reviews = validate_import(
        manifest=seed_sample.manifest, rows=rows, reviewer=HANDLE, confirmed=True
    )
    return build_evidence(seed_sample.manifest, reviews, reviewer_handle=HANDLE, recorded_at=WHEN)


def rehash(body):
    body = {k: v for k, v in body.items() if k != "records_sha256"}
    return {**body, "records_sha256": canonical_hash(body)}


def forge(evidence, mutate_manifest, mutate_reviews=None):
    """Edit the embedded manifest, then recompute EVERY hash (manifest digest, batch id, the
    top-level copies and the evidence self-hash). Only the committed seed can still object."""
    body = copy.deepcopy(evidence)
    mutate_manifest(body["manifest"])
    manifest = body["manifest"]
    manifest["manifest_sha256"] = manifest_digest(manifest)
    manifest["review_batch_id"] = batch_id(
        manifest["dataset_id"], manifest["dataset_version"], manifest["manifest_sha256"]
    )
    body["manifest_sha256"] = manifest["manifest_sha256"]
    body["review_batch_id"] = manifest["review_batch_id"]
    body["dataset_id"] = manifest["dataset_id"]
    body["dataset_version"] = manifest["dataset_version"]
    body["dataset_checksum_sha256"] = manifest["dataset_checksum_sha256"]
    if mutate_reviews:
        mutate_reviews(body["reviews"])
    return rehash(body)


def write(tmp_path, body, manifest=None):
    return write_evidence(tmp_path, body, manifest or body["manifest"])


def load_from(tmp_path, body):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return load_evidence(path)


# ---- format and privacy -----------------------------------------------------------------


def test_evidence_is_self_contained_and_has_every_required_field(
    evidence, seed_sample, seed_facts, tmp_path
):
    for key in (
        "schema_version",
        "review_batch_id",
        "manifest_sha256",
        "dataset_id",
        "dataset_version",
        "dataset_checksum_sha256",
        "manifest",
        "reviewer_handle",
        "recorded_at",
        "product_count",
        "reviews",
        "records_sha256",
    ):
        assert key in evidence, key
    assert evidence["schema_version"] == 2
    assert evidence["manifest"] == seed_sample.manifest  # the exact reviewed manifest
    assert evidence["product_count"] == 36 == len(evidence["reviews"])
    assert evidence["recorded_at"] == "2026-10-01T12:30:05Z"
    manifest = evidence["manifest"]
    for key in (
        "manifest_schema_version",
        "review_batch_id",
        "dataset_id",
        "dataset_version",
        "dataset_checksum_sha256",
        "sample_seed",
        "selection_version",
        "selection_method",
        "quotas",
        "rules_version",
        "taxonomy_version",
        "transform_version",
    ):
        assert key in manifest, key
    assert set(manifest["products"][0]) == {"product_id", "content_sha256", "row_sha256"}
    assert set(evidence["reviews"][0]) == {
        "product_id",
        "product_content_sha256",
        "verdict",
        "issue_fields",
        "notes",
    }
    assert [r["product_id"] for r in evidence["reviews"]] == manifest["product_ids"]
    path = write(tmp_path, evidence)
    assert path.name == evidence_filename(seed_sample.manifest)
    assert re.fullmatch(r"catalog_review_synthetic-seed_v1_[0-9a-f]{12}\.json", path.name)
    result = verify_evidence(load_evidence(path), seed_facts, seed_sample.manifest)
    assert result.matches_current_sampling_rules is True
    assert find_evidence(tmp_path, "synthetic-seed", "1") == [path]


def test_evidence_is_privacy_conscious(evidence, tmp_path):
    text = write(tmp_path, evidence).read_text(encoding="utf-8")
    assert not re.search(r"\S+@\S+\.\S+", text)
    assert str(tmp_path) not in text and "\\Users\\" not in text and "/Users/" not in text
    assert "audit trail" in text and "not cryptographic proof" in text
    assert "never an email" in text


def test_evidence_is_never_overwritten(evidence, tmp_path):
    path = write(tmp_path, evidence)
    before = path.read_bytes()
    with pytest.raises(EvidenceError, match="never overwritten"):
        write(tmp_path, rehash({**evidence, "reviewer_handle": "someone-else"}))
    assert path.read_bytes() == before


@pytest.mark.parametrize("handle", ["a@b.co", "", "   ", "x" * 65, "-x", "bad/char"])
def test_reviewer_handle_validation(handle):
    with pytest.raises(EvidenceError):
        validate_handle(handle)
    assert validate_handle("  Reviewer_1.a-b  ") == "Reviewer_1.a-b"


def test_canonical_self_hash_is_deterministic(evidence, seed_sample):
    again = build_evidence(
        seed_sample.manifest,
        [_row(r) for r in evidence["reviews"]],
        reviewer_handle=HANDLE,
        recorded_at=WHEN,
    )
    assert again["records_sha256"] == evidence["records_sha256"]
    body = {k: v for k, v in evidence.items() if k != "records_sha256"}
    assert canonical_hash(body) == canonical_hash(json.loads(json.dumps(body)))
    shuffled = dict(reversed(list(body.items())))  # key order never matters
    assert canonical_hash(shuffled) == evidence["records_sha256"]


def _row(review):
    class Row:
        pass

    row = Row()
    row.__dict__.update(review)
    return row


# ---- A. historical durability: current code/constants never invalidate old evidence -------


def test_changing_current_sampling_constants_never_invalidates_historical_evidence(
    evidence,
    seed_sample,
    seed_catalog,
    seed_provenance,
    seed_report,
    seed_facts,
    monkeypatch,
    tmp_path,
):
    from ecommerce_search.ingestion.service import dataset_info

    info = dataset_info(seed_provenance, seed_catalog.checksum_sha256)
    path = write(tmp_path, evidence)

    def current_after(change):
        with monkeypatch.context() as patch:
            change(patch)
            return prepare_sample(seed_catalog, seed_provenance, seed_report, info).manifest

    changes = {
        "rules version": lambda m: m.setattr(qp, "RULES_VERSION", "99"),
        "selection method wording": lambda m: m.setattr(
            review_module, "SELECTION_METHOD", "reworded"
        ),
        "selection version": lambda m: m.setattr(review_module, "SELECTION_VERSION", "2"),
        "quotas": lambda m: m.setattr(
            review_module,
            "SAMPLE_QUOTAS",
            {**review_module.SAMPLE_QUOTAS, **{next(iter(review_module.SAMPLE_QUOTAS)): 99}},
        ),
        "taxonomy version": lambda m: m.setattr(review_module, "TAXONOMY_VERSION", "99"),
        "transform version": lambda m: m.setattr(review_module, "TRANSFORM_VERSION", "99"),
        "row rendering": lambda m: m.setattr(review_module, "row_hash", lambda row: "f" * 64),
    }
    for label, change in changes.items():
        current = current_after(change)
        assert current != seed_sample.manifest, label  # the current rules really differ now
        result = verify_evidence(load_evidence(path), seed_facts, current)
        assert result.matches_current_sampling_rules is False, label  # informational only
        assert result.manifest == seed_sample.manifest, label  # historical values untouched

    other_seed = build_manifest(
        dataset=info, products=seed_sample.manifest["products"], seed="another-seed"
    )
    assert (
        verify_evidence(load_evidence(path), seed_facts, other_seed).matches_current_sampling_rules
        is False
    )
    assert (
        verify_evidence(load_evidence(path), seed_facts, None).matches_current_sampling_rules
        is None
    )


def test_status_stays_recorded_through_the_cli_after_current_rules_change(
    evidence, monkeypatch, tmp_path, capsys
):
    write(tmp_path, evidence)
    monkeypatch.setattr(qp, "RULES_VERSION", "99")
    monkeypatch.setattr(review_module, "SELECTION_METHOD", "reworded after the review")
    assert (
        main(["review-status", *BASE, "--evidence-dir", str(tmp_path), "--require-complete"]) == 0
    )
    out = capsys.readouterr().out
    assert "RECORDED (36/36" in out
    assert "matches_current_sampling_rules: false" in out and "informational" in out
    assert main(["review-verify", *BASE, "--evidence-dir", str(tmp_path)]) == 0
    assert "matches_current_sampling_rules: false" in capsys.readouterr().out


# ---- B. committed seed and content safety stay strict ------------------------------------


def verify_fails(body, tmp_path, seed_facts, message):
    with pytest.raises(EvidenceError, match=message):
        verify_evidence(load_from(tmp_path, body), seed_facts)


def test_seed_content_and_dataset_mismatches_fail(evidence, seed_facts, tmp_path):
    pid = evidence["reviews"][0]["product_id"]
    body = load_from(tmp_path, evidence)
    changed = replace(seed_facts, content_hashes={**seed_facts.content_hashes, pid: "0" * 64})
    with pytest.raises(
        EvidenceError, match=f"{pid}: content hash does not match the committed seed"
    ):
        verify_evidence(body, changed)
    with pytest.raises(
        EvidenceError, match="dataset_checksum_sha256 does not match the committed seed"
    ):
        verify_evidence(body, replace(seed_facts, dataset_checksum_sha256="1" * 64))
    with pytest.raises(EvidenceError, match="dataset_version does not match the committed seed"):
        verify_evidence(body, replace(seed_facts, dataset_version="2"))
    gone = {k: v for k, v in seed_facts.content_hashes.items() if k != pid}
    with pytest.raises(EvidenceError, match="does not match the committed seed"):
        verify_evidence(body, replace(seed_facts, content_hashes=gone))


def test_edits_without_recomputing_hashes_are_detected(evidence, seed_facts, tmp_path):
    def stale(mutate, message):
        body = copy.deepcopy(evidence)
        mutate(body)
        verify_fails(body, tmp_path, seed_facts, message)

    stale(lambda b: b["manifest"]["products"][0].update(content_sha256="0" * 64), "records_sha256")
    stale(lambda b: b["manifest"]["products"][0].update(row_sha256="0" * 64), "records_sha256")
    stale(lambda b: b["manifest"]["quotas"].update(laptop=99), "records_sha256")
    stale(lambda b: b["reviews"][2].update(verdict="unsure"), "records_sha256")
    stale(lambda b: b.update(reviewer_handle="SOMEONE-ELSE"), "records_sha256")
    stale(lambda b: b["reviews"][2].update(notes="edited note"), "records_sha256")
    stale(lambda b: b.update(dataset_checksum_sha256="0" * 64), "records_sha256")


def test_embedded_manifest_edits_with_only_the_self_hash_recomputed_fail(
    evidence, seed_facts, tmp_path
):
    for mutate in (
        lambda m: m["products"][0].update(row_sha256="0" * 64),
        lambda m: m["quotas"].update(laptop=99),
        lambda m: m.update(selection_method="altered"),
        lambda m: m["products"][0].update(content_sha256="0" * 64),
    ):
        body = copy.deepcopy(evidence)
        mutate(body["manifest"])
        verify_fails(rehash(body), tmp_path, seed_facts, "embedded manifest is invalid")


def test_forging_the_embedded_manifest_with_every_hash_recomputed_still_fails_the_seed(
    evidence, seed_facts, tmp_path
):
    first = evidence["reviews"][0]["product_id"]

    def new_content(m):
        m["products"][0]["content_sha256"] = "0" * 64

    def sync_review(reviews):
        reviews[0]["product_content_sha256"] = "0" * 64

    verify_fails(
        forge(evidence, new_content, sync_review),
        tmp_path,
        seed_facts,
        f"{first}: content hash does not match the committed seed",
    )

    def other_dataset(m):
        m["dataset_checksum_sha256"] = "2" * 64

    verify_fails(
        forge(evidence, other_dataset),
        tmp_path,
        seed_facts,
        "dataset_checksum_sha256 does not match the committed seed",
    )

    def unknown_product(m):
        m["products"][0]["product_id"] = "SYN-LAP-9999"
        m["product_ids"][0] = "SYN-LAP-9999"

    def rename_review(reviews):
        reviews[0]["product_id"] = "SYN-LAP-9999"

    verify_fails(
        forge(evidence, unknown_product, rename_review),
        tmp_path,
        seed_facts,
        "SYN-LAP-9999: content hash does not match the committed seed",
    )


def test_a_forged_row_hash_cannot_be_checked_against_current_code_but_is_flagged_informationally(
    evidence, seed_facts, seed_sample, tmp_path
):
    # Row hashes are historical rendering facts: they are covered by the manifest digest and the
    # self-hash, and are NOT re-derived from current rendering code (that would couple evidence
    # to code). A forger recomputing every hash therefore cannot be caught by them, but the
    # informational flag shows the reviewed manifest differs from what today's code produces.
    forged = forge(evidence, lambda m: m["products"][0].update(row_sha256="0" * 64))
    result = verify_evidence(load_from(tmp_path, forged), seed_facts, seed_sample.manifest)
    assert result.matches_current_sampling_rules is False


def test_missing_extra_duplicated_and_reordered_reviews_are_detected(
    evidence, seed_facts, tmp_path
):
    def check(mutate, message, **kw):
        body = copy.deepcopy(evidence)
        mutate(body["reviews"])
        verify_fails(rehash(body), tmp_path, seed_facts, message)

    check(lambda r: r.pop(), "missing products")
    check(lambda r: r.append({**r[0], "product_id": "SYN-LAP-9999"}), "unexpected products")
    check(lambda r: r.__setitem__(5, dict(r[0])), "duplicated")
    check(lambda r: r.reverse(), "embedded manifest order")
    body = copy.deepcopy(evidence)
    body["product_count"] = 35
    verify_fails(rehash(body), tmp_path, seed_facts, "product_count")
    body = copy.deepcopy(evidence)
    body["reviews"][0]["product_content_sha256"] = "0" * 64
    verify_fails(rehash(body), tmp_path, seed_facts, "does not match the embedded manifest")


def test_top_level_copies_must_match_the_embedded_manifest(evidence, seed_facts, tmp_path):
    for key, value in (
        ("manifest_sha256", "0" * 64),
        ("review_batch_id", "x-v1-000000000000"),
        ("dataset_id", "other"),
        ("dataset_version", "2"),
        ("dataset_checksum_sha256", "0" * 64),
    ):
        body = copy.deepcopy(evidence)
        body[key] = value
        verify_fails(rehash(body), tmp_path, seed_facts, f"{key} does not match")


def test_embedded_manifest_must_have_the_minimum_products_and_a_known_schema(
    evidence, seed_facts, tmp_path
):
    def too_few(m):
        m["product_ids"] = m["product_ids"][:5]
        m["products"] = m["products"][:5]
        m["quotas"] = {"laptop": 5}

    verify_fails(
        forge(evidence, too_few, lambda r: r.__delitem__(slice(5, None))),
        tmp_path,
        seed_facts,
        "at least 30",
    )
    verify_fails(
        forge(evidence, lambda m: m.update(manifest_schema_version=99)),
        tmp_path,
        seed_facts,
        "unsupported manifest schema",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["reviews"][0].update(verdict="approved"),
        lambda b: b["reviews"][0].update(surprise=1),
        lambda b: b.update(surprise=1),
        lambda b: b.update(reviewer_handle="a@b.co"),
        lambda b: b.update(schema_version=1),
        lambda b: b.pop("reviews"),
        lambda b: b.pop("manifest"),
        lambda b: b.update(recorded_at="2026-10-01 12:00:00"),
    ],
)
def test_malformed_evidence_is_rejected_on_load(evidence, tmp_path, mutate):
    body = copy.deepcopy(evidence)
    mutate(body)
    with pytest.raises(EvidenceError, match="malformed"):
        load_from(tmp_path, rehash(body))


def test_unreadable_evidence_files(tmp_path):
    with pytest.raises(EvidenceError, match="does not exist"):
        load_evidence(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(EvidenceError, match="not valid JSON"):
        load_evidence(bad)
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(EvidenceError, match="UTF-8"):
        load_evidence(bad)


# ---- manifest-exact status from recorded rows (I7) --------------------------------------


@pytest.fixture
def good_rows(seed_sample):
    m = seed_sample.manifest
    return [
        {
            "product_id": p["product_id"],
            "verdict": "accept",
            "reviewer": HANDLE,
            "manifest_sha256": m["manifest_sha256"],
            "dataset_id": m["dataset_id"],
            "dataset_version": m["dataset_version"],
            "dataset_checksum_sha256": m["dataset_checksum_sha256"],
            "product_content_sha256": p["content_sha256"],
        }
        for p in m["products"]
    ]


def test_status_recorded_only_for_an_exact_complete_single_reviewer_batch(seed_sample, good_rows):
    status = evaluate_rows(seed_sample.manifest, good_rows)
    assert (status.state, status.reviewed, status.total, status.reviewers) == (
        "RECORDED",
        36,
        36,
        (HANDLE,),
    )


def test_status_pending_when_empty_or_partial(seed_sample, good_rows):
    assert evaluate_rows(seed_sample.manifest, []).state == "PENDING"
    partial = evaluate_rows(seed_sample.manifest, good_rows[:20])
    assert (partial.state, partial.reviewed) == ("PENDING", 20)


@pytest.mark.parametrize(
    ("label", "mutate", "expect"),
    [
        ("second reviewer", lambda r: r[0].update(reviewer="OTHER"), "different reviewers"),
        (
            "foreign product",
            lambda r: r.append({**r[0], "product_id": "SYN-LAP-9999"}),
            "outside the sample",
        ),
        ("duplicate outcome", lambda r: r.append(dict(r[3])), "more than one outcome"),
        ("wrong manifest hash", lambda r: r[0].update(manifest_sha256="0" * 64), "manifest_sha256"),
        (
            "wrong dataset checksum",
            lambda r: r[0].update(dataset_checksum_sha256="0" * 64),
            "dataset/checksum",
        ),
        ("wrong dataset version", lambda r: r[0].update(dataset_version="2"), "dataset/checksum"),
        (
            "wrong content hash",
            lambda r: r[0].update(product_content_sha256="0" * 64),
            "product_content_sha256",
        ),
        ("invalid verdict", lambda r: r[0].update(verdict="approved"), "invalid verdict"),
    ],
)
def test_status_is_error_for_inconsistent_rows(seed_sample, good_rows, label, mutate, expect):
    rows = copy.deepcopy(good_rows)
    mutate(rows)
    status = evaluate_rows(seed_sample.manifest, rows)
    assert status.state == "ERROR", label
    assert any(expect in p for p in status.problems), (label, status.problems)


def test_a_foreign_product_cannot_complete_a_partial_batch(seed_sample, good_rows):
    rows = copy.deepcopy(good_rows[:35])
    rows.append({**good_rows[0], "product_id": "SYN-LAP-9999"})
    assert evaluate_rows(seed_sample.manifest, rows).state == "ERROR"


# ---- offline CLI status / verify --------------------------------------------------------


def test_cli_status_is_pending_without_evidence_and_never_needs_a_database(tmp_path, capsys):
    assert main(["review-status", *BASE, "--evidence-dir", str(tmp_path)]) == 0
    assert "PENDING (0/36" in capsys.readouterr().out
    assert (
        main(["review-status", *BASE, "--evidence-dir", str(tmp_path), "--require-complete"]) == 1
    )


def test_cli_status_recorded_36_of_36_from_valid_evidence(evidence, tmp_path, capsys):
    write(tmp_path, evidence)
    assert (
        main(["review-status", *BASE, "--evidence-dir", str(tmp_path), "--require-complete"]) == 0
    )
    out = capsys.readouterr().out
    assert "RECORDED (36/36" in out and "matches_current_sampling_rules: true" in out
    assert main(["review-verify", *BASE, "--evidence-dir", str(tmp_path)]) == 0
    assert "36 reviews" in capsys.readouterr().out


def test_cli_reports_error_for_invalid_or_ambiguous_evidence(evidence, tmp_path, capsys):
    broken = copy.deepcopy(evidence)
    broken["reviews"][0]["verdict"] = "unsure"  # self-hash no longer matches
    path = tmp_path / "catalog_review_x.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["review-status", *BASE, "--evidence", str(path)]) == 1
    assert "ERROR" in capsys.readouterr().out
    assert main(["review-verify", *BASE, "--evidence", str(path)]) == 1
    assert "records_sha256" in capsys.readouterr().err

    folder = tmp_path / "two"
    write(folder, evidence)
    (folder / "catalog_review_synthetic-seed_v1_000000000000.json").write_text(
        "{}", encoding="utf-8"
    )
    assert main(["review-status", *BASE, "--evidence-dir", str(folder)]) == 1
    assert "more than one evidence file" in capsys.readouterr().out


def test_cli_verify_without_evidence_fails(tmp_path, capsys):
    assert main(["review-verify", *BASE, "--evidence-dir", str(tmp_path)]) == 1
    assert "no evidence file" in capsys.readouterr().err
