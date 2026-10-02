"""`dense_embedding_consistency` and the `dense_index` report section on scratch databases."""

import pytest
from catalog_support import PROVENANCE, SEED
from dense_support import SPEC, change_product, embed_fake, sql
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ecommerce_search.catalog.audit import audit_database
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.search.dense_audit import CHECK_NAME, NO_TABLE_NOTE

pytestmark = pytest.mark.integration

PID = "SYN-HDP-0002"


@pytest.fixture
def catalog(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    return migrated_engine


def audit(engine):
    with Session(engine) as session:
        report = audit_database(session)
    check = {c["name"]: c for c in report.checks}[CHECK_NAME]
    findings = [f for f in report.findings if f.check == CHECK_NAME]
    return report, check, findings, report.to_dict()["dense_index"]


CLEAN = {
    "applicable": True,
    "model_id": SPEC.model_id,
    "model_revision": SPEC.revision,
    "dimension": 384,
    "normalized": True,
    "max_seq_length": 256,
    "embedding_text_version": "1",
    "config_sha256": SPEC.config_sha256(),
    "distance_metric": "cosine",
    "vector_index": "none (exact scan)",
    "products": 240,
    "products_audited": 240,
    "embeddings": 240,
    "missing": 0,
    "stale_model": 0,
    "stale_text_version": 0,
    "stale_config": 0,
    "source_hash_mismatch": 0,
    "text_hash_mismatch": 0,
    "dimension_mismatch": 0,
    "normalization_mismatch": 0,
    "invalid_vector": 0,
}


def test_unembedded_catalog_is_a_warning_not_an_error(catalog):
    report, check, findings, section = audit(catalog)
    assert check["status"] == "warn" and check["evaluated"] == 240
    assert len(findings) == 240 and {f.severity.value for f in findings} == {"warning"}
    assert report.error_count == 0
    assert section == {**CLEAN, "embeddings": 0, "missing": 240}
    # check 10 (lexical index leakage) keeps its Milestone 3 meaning
    assert {c["name"]: c for c in report.checks}["embedding_search_text_leakage"][
        "status"
    ] == "pass"


def test_current_embeddings_pass_with_exact_metadata(catalog):
    embed_fake(catalog)
    report, check, findings, section = audit(catalog)
    assert (check["status"], check["findings"]) == ("pass", 0)
    assert section == CLEAN and report.error_count == 0


def test_stale_content_is_a_warning(catalog):
    embed_fake(catalog)
    change_product(catalog, PID, "Changed headphones title")
    report, check, findings, section = audit(catalog)
    # (the direct SQL edit also trips the M2 raw/normalized consistency check: not dense errors)
    assert check["status"] == "warn"
    assert [(f.product_id, f.severity.value) for f in findings] == [(PID, "warning")]
    assert section["source_hash_mismatch"] == 1 and section["text_hash_mismatch"] == 1


def test_stale_model_revision_is_a_warning(catalog):
    embed_fake(catalog)
    sql(
        catalog,
        "UPDATE product_embeddings SET model_revision = :r WHERE product_id = :p",
        r="f" * 40,
        p=PID,
    )
    _, check, findings, section = audit(catalog)
    assert check["status"] == "warn" and section["stale_model"] == 1
    assert {f.severity.value for f in findings} == {"warning"}


def test_tampered_text_hash_on_a_current_row_is_an_error(catalog):
    embed_fake(catalog)
    sql(
        catalog,
        "UPDATE product_embeddings SET embedding_text_sha256 = :s WHERE product_id = :p",
        s="0" * 64,
        p=PID,
    )
    report, check, findings, section = audit(catalog)
    assert check["status"] == "fail" and report.error_count == 1
    assert findings[0].product_id == PID and "whitelisted embedding text" in findings[0].message
    assert section["text_hash_mismatch"] == 1


def test_normalization_flag_mismatch_is_an_error(catalog):
    embed_fake(catalog)
    sql(catalog, "UPDATE product_embeddings SET normalized = false WHERE product_id = :p", p=PID)
    report, check, _, section = audit(catalog)
    assert check["status"] == "fail" and section["normalization_mismatch"] == 1


def test_invalid_vector_is_an_error_even_if_constraints_were_bypassed(catalog):
    embed_fake(catalog)
    sql(
        catalog,
        "ALTER TABLE product_embeddings "
        "DROP CONSTRAINT ck_product_embeddings_embedding_nonzero, "
        "DROP CONSTRAINT ck_product_embeddings_embedding_normalized",
    )
    sql(
        catalog,
        "UPDATE product_embeddings SET embedding = CAST(:v AS vector) WHERE product_id = :p",
        v="[" + ",".join(["0"] * 384) + "]",
        p=PID,
    )
    report, check, findings, section = audit(catalog)
    assert check["status"] == "fail" and section["invalid_vector"] == 1
    assert any("zero, non-finite or not unit" in f.message for f in findings)


def test_database_below_0004_reports_not_applicable(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database, "0003")
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        ingest_file(engine, SEED, PROVENANCE)
        report, check, findings, section = audit(engine)
    finally:
        engine.dispose()
    assert check["status"] == "not_applicable" and findings == []
    assert section == {"applicable": False, "reason": NO_TABLE_NOTE}
    assert report.error_count == 0
