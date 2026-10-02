"""Embedding generation, healing, atomicity and concurrency on scratch databases (fake model)."""

import threading
import time
from dataclasses import replace

import pytest
from catalog_support import PROVENANCE, SEED, edit_line, make_variant, seed_lines
from dense_support import (
    SPEC,
    change_product,
    dense_ids,
    embed_fake,
    embeddings,
    sql,
    status,
)
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError

from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.ingestion.service import ingest_file, ingest_lock_key
from ecommerce_search.search import cli as search_cli
from ecommerce_search.search import dense_indexing as di
from ecommerce_search.search.indexing import search_lock_key
from fake_embedder import FakeEmbedder

pytestmark = pytest.mark.integration

PID = "SYN-SHO-0001"


@pytest.fixture
def catalog(migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    return migrated_engine


def test_ingestion_writes_no_embeddings(catalog):
    assert embeddings(catalog) == {}
    current = status(catalog)
    assert current.counts["missing"] == 240 and not current.current


def test_first_embed_creates_every_row_and_status_is_current(catalog):
    result = embed_fake(catalog)
    assert (result.inserted, result.updated, result.unchanged) == (240, 0, 0)
    assert result.reasons["missing"] == 240
    rows = embeddings(catalog)
    assert len(rows) == 240
    row = rows[PID]
    assert (row["model_id"], row["model_revision"]) == (SPEC.model_id, SPEC.revision)
    assert row["embedding_config_sha256"] == SPEC.config_sha256()
    current = status(catalog)
    assert current.current and current.total_embeddings == current.total_products == 240


def test_rerun_is_idempotent_and_keeps_embedded_at(catalog):
    embed_fake(catalog)
    before = embeddings(catalog)
    fake = FakeEmbedder()
    result = embed_fake(catalog, fake)
    assert (result.inserted, result.updated, result.unchanged) == (0, 0, 240)
    assert fake.encoded == [] and embeddings(catalog) == before


def test_rebuild_all_rewrites_every_row(catalog):
    embed_fake(catalog)
    before = embeddings(catalog)
    result = embed_fake(catalog, rebuild_all=True)
    assert result.updated == 240
    after = embeddings(catalog)
    assert all(after[p]["embedded_at"] > before[p]["embedded_at"] for p in before)


def test_changed_product_is_excluded_from_dense_results_until_healed(catalog):
    embed_fake(catalog)
    query = "Strato running shoes"
    assert PID in dense_ids(catalog, query)
    old_vector = embeddings(catalog)[PID]["vector"]
    change_product(catalog, PID, "Nike Strato Running Shoes for Men - Blue Model SH101")
    stale = status(catalog)
    assert stale.counts["source_hash_mismatch"] == 1 and stale.counts["text_hash_mismatch"] == 1
    assert not stale.current
    # the old vector must never rank the updated product
    assert PID not in dense_ids(catalog, query)
    assert len(dense_ids(catalog, query)) == 239
    result = embed_fake(catalog)
    assert (result.updated, result.unchanged) == (1, 239)
    assert result.reasons["source_hash_mismatch"] == 1
    healed = embeddings(catalog)[PID]
    assert healed["vector"] != old_vector
    assert PID in dense_ids(catalog, query) and status(catalog).current


def test_model_revision_change_makes_every_row_stale_and_is_healed(catalog):
    embed_fake(catalog)
    other = replace(SPEC, revision="f" * 40)
    stale = status(catalog, spec=other)
    assert stale.counts["stale_model"] == 240 and stale.counts["stale_config"] == 240
    assert dense_ids(catalog, "running shoes", spec=other) == []  # other vector space: excluded
    result = embed_fake(catalog, FakeEmbedder(other))
    assert result.updated == 240 and status(catalog, spec=other).current
    assert dense_ids(catalog, "running shoes") == []  # and the old spec's rows are gone


def test_text_version_change_is_stale(catalog, monkeypatch):
    embed_fake(catalog)
    monkeypatch.setattr(di, "EMBEDDING_TEXT_VERSION", "2")
    monkeypatch.setattr("ecommerce_search.embeddings.spec.EMBEDDING_TEXT_VERSION", "2")
    stale = status(catalog)
    assert stale.counts["stale_text_version"] == 240 and stale.counts["stale_config"] == 240


def test_tampered_text_hash_is_detected_and_healed(catalog):
    embed_fake(catalog)
    sql(
        catalog,
        "UPDATE product_embeddings SET embedding_text_sha256 = :s WHERE product_id = :p",
        s="0" * 64,
        p=PID,
    )
    assert status(catalog).counts["text_hash_mismatch"] == 1
    assert embed_fake(catalog).updated == 1 and status(catalog).current


def test_encoder_failure_writes_nothing(catalog):
    embed_fake(catalog)
    before = embeddings(catalog)
    fake = FakeEmbedder()
    fake.fail_after_batches = 1  # second batch fails
    with pytest.raises(EmbedderUnavailable):
        embed_fake(catalog, fake, rebuild_all=True)
    assert embeddings(catalog) == before
    sql(catalog, "DELETE FROM product_embeddings")
    first_run = FakeEmbedder()
    first_run.fail_after_batches = 2  # third of four batches fails: earlier batches never land
    with pytest.raises(EmbedderUnavailable):
        embed_fake(catalog, first_run)
    assert embeddings(catalog) == {}


def test_database_rejection_rolls_back_the_whole_write(catalog, monkeypatch):
    embed_fake(catalog)
    before = embeddings(catalog)
    original = di.vector_literal
    calls = {"n": 0}

    def poisoned(vector):
        calls["n"] += 1
        if calls["n"] == 120:  # one invalid (zero) vector in the middle of the batch
            return "[" + ",".join(["0"] * 384) + "]"
        return original(vector)

    monkeypatch.setattr(di, "vector_literal", poisoned)
    with pytest.raises(IntegrityError):
        embed_fake(catalog, rebuild_all=True)
    assert embeddings(catalog) == before  # nothing changed, embedded_at included


def test_any_text_over_the_token_limit_refuses_the_whole_run(catalog):
    fake = FakeEmbedder()
    fake.token_overrides["Strato"] = 257
    with pytest.raises(di.EmbeddingError, match="never truncated"):
        embed_fake(catalog, fake)
    assert embeddings(catalog) == {} and fake.encoded == []


def test_product_changed_during_the_run_is_not_written(catalog, settings):
    def change_mid_run(texts):
        if len(fake.encoded) == 0:  # during the first encode batch, before the write phase
            change_product(catalog, PID, "Changed while embedding")

    fake = FakeEmbedder()
    fake.on_encode = change_mid_run
    result = embed_fake(catalog, fake)
    assert result.changed_during_run == 1 and result.inserted == 239
    assert PID not in embeddings(catalog)
    assert status(catalog).counts["missing"] == 1
    assert embed_fake(catalog).inserted == 1 and status(catalog).current


def test_uncommitted_concurrent_change_leaves_a_detectably_stale_row(catalog):
    holder = catalog.connect()
    transaction = holder.begin()
    holder.execute(
        text("UPDATE products SET title = 'Concurrent', content_sha256 = :s WHERE product_id = :p"),
        {"s": "9" * 64, "p": PID},
    )
    try:
        result = embed_fake(catalog)  # sees only committed content; takes no product row locks
        assert result.inserted == 240
    finally:
        transaction.commit()
        holder.close()
    assert status(catalog).counts["source_hash_mismatch"] == 1  # never silently current
    assert PID not in dense_ids(catalog, "Strato running shoes")
    assert embed_fake(catalog).updated == 1 and status(catalog).current


def _waiting_advisory(engine) -> int:
    with engine.connect() as probe:
        return probe.execute(
            text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted")
        ).scalar_one()


def test_concurrent_embeds_are_serialized_by_the_embedding_lock(catalog):
    holder = catalog.connect()
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(di.embedding_lock_key())))
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = embed_fake(catalog)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    try:
        deadline = time.monotonic() + 30
        while _waiting_advisory(catalog) != 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert _waiting_advisory(catalog) == 1 and worker.is_alive()
        assert embeddings(catalog) == {}  # nothing written while blocked
    finally:
        transaction.rollback()
        holder.close()
    worker.join(120)
    assert not worker.is_alive() and "error" not in outcome
    assert outcome["result"].inserted == 240


def test_embed_does_not_wait_for_ingestion_or_reindex_locks(catalog):
    # Lock order: dataset -> search -> embedding. Embedding writers take only the last one, so a
    # holder of the dataset and search locks (an ingestion in progress) never blocks embed.
    holder = catalog.connect()
    transaction = holder.begin()
    holder.execute(select(func.pg_advisory_xact_lock(ingest_lock_key("synthetic-seed"))))
    holder.execute(select(func.pg_advisory_xact_lock(search_lock_key())))
    try:
        result = embed_fake(catalog)
        assert result.inserted == 240
    finally:
        transaction.rollback()
        holder.close()


def test_reingesting_the_same_seed_leaves_embeddings_untouched(catalog):
    embed_fake(catalog)
    before = embeddings(catalog)
    ingest_file(catalog, SEED, PROVENANCE)
    assert embeddings(catalog) == before


def test_products_absent_from_a_newer_file_keep_their_embeddings(catalog, tmp_path):
    embed_fake(catalog)
    lines = [ln for ln in seed_lines() if PID not in ln]
    lines[0] = edit_line(lines[0], description="Updated description for version two.")
    path, prov = make_variant(tmp_path, lines, version="2")
    outcome = ingest_file(catalog, path, prov)
    assert outcome.result is not None and PID in outcome.result.absent_product_ids
    assert PID in embeddings(catalog)  # never deleted
    result = embed_fake(catalog)
    assert result.updated == 1 and result.unchanged == 239 and status(catalog).current


def test_embed_refuses_a_database_not_migrated_to_0004(settings, scratch_database, migrator):
    migrator.upgrade(scratch_database, "0003")
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        with pytest.raises(di.EmbeddingError, match="alembic upgrade head"):
            embed_fake(engine)
    finally:
        engine.dispose()


# ---- CLI --------------------------------------------------------------------------------------------


def test_cli_embed_and_status(catalog, scratch_database, monkeypatch, capsys):
    monkeypatch.setattr(search_cli, "make_embedder", lambda settings: FakeEmbedder())
    assert (
        search_cli.main(["embed-status", "--database", scratch_database, "--require-current"]) == 1
    )
    assert "missing=240" in capsys.readouterr().out
    assert search_cli.main(["embed", "--database", scratch_database]) == 0
    out = capsys.readouterr().out
    assert f"target database: {scratch_database}" in out and "inserted=240" in out
    code = search_cli.main(
        ["embed-status", "--database", scratch_database, "--require-current", "--verify-vectors"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "state=CURRENT" in out and "vector_mismatch=0" in out
    assert "vectors_reencoded=yes" in out


def test_cli_status_reencode_detects_a_replaced_vector(
    catalog, scratch_database, monkeypatch, capsys
):
    monkeypatch.setattr(search_cli, "make_embedder", lambda settings: FakeEmbedder())
    embed_fake(catalog)
    other = "[" + ",".join(["0"] * 383 + ["1"]) + "]"
    sql(
        catalog,
        "UPDATE product_embeddings SET embedding = CAST(:v AS vector) WHERE product_id = :p",
        v=other,
        p=PID,
    )
    code = search_cli.main(
        ["embed-status", "--database", scratch_database, "--require-current", "--verify-vectors"]
    )
    assert code == 1 and "vector_mismatch=1" in capsys.readouterr().out


def test_cli_embed_failures_are_sanitized(catalog, scratch_database, monkeypatch, capsys):
    fake = FakeEmbedder()
    fake.fail_after_batches = 0
    fake.failure = EmbedderUnavailable("snapshot_missing")
    monkeypatch.setattr(search_cli, "make_embedder", lambda settings: fake)
    assert search_cli.main(["embed", "--database", scratch_database]) == 1
    err = capsys.readouterr().err
    assert "snapshot_missing" in err and "model-fetch" in err and "nothing was written" in err
    assert embeddings(catalog) == {}


def test_cli_embed_requires_an_explicit_database(capsys):
    with pytest.raises(SystemExit) as info:
        search_cli.main(["embed"])
    assert info.value.code == 2
    with pytest.raises(SystemExit):
        search_cli.main(["embed-status"])
