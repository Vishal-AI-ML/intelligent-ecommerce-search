"""Offline smoke tests with the real, pinned MiniLM snapshot (ADR-006). Select with `-m real_model`.

They need the snapshot fetched into the git-ignored `models/` directory and FAIL (they are not
skipped) when it is missing. They never use the network: the model-only tests block outbound
sockets entirely. No relevance is asserted: probe queries only have to return `top_k` rows.
"""

import math
import socket
from pathlib import Path

import pytest
from catalog_support import PROVENANCE, SEED
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ecommerce_search.api.app import create_app
from ecommerce_search.embeddings.fetch import verify_manifest
from ecommerce_search.embeddings.sentence_transformers_provider import SentenceTransformerEmbedder
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2, repository_models_dir
from ecommerce_search.embeddings.text import build_embedding_text
from ecommerce_search.ingestion.loader import load_catalog
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.search.dense import dense_search
from ecommerce_search.search.dense_indexing import embed, embedding_status

pytestmark = pytest.mark.real_model

SPEC = ALL_MINILM_L6_V2
PROBES = [
    "coding laptop",
    "student laptop",
    "lightweight laptop",
    "premium phone",
    "running shoes",
    "noise cancelling headphones",
]


def models_dir() -> Path:
    directory = repository_models_dir()
    assert directory is not None
    problems = verify_manifest(directory, SPEC.model_id, SPEC.revision)
    assert problems == [], f"MiniLM snapshot missing or modified (run model-fetch): {problems}"
    return directory


@pytest.fixture(scope="module")
def embedder():
    provider = SentenceTransformerEmbedder(SPEC, models_dir())
    provider.load()
    return provider


def _refuse(*args, **kwargs):
    raise OSError("network access is not allowed in real-model tests")


def test_loads_offline_from_the_pinned_snapshot(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    provider = SentenceTransformerEmbedder(SPEC, models_dir())
    assert provider.load() is not None and provider.loaded
    vector = provider.embed_query("running shoes")
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-3)


def test_corpus_fits_the_token_limit_and_encodes_deterministically(embedder):
    texts = [build_embedding_text(line.record) for line in load_catalog(SEED).lines]
    counts = embedder.count_tokens(texts, "document")
    assert max(counts) <= SPEC.max_seq_length == 256
    first = embedder.embed_documents(texts[:40])
    again = embedder.embed_documents(texts[:40])
    for a, b in zip(first, again, strict=True):
        assert math.fsum(x * y for x, y in zip(a, b, strict=True)) >= 0.9999


@pytest.mark.integration
def test_real_embeddings_on_a_scratch_database(settings, migrated_engine, embedder):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    result = embed(migrated_engine, embedder)
    assert result.inserted == 240
    with Session(migrated_engine) as session:
        status = embedding_status(session, SPEC, embedder)
        assert status.current and status.vectors_verified
        for probe in PROBES:
            hits = dense_search(session, embedder.embed_query(probe), SPEC, 10).hits
            assert len(hits) == 10  # no relevance assertion: qualitative probes only
            assert [h.dense_score for h in hits] == sorted(
                (h.dense_score for h in hits), reverse=True
            )


@pytest.mark.integration
def test_api_serves_dense_search_with_the_real_model(settings, migrated_engine):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    embed(migrated_engine, SentenceTransformerEmbedder(SPEC, models_dir()))
    scratch = settings.model_copy(update={"postgres_db": migrated_engine.url.database})
    with TestClient(create_app(scratch)) as client:
        first = client.get("/search/dense", params={"q": "noise cancelling headphones"}).json()
        second = client.post("/search/dense", json={"query": "running shoes", "top_k": 3}).json()
    assert first["result_count"] == 10 and first["latency_ms"]["model_load_ms"] > 0
    assert second["result_count"] == 3 and second["latency_ms"]["model_load_ms"] is None
