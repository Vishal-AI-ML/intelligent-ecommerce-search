"""Offline smoke test of `/search/hybrid` with the real, pinned MiniLM snapshot (ADR-006).

Select with `-m real_model`. FAILS (is not skipped) when the snapshot is missing. Uses a scratch
database only. No relevance is asserted: only structure, ordering and determinism.
"""

import socket

import pytest
from catalog_support import PROVENANCE, SEED
from fastapi.testclient import TestClient

from ecommerce_search.api.app import create_app
from ecommerce_search.embeddings.fetch import verify_manifest
from ecommerce_search.embeddings.sentence_transformers_provider import SentenceTransformerEmbedder
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2, repository_models_dir
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.search.dense_indexing import embed

pytestmark = [pytest.mark.real_model, pytest.mark.integration]

SPEC = ALL_MINILM_L6_V2
PROBES = ["hp laptop", "noise cancelling headphones", "running shoes"]


def models_dir():
    directory = repository_models_dir()
    assert directory is not None
    problems = verify_manifest(directory, SPEC.model_id, SPEC.revision)
    assert problems == [], f"MiniLM snapshot missing or modified (run model-fetch): {problems}"
    return directory


def _refuse(*args, **kwargs):
    raise OSError("network access is not allowed in real-model tests")


def test_api_serves_hybrid_search_with_the_real_model(settings, migrated_engine, monkeypatch):
    ingest_file(migrated_engine, SEED, PROVENANCE)
    assert embed(migrated_engine, SentenceTransformerEmbedder(SPEC, models_dir())).inserted == 240
    scratch = settings.model_copy(update={"postgres_db": migrated_engine.url.database})
    assert scratch.postgres_db != settings.postgres_db  # never the development database
    # No name resolution or outbound Python connections (the database is reached by libpq).
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    both_sources: list[str] = []
    with TestClient(create_app(scratch)) as client:
        for probe in PROBES:
            first = client.get("/search/hybrid", params={"q": probe})
            second = client.post("/search/hybrid", json={"query": probe})
            assert first.status_code == second.status_code == 200
            body = first.json()
            assert body["result_count"] == 10 and body["dense_hit_count"] == 50
            ids = [r["product_id"] for r in body["results"]]
            assert len(set(ids)) == len(ids)
            scores = [r["rrf_score"] for r in body["results"]]
            assert scores == sorted(scores, reverse=True)
            both_sources += [
                r["product_id"]
                for r in body["results"]
                if r["lexical_rank"] is not None and r["dense_rank"] is not None
            ]
            assert body["embedding_model_revision"] == SPEC.revision
            first_body, second_body = first.json(), second.json()
            first_body.pop("latency_ms"), second_body.pop("latency_ms")
            assert first_body == second_body  # deterministic across requests
        assert client.app.state.engine.pool.checkedout() == 0
    # Strict-AND lexical retrieval can find nothing for a probe (e.g. an unseen word), so the
    # fused-from-both-sources check is over all probes, not per probe.
    assert both_sources
