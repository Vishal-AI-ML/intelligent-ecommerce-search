from pathlib import Path

import pytest

from ecommerce_search.embeddings import text as et
from ecommerce_search.embeddings.spec import (
    EmbeddingModelSpec,
    SpecError,
    check_revision,
    repo_folder_name,
    snapshot_dir,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")

REV = "0123456789abcdef0123456789abcdef01234567"


def make(**overrides) -> EmbeddingModelSpec:
    values = {
        "model_id": "org/model-name",
        "revision": REV,
        "dimension": 8,
        "max_seq_length": 128,
        "normalize": True,
    }
    values.update(overrides)
    return EmbeddingModelSpec(**values)


@pytest.mark.parametrize("revision", ["main", "v1.0", REV.upper(), REV[:39], REV + "0", ""])
def test_only_full_commit_hashes_are_accepted(revision):
    with pytest.raises(SpecError):
        check_revision(revision)


@pytest.mark.parametrize("model_id", ["name-only", "a/b/c", "../x", "org/..", "/x", "org/"])
def test_malformed_model_ids_are_refused(model_id):
    with pytest.raises(SpecError):
        make(model_id=model_id)


@pytest.mark.parametrize("field", ["dimension", "max_seq_length"])
def test_sizes_must_be_positive(field):
    with pytest.raises(SpecError):
        make(**{field: 0})


def test_config_hash_is_stable_and_covers_every_field():
    base = make()
    assert base.config_sha256() == make().config_sha256()
    changed = [
        make(model_id="org/other"),
        make(revision="f" * 40),
        make(dimension=16),
        make(max_seq_length=256),
        make(normalize=False),
        make(query_prefix="query: "),
        make(document_prefix="passage: "),
    ]
    hashes = {base.config_sha256()} | {spec.config_sha256() for spec in changed}
    assert len(hashes) == len(changed) + 1
    assert base.config()["embedding_text_version"] == et.EMBEDDING_TEXT_VERSION


def test_text_version_is_part_of_the_config(monkeypatch):
    before = make().config_sha256()
    monkeypatch.setattr("ecommerce_search.embeddings.spec.EMBEDDING_TEXT_VERSION", "999")
    assert make().config_sha256() != before


def test_snapshot_layout_is_the_hub_cache_layout():
    assert repo_folder_name("BAAI/bge-small-en-v1.5") == "models--BAAI--bge-small-en-v1.5"
    assert snapshot_dir(Path("models"), "org/name", REV) == Path(
        "models/hub/models--org--name/snapshots", REV
    )


def test_production_registry_pins_the_signed_off_model():
    from ecommerce_search.embeddings.spec import (
        DEFAULT_EMBEDDING_MODEL_ID,
        EMBEDDING_MODELS,
    )

    assert list(EMBEDDING_MODELS) == ["sentence-transformers/all-MiniLM-L6-v2"]
    spec = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_ID]
    # Values approved in ADR-006; changing any of them requires a new ADR and re-embedding.
    assert spec == EmbeddingModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        dimension=384,
        max_seq_length=256,
        normalize=True,
        query_prefix="",
        document_prefix="",
    )
