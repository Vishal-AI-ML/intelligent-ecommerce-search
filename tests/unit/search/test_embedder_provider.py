"""The sentence-transformers provider, with the library replaced by a fake module.

No test here imports torch or sentence-transformers, downloads anything or opens a socket.
"""

import logging
import math
import os
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

from ecommerce_search.embeddings import sentence_transformers_provider as stp
from ecommerce_search.embeddings.provider import EmbedderUnavailable, validate_vectors
from ecommerce_search.embeddings.spec import EmbeddingModelSpec, snapshot_dir

pytestmark = pytest.mark.usefixtures("no_socket_connect")

REV = "0123456789abcdef0123456789abcdef01234567"
DIM = 4
ROOT = Path(__file__).resolve().parents[3]


def spec(**overrides) -> EmbeddingModelSpec:
    values = {
        "model_id": "org/fake",
        "revision": REV,
        "dimension": DIM,
        "max_seq_length": 16,
        "normalize": True,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }
    values.update(overrides)
    return EmbeddingModelSpec(**values)


class _Array:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class FakeSentenceTransformer:
    instances: list["FakeSentenceTransformer"] = []
    output: list[float] | None = None  # override every vector
    dimension = DIM
    max_seq_length = 16
    fail = False

    def __init__(self, path, **kwargs):
        if FakeSentenceTransformer.fail:
            raise OSError(f"cannot read {path}")  # message must never escape
        self.path, self.kwargs = path, kwargs
        self.calls: list[list[str]] = []
        FakeSentenceTransformer.instances.append(self)
        self.tokenizer = lambda texts, **kw: {"input_ids": [t.split() for t in texts]}

    def get_embedding_dimension(self):
        return FakeSentenceTransformer.dimension

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.encode_kwargs = kwargs
        if FakeSentenceTransformer.output is not None:
            return _Array([list(FakeSentenceTransformer.output) for _ in texts])
        return _Array([[0.5, 0.5, 0.5, 0.5] for _ in texts])


@pytest.fixture
def fake_library(monkeypatch):
    FakeSentenceTransformer.instances = []
    FakeSentenceTransformer.output = None
    FakeSentenceTransformer.dimension = DIM
    FakeSentenceTransformer.max_seq_length = 16
    FakeSentenceTransformer.fail = False
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for key in stp.OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    return FakeSentenceTransformer


def make_snapshot(models_dir: Path) -> Path:
    path = snapshot_dir(models_dir, "org/fake", REV)
    path.mkdir(parents=True)
    for name in stp.REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    return path


def test_constructing_the_provider_touches_nothing(tmp_path, fake_library):
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path / "absent")
    assert not embedder.loaded and fake_library.instances == []


def test_missing_snapshot_is_detected_before_importing_the_library(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    with pytest.raises(EmbedderUnavailable) as info:
        embedder.load()
    assert info.value.reason == "snapshot_missing"
    assert str(tmp_path) not in str(info.value)
    assert "sentence_transformers" not in sys.modules


def test_load_uses_the_local_snapshot_offline_without_remote_code(tmp_path, fake_library):
    path = make_snapshot(tmp_path)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    load_ms = embedder.load()
    assert load_ms is not None and load_ms >= 0
    (model,) = fake_library.instances
    assert Path(model.path) == path
    assert model.kwargs == {"device": "cpu", "trust_remote_code": False, "local_files_only": True}
    assert {k: os.environ[k] for k in stp.OFFLINE_ENV} == stp.OFFLINE_ENV
    assert embedder.load() is None  # reused, not reloaded
    assert len(fake_library.instances) == 1


def test_concurrent_first_use_loads_once(tmp_path, fake_library):
    make_snapshot(tmp_path)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    threads = [threading.Thread(target=embedder.load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(fake_library.instances) == 1


def test_a_failed_load_is_sanitized_and_retried(tmp_path, fake_library):
    make_snapshot(tmp_path)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    fake_library.fail = True
    with pytest.raises(EmbedderUnavailable) as info:
        embedder.load()
    assert info.value.reason == "load_failed"
    assert "cannot read" not in str(info.value) and info.value.__cause__ is None
    fake_library.fail = False
    assert embedder.load() is not None and embedder.loaded


@pytest.fixture
def library_logger():
    logger = logging.getLogger(stp.LIBRARY_LOGGER)
    level = logger.level
    logger.setLevel(logging.NOTSET)
    yield logger
    logger.setLevel(level)


def test_loading_does_not_log_the_local_snapshot_path(
    tmp_path, fake_library, library_logger, caplog, monkeypatch
):
    models_dir = tmp_path / "SENTINEL_models_7f3a9c"
    path = make_snapshot(models_dir)
    child = logging.getLogger(f"{stp.LIBRARY_LOGGER}.base.model")

    def logging_init(self, model_path, **kwargs):  # what the real library logs while loading
        child.info(f"Loading SentenceTransformer model from {model_path}.")
        child.warning("library warning")
        child.error("library error")
        original_init(self, model_path, **kwargs)

    original_init = FakeSentenceTransformer.__init__
    monkeypatch.setattr(FakeSentenceTransformer, "__init__", logging_init)
    caplog.set_level(logging.DEBUG)
    embedder = stp.SentenceTransformerEmbedder(spec(), models_dir)
    assert embedder.load() is not None
    logging.getLogger("ecommerce_search.api.dense").error(
        "dense search unavailable: %s", "load_failed"
    )

    (model,) = fake_library.instances
    assert Path(model.path) == path
    messages = [(r.name, r.levelno, r.getMessage()) for r in caplog.records]
    assert not any("Loading SentenceTransformer" in m for _, _, m in messages)
    assert "SENTINEL_models_7f3a9c" not in caplog.text and str(path) not in caplog.text
    assert ("sentence_transformers.base.model", logging.WARNING, "library warning") in messages
    assert ("sentence_transformers.base.model", logging.ERROR, "library error") in messages
    assert (
        "ecommerce_search.api.dense",
        logging.ERROR,
        "dense search unavailable: load_failed",
    ) in messages
    assert library_logger.level == logging.WARNING
    assert logging.getLogger("ecommerce_search").level == logging.NOTSET  # project logs untouched


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [("dimension", 8, "dimension_mismatch"), ("max_seq_length", 512, "config_mismatch")],
)
def test_model_must_match_the_spec(tmp_path, fake_library, attribute, value, reason):
    make_snapshot(tmp_path)
    setattr(fake_library, attribute, value)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    with pytest.raises(EmbedderUnavailable) as info:
        embedder.load()
    assert info.value.reason == reason and not embedder.loaded


def test_prefixes_and_normalization_are_applied(tmp_path, fake_library):
    make_snapshot(tmp_path)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path, batch_size=7)
    docs = embedder.embed_documents(["a b", "c"])
    query = embedder.embed_query("x y")
    model = fake_library.instances[0]
    assert model.calls == [["passage: a b", "passage: c"], ["query: x y"]]
    assert model.encode_kwargs["normalize_embeddings"] is True
    assert model.encode_kwargs["batch_size"] == 7
    assert len(docs) == 2 and query == [0.5] * 4
    assert embedder.embed_documents([]) == []


def test_token_counts_include_the_prefix(tmp_path, fake_library):
    make_snapshot(tmp_path)
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    assert embedder.count_tokens(["a b c"], "document") == [4]
    assert embedder.count_tokens(["a"], "query") == [2]


@pytest.mark.parametrize(
    "vector",
    [
        [math.nan, 0.0, 0.0, 1.0],
        [math.inf, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
    ],
)
def test_invalid_model_output_is_rejected(tmp_path, fake_library, vector):
    make_snapshot(tmp_path)
    fake_library.output = vector
    embedder = stp.SentenceTransformerEmbedder(spec(), tmp_path)
    with pytest.raises(EmbedderUnavailable) as info:
        embedder.embed_query("x")
    assert info.value.reason == "invalid_output"


def test_validate_vectors_checks_count_dimension_and_norm():
    unit = [1.0, 0.0, 0.0, 0.0]
    assert validate_vectors([unit], count=1, dimension=4, normalized=True) == [unit]
    assert validate_vectors([[3.0, 4.0]], count=1, dimension=2, normalized=False) == [[3.0, 4.0]]
    with pytest.raises(EmbedderUnavailable, match="invalid_output"):
        validate_vectors([unit], count=2, dimension=4, normalized=True)
    with pytest.raises(EmbedderUnavailable, match="dimension_mismatch"):
        validate_vectors([unit], count=1, dimension=3, normalized=True)
    within = [[0.9995, 0.0, 0.0, 0.0]]
    assert validate_vectors(within, count=1, dimension=4, normalized=True) == within
    with pytest.raises(EmbedderUnavailable, match="invalid_output"):
        validate_vectors([[0.99, 0.0, 0.0, 0.0]], count=1, dimension=4, normalized=True)


def test_importing_the_application_does_not_import_torch():
    # Ingestion, the CLIs, the dense indexing/audit modules and app startup never load the model.
    code = textwrap.dedent(
        """
        import sys
        import ecommerce_search.embeddings.sentence_transformers_provider
        import ecommerce_search.embeddings.fetch
        import ecommerce_search.search.cli
        import ecommerce_search.catalog.cli
        import ecommerce_search.ingestion.service
        import ecommerce_search.search.dense_indexing
        import ecommerce_search.search.dense_audit
        from ecommerce_search.api.app import create_app
        from ecommerce_search.config import Settings
        create_app(Settings(_env_file=None, postgres_password="x"))
        heavy = {"torch", "sentence_transformers", "transformers", "huggingface_hub", "hf_xet"}
        print(sorted(heavy & set(sys.modules)))
        """
    )
    done = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"
