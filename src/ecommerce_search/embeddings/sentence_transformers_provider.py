"""`Embedder` backed by a local sentence-transformers snapshot. CPU only.

This is the only application module that imports torch / sentence-transformers, and it does so
lazily on the first `load()`. Constructing the provider is cheap and touches no files, so the
application can create it at startup without importing torch.

Loading never contacts the network: the model is read from an explicit local snapshot
directory (see `spec.snapshot_dir`), hub offline mode is forced before the import, `HF_HOME` and
other caches are never consulted, and remote code is never trusted. A missing snapshot is
detected by a path check before anything heavy is imported. A failed load is not cached: the
next call tries again (for example after `model-fetch`).
"""

import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from ecommerce_search.embeddings.provider import EmbedderUnavailable, validate_vectors
from ecommerce_search.embeddings.spec import EmbeddingModelSpec, snapshot_dir

REQUIRED_FILES = ("modules.json", "config.json")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class SentenceTransformerEmbedder:
    def __init__(self, spec: EmbeddingModelSpec, models_dir: Path, batch_size: int = 32) -> None:
        self._spec = spec
        self._path = snapshot_dir(models_dir, spec.model_id, spec.revision)
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self._spec

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def snapshot_present(self) -> bool:
        return all((self._path / name).is_file() for name in REQUIRED_FILES)

    def load(self) -> float | None:
        """Load the model if needed. Returns the load time in ms, or None if already loaded."""
        if self._model is not None:
            return None
        with self._lock:
            if self._model is not None:
                return None
            started = time.perf_counter()
            self._model = self._load_model()
            return round((time.perf_counter() - started) * 1000, 3)

    def _load_model(self) -> Any:
        if not self.snapshot_present():
            raise EmbedderUnavailable("snapshot_missing")
        os.environ.update(OFFLINE_ENV)
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                str(self._path),
                device="cpu",
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception:  # any library failure is the same sanitized outcome
            raise EmbedderUnavailable("load_failed") from None
        if model.get_embedding_dimension() != self._spec.dimension:
            raise EmbedderUnavailable("dimension_mismatch")
        if model.max_seq_length != self._spec.max_seq_length:
            raise EmbedderUnavailable("config_mismatch")
        return model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        try:
            array = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._spec.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = array.tolist()
        except Exception:
            raise EmbedderUnavailable("encode_failed") from None
        return validate_vectors(
            vectors,
            count=len(texts),
            dimension=self._spec.dimension,
            normalized=self._spec.normalize,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self._spec.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._spec.query_prefix + text])[0]

    def count_tokens(self, texts: Sequence[str], kind: Literal["document", "query"]) -> list[int]:
        """Token counts including special tokens and the prefix, without truncation."""
        self.load()
        prefix = self._spec.document_prefix if kind == "document" else self._spec.query_prefix
        try:
            encoded = self._model.tokenizer(
                [prefix + t for t in texts], add_special_tokens=True, truncation=False
            )
        except Exception:
            raise EmbedderUnavailable("encode_failed") from None
        return [len(ids) for ids in encoded["input_ids"]]
