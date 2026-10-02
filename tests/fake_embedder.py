"""A deterministic, offline stand-in for the embedding model (tests only; no torch, no files).

Vectors are a normalized, hashed bag of words: each lower-cased `[a-z0-9]+` token adds +-1 to a
dimension chosen by SHA-256 of the token. Texts that share words therefore have higher cosine
similarity, which is enough to exercise ranking plumbing. It says nothing about the quality of
any real model. Token counts are `words + 2` (like special tokens around a wordpiece sequence).
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Literal

from ecommerce_search.embeddings.provider import EmbedderUnavailable
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2, EmbeddingModelSpec

TOKEN = re.compile(r"[a-z0-9]+")


def bag_of_words_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in TOKEN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    if not any(vector):
        vector[0] = 1.0  # never a zero vector
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class FakeEmbedder:
    def __init__(self, spec: EmbeddingModelSpec = ALL_MINILM_L6_V2) -> None:
        self._spec = spec
        self.load_calls = 0
        self.loaded = False
        self.encoded: list[str] = []
        self.token_overrides: dict[str, int] = {}  # substring -> token count
        self.fail_after_batches: int | None = None  # raise on the batch after N batches
        self.failure: Exception = EmbedderUnavailable("encode_failed")
        self.on_encode = None  # callable(texts) hook, e.g. to change the database mid-run
        self._batches = 0

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self._spec

    def load(self) -> float | None:
        self.load_calls += 1
        if self.loaded:
            return None
        self.loaded = True
        return 1.0

    def count_tokens(self, texts: Sequence[str], kind: Literal["document", "query"]) -> list[int]:
        self.load()
        counts = []
        for text in texts:
            count = len(TOKEN.findall(text.lower())) + 2
            for marker, forced in self.token_overrides.items():
                if marker in text:
                    count = forced
            counts.append(count)
        return counts

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.load()
        if self.fail_after_batches is not None and self._batches >= self.fail_after_batches:
            raise self.failure
        self._batches += 1
        if self.on_encode is not None:
            self.on_encode(list(texts))
        self.encoded += list(texts)
        return [bag_of_words_vector(t, self._spec.dimension) for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
