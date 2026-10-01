"""The typed embedding-provider interface and model-independent vector validation.

Callers depend on `Embedder`, never on a concrete library. Failures to make a model available
raise `EmbedderUnavailable` with a fixed reason code; its message never contains paths,
exception text or other internal detail, so it is safe to log and to map to a fixed HTTP 503.
"""

import math
from collections.abc import Sequence
from typing import Literal, Protocol

from ecommerce_search.embeddings.spec import EmbeddingModelSpec

ReasonCode = Literal[
    "snapshot_missing",
    "load_failed",
    "dimension_mismatch",
    "config_mismatch",
    "invalid_output",
    "encode_failed",
]

NORM_TOLERANCE = 1e-3


class EmbedderUnavailable(Exception):
    """The embedding model cannot be used. `reason` is one of `ReasonCode`."""

    def __init__(self, reason: ReasonCode) -> None:
        self.reason = reason
        super().__init__(f"embedding model unavailable: {reason}")


class Embedder(Protocol):
    """A local embedding model. Prefixes from `spec` are applied by the implementation."""

    @property
    def spec(self) -> EmbeddingModelSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(
        self, texts: Sequence[str], kind: Literal["document", "query"]
    ) -> list[int]: ...


def validate_vectors(
    vectors: Sequence[Sequence[float]], *, count: int, dimension: int, normalized: bool
) -> list[list[float]]:
    """Return `vectors` as plain float lists, or raise `EmbedderUnavailable("invalid_output")`.

    Checks: the expected number of vectors, the expected dimension, every component finite,
    a non-zero L2 norm and, when the model normalizes, a norm within 1 +/- NORM_TOLERANCE."""
    if len(vectors) != count:
        raise EmbedderUnavailable("invalid_output")
    out: list[list[float]] = []
    for vector in vectors:
        values = [float(v) for v in vector]
        if len(values) != dimension:
            raise EmbedderUnavailable("dimension_mismatch")
        if not all(math.isfinite(v) for v in values):
            raise EmbedderUnavailable("invalid_output")
        norm = math.sqrt(math.fsum(v * v for v in values))
        if norm == 0.0 or (normalized and abs(norm - 1.0) > NORM_TOLERANCE):
            raise EmbedderUnavailable("invalid_output")
        out.append(values)
    return out
