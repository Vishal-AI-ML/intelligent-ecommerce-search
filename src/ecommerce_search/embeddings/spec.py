"""Embedding-model specification and on-disk snapshot layout.

A spec pins everything that changes what a vector means: model id, immutable revision,
dimension, maximum input length, normalization and the query/document prefixes. Together with
the embedding-text version it is hashed into `config_sha256()`, so any change to any of them is
detectable on stored embeddings.

Snapshots live under the repository's git-ignored `models/` directory in the Hugging Face hub
cache layout: `<models_dir>/hub/models--<org>--<name>/snapshots/<revision>/`. Loading reads that
directory directly, so no hub lookup (and no network access) is involved.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class SpecError(ValueError):
    """An embedding-model specification is malformed."""


def check_model_id(model_id: str) -> str:
    if not MODEL_ID_RE.fullmatch(model_id) or ".." in model_id:
        raise SpecError("model id must look like 'organization/name'")
    return model_id


def check_revision(revision: str) -> str:
    """Only a full 40-character lowercase commit hash is immutable (branches and tags move)."""
    if not REVISION_RE.fullmatch(revision):
        raise SpecError("revision must be a full 40-character lowercase hexadecimal commit hash")
    return revision


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str
    revision: str
    dimension: int
    max_seq_length: int
    normalize: bool
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        check_model_id(self.model_id)
        check_revision(self.revision)
        if self.dimension <= 0 or self.max_seq_length <= 0:
            raise SpecError("dimension and max_seq_length must be positive")

    def config(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        }

    def config_sha256(self) -> str:
        canonical = json.dumps(self.config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def repo_folder_name(model_id: str) -> str:
    """Hub cache folder name: `org/name` -> `models--org--name`."""
    return "models--" + check_model_id(model_id).replace("/", "--")


def snapshot_dir(models_dir: Path, model_id: str, revision: str) -> Path:
    return models_dir / "hub" / repo_folder_name(model_id) / "snapshots" / check_revision(revision)


# ---- production registry -------------------------------------------------------------------------
# Selected in Milestone 4 (docs/decisions/ADR-006-embedding-model-selection.md) from the measured
# selection experiment and explicit user sign-off. Provisional until Golden Dataset evaluation.
# Adding or changing an entry is a reviewed code change plus an ADR, never a configuration edit.

ALL_MINILM_L6_V2 = EmbeddingModelSpec(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    dimension=384,
    max_seq_length=256,
    normalize=True,
    query_prefix="",
    document_prefix="",
)

EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {ALL_MINILM_L6_V2.model_id: ALL_MINILM_L6_V2}
DEFAULT_EMBEDDING_MODEL_ID = ALL_MINILM_L6_V2.model_id


def repository_models_dir() -> Path | None:
    """`<repository root>/models` (git-ignored) when running from a source checkout, else None.

    Anchored to this file, never to the current working directory, so commands run from a
    subdirectory still use the ignored repository-root `models/`. An installed (non-checkout)
    package has no repository root; callers must then be given an explicit directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "ecommerce_search").is_dir():
            return parent / "models"
    return None
