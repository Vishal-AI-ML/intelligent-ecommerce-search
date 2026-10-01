"""Schema and integrity rules for a human-review sample manifest.

A manifest is a *historical record* of one review sample. Completed review evidence embeds the
exact manifest that was reviewed and verifies it with the rules in this module only: schema,
recomputed digest and internal consistency. Verification never consults the CURRENT sampling
code or constants, so later changes to rules, selection wording, quotas or rendering cannot
invalidate old evidence. (Comparison with today's sampling code is a separate, informational
step.)
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_SCHEMA_VERSION = 3
SHA256_RE = r"^[0-9a-f]{64}$"
MIN_REVIEW_PRODUCTS = 30  # Milestone 2 requires at least 30 representative products


class ManifestError(Exception):
    """The manifest is malformed, inconsistent or does not match its recorded digest."""


class ManifestProduct(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=SHA256_RE)
    row_sha256: str = Field(pattern=SHA256_RE)


class ManifestV3(BaseModel):
    """Manifest schema version 3. Every field is part of the hashed historical record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_kind: Literal["catalog-review-sample"]
    manifest_schema_version: Literal[3]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^[1-9][0-9]{0,8}$")
    dataset_checksum_sha256: str = Field(pattern=SHA256_RE)
    is_synthetic: bool
    sample_seed: str = Field(min_length=1)
    selection_version: str = Field(min_length=1)
    selection_method: str = Field(min_length=1)
    rules_version: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    quotas: dict[str, int]
    product_ids: list[str]
    products: list[ManifestProduct]
    manifest_sha256: str = Field(pattern=SHA256_RE)
    review_batch_id: str = Field(min_length=1)


# Future manifest schemas are added here; existing entries are never edited.
MANIFEST_MODELS: dict[int, type[BaseModel]] = {3: ManifestV3}


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "review_batch_id")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_id(dataset_id: str, dataset_version: str, digest: str) -> str:
    return f"{dataset_id}-v{dataset_version}-{digest[:12]}"


def parse_manifest(raw: Any) -> BaseModel:
    """Validate a manifest on its own terms (no current-code constants). Raises ManifestError."""
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest is not an object")
    version = raw.get("manifest_schema_version")
    model = MANIFEST_MODELS.get(version) if isinstance(version, int) else None
    if model is None:
        raise ManifestError(f"unsupported manifest schema version {version!r}")
    try:
        parsed = model.model_validate(dict(raw))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<manifest>'}: {err['msg']}"
            for err in exc.errors(include_input=False)[:6]
        )
        raise ManifestError(f"manifest is malformed: {problems}") from None

    data = parsed.model_dump(mode="json")
    if manifest_digest(data) != parsed.manifest_sha256:
        raise ManifestError("manifest content does not match its manifest_sha256")
    expected_batch = batch_id(parsed.dataset_id, parsed.dataset_version, parsed.manifest_sha256)
    if parsed.review_batch_id != expected_batch:
        raise ManifestError("manifest review_batch_id does not match its content")
    ids = [p.product_id for p in parsed.products]
    if ids != parsed.product_ids:
        raise ManifestError("manifest product_ids do not match its products list")
    if len(set(ids)) != len(ids):
        raise ManifestError("manifest lists a product more than once")
    if len(ids) < MIN_REVIEW_PRODUCTS:
        raise ManifestError(
            f"manifest lists {len(ids)} products; at least {MIN_REVIEW_PRODUCTS} are required"
        )
    if sum(parsed.quotas.values()) != len(ids):
        raise ManifestError("manifest quotas do not add up to its number of products")
    return parsed
