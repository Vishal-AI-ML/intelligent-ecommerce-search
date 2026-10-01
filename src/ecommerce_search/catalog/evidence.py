"""Durable, versioned human-review evidence and manifest-exact review status.

The evidence file lives under `data/labels/`, is committed only after a real human review, and
can be verified offline (no database) against the committed seed. It embeds the exact manifest
that was reviewed, so it is self-contained: verification checks that embedded historical record
on its own terms plus the committed seed's dataset checksum and product content hashes, and
NEVER depends on the current sampling code or constants (rules version, selection wording,
quotas, seed, row rendering). Comparison with today's sampling rules is informational only.

It records a reviewer *handle* (not an email). Git history plus the handle give an audit trail
and the hashes give integrity; neither is cryptographic proof of a human's identity.
"""

import copy
import glob
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ecommerce_search.catalog.manifest import ManifestError, parse_manifest

EVIDENCE_SCHEMA_VERSION = 2
EVIDENCE_DIR = Path("data/labels")

SHA256_RE = r"^[0-9a-f]{64}$"
HANDLE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$"
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
FREE_TEXT_MAX = 1000
RECORDED_AT_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

AUDIT_NOTE = (
    "Human-review evidence for a SYNTHETIC catalog sample. The reviewer handle and this file's "
    "Git history are an audit trail, not cryptographic proof of a human's identity."
)
PRIVACY_RULES = (
    "Use a handle or pseudonym, never an email address. Do not put personal information, "
    "secrets or machine-specific paths in issue_fields or notes."
)


class EvidenceError(Exception):
    """The evidence file is missing, malformed, tampered with or does not match the sample."""


Verdict = Literal["accept", "needs_correction", "unsure"]


class EvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str
    product_content_sha256: str = Field(pattern=SHA256_RE)
    verdict: Verdict
    issue_fields: str | None = Field(default=None, max_length=FREE_TEXT_MAX)
    notes: str | None = Field(default=None, max_length=FREE_TEXT_MAX)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2]
    evidence_kind: Literal["catalog-human-review"]
    review_batch_id: str
    manifest_sha256: str = Field(pattern=SHA256_RE)
    dataset_id: str
    dataset_version: str = Field(pattern=r"^[1-9][0-9]{0,8}$")
    dataset_checksum_sha256: str = Field(pattern=SHA256_RE)
    manifest: dict[str, Any]  # the exact reviewed manifest (a historical record)
    reviewer_handle: str = Field(pattern=HANDLE_PATTERN)
    recorded_at: str = Field(pattern=RECORDED_AT_PATTERN)
    product_count: int = Field(ge=1)
    reviews: list[EvidenceReview]
    audit_note: str
    privacy_rules: str
    records_sha256: str = Field(pattern=SHA256_RE)


def canonical_hash(body: Mapping[str, Any]) -> str:
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_handle(handle: str | None) -> str:
    if handle is None or not handle.strip():
        raise EvidenceError("a non-blank reviewer handle is required")
    handle = handle.strip()
    if "@" in handle or not re.fullmatch(HANDLE_PATTERN, handle):
        raise EvidenceError(
            "the reviewer handle must be a short handle or pseudonym (letters, digits, space, "
            "'.', '_', '-'; at most 64 characters), never an email address"
        )
    return handle


def validate_free_text(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) > FREE_TEXT_MAX:
        raise EvidenceError(f"{label} is longer than {FREE_TEXT_MAX} characters")
    if EMAIL_RE.search(value):
        raise EvidenceError(
            f"{label} looks like it contains an email address; remove personal data"
        )
    return value


def evidence_filename(manifest: Mapping) -> str:
    return (
        f"catalog_review_{manifest['dataset_id']}_v{manifest['dataset_version']}_"
        f"{manifest['manifest_sha256'][:12]}.json"
    )


def evidence_path(directory: Path, manifest: Mapping) -> Path:
    return directory / evidence_filename(manifest)


def build_evidence(
    manifest: Mapping,
    reviews: Sequence[Any],
    *,
    reviewer_handle: str,
    recorded_at: datetime,
) -> dict[str, Any]:
    """`reviews` are objects with product_id, product_content_sha256, verdict, issue_fields,
    notes (see review.ReviewRow), in manifest order."""
    stamp = recorded_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "catalog-human-review",
        "review_batch_id": manifest["review_batch_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "dataset_checksum_sha256": manifest["dataset_checksum_sha256"],
        "manifest": copy.deepcopy(dict(manifest)),
        "reviewer_handle": validate_handle(reviewer_handle),
        "recorded_at": stamp,
        "product_count": len(reviews),
        "reviews": [
            {
                "product_id": r.product_id,
                "product_content_sha256": r.product_content_sha256,
                "verdict": r.verdict,
                "issue_fields": r.issue_fields,
                "notes": r.notes,
            }
            for r in reviews
        ],
        "audit_note": AUDIT_NOTE,
        "privacy_rules": PRIVACY_RULES,
    }
    body["records_sha256"] = canonical_hash(body)
    return body


def write_evidence(directory: Path, evidence: Mapping[str, Any], manifest: Mapping) -> Path:
    """Write the evidence file exclusively (never overwrite an existing file)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = evidence_path(directory, manifest)
    data = (json.dumps(evidence, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with open(path, "xb") as handle:  # noqa: PTH123 - exclusive create
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise EvidenceError(
            f"evidence file {path.name} already exists; evidence is never overwritten"
        ) from None
    return path


def load_evidence(path: Path) -> Evidence:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceError(f"evidence file {path.name} does not exist") from None
    except UnicodeDecodeError:
        raise EvidenceError(f"evidence file {path.name} is not valid UTF-8") from None
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"evidence file {path.name} is not valid JSON ({exc.msg})") from None
    try:
        return Evidence.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<file>'}: {err['msg']}"
            for err in exc.errors(include_input=False)[:8]
        )
        raise EvidenceError(f"evidence file {path.name} is malformed: {problems}") from None


def find_evidence(directory: Path, dataset_id: str, dataset_version: str) -> list[Path]:
    """Evidence files for a dataset version. The name embeds the (historical) manifest hash,
    so it is located by dataset id/version, never by today's regenerated manifest."""
    pattern = glob.escape(f"catalog_review_{dataset_id}_v{dataset_version}_") + "*.json"
    return sorted(directory.glob(pattern)) if directory.is_dir() else []


@dataclass(frozen=True)
class SeedFacts:
    """What the committed seed says: the only external facts historical verification uses."""

    dataset_id: str
    dataset_version: str
    dataset_checksum_sha256: str
    content_hashes: Mapping[str, str]  # product_id -> normalized content_sha256


@dataclass(frozen=True)
class VerificationResult:
    manifest: Mapping[str, Any]  # the validated embedded (historical) manifest
    # Informational only: does today's sampling code reproduce the reviewed manifest exactly?
    # None when no current manifest was supplied or it could not be derived.
    matches_current_sampling_rules: bool | None


def verify_evidence(
    evidence: Evidence, seed: SeedFacts, current_manifest: Mapping | None = None
) -> VerificationResult:
    """Offline historical verification. Depends on the evidence file and the committed seed
    only; `current_manifest` merely feeds the informational compatibility flag."""
    problems: list[str] = []
    body = evidence.model_dump(mode="json")
    stored = body.pop("records_sha256")
    if canonical_hash(body) != stored:
        problems.append("records_sha256 does not match the file content (file was altered)")

    embedded: dict[str, Any] | None = None
    try:
        embedded = parse_manifest(evidence.manifest).model_dump(mode="json")
    except ManifestError as exc:
        problems.append(f"embedded manifest is invalid: {exc}")

    if embedded is not None:
        for label, got, want in (
            ("manifest_sha256", evidence.manifest_sha256, embedded["manifest_sha256"]),
            ("review_batch_id", evidence.review_batch_id, embedded["review_batch_id"]),
            ("dataset_id", evidence.dataset_id, embedded["dataset_id"]),
            ("dataset_version", evidence.dataset_version, embedded["dataset_version"]),
            (
                "dataset_checksum_sha256",
                evidence.dataset_checksum_sha256,
                embedded["dataset_checksum_sha256"],
            ),
        ):
            if got != want:
                problems.append(f"{label} does not match the embedded manifest")
        for label, got, want in (
            ("dataset_id", embedded["dataset_id"], seed.dataset_id),
            ("dataset_version", embedded["dataset_version"], seed.dataset_version),
            (
                "dataset_checksum_sha256",
                embedded["dataset_checksum_sha256"],
                seed.dataset_checksum_sha256,
            ),
        ):
            if got != want:
                problems.append(f"{label} does not match the committed seed")

        want_ids = [p["product_id"] for p in embedded["products"]]
        embedded_hash = {p["product_id"]: p["content_sha256"] for p in embedded["products"]}
        for pid, content in embedded_hash.items():
            if seed.content_hashes.get(pid) != content:
                problems.append(f"{pid}: content hash does not match the committed seed")

        got_ids = [r.product_id for r in evidence.reviews]
        if evidence.product_count != len(evidence.reviews):
            problems.append("product_count does not equal the number of reviews")
        dupes = sorted(pid for pid, n in Counter(got_ids).items() if n > 1)
        if dupes:
            problems.append(f"duplicated products: {', '.join(dupes[:5])}")
        missing = sorted(set(want_ids) - set(got_ids))
        extra = sorted(set(got_ids) - set(want_ids))
        if missing:
            problems.append(f"missing products ({len(missing)}): {', '.join(missing[:5])}")
        if extra:
            problems.append(f"unexpected products ({len(extra)}): {', '.join(extra[:5])}")
        if not (dupes or missing or extra) and got_ids != want_ids:
            problems.append("reviews are not in the embedded manifest order")
        for review in evidence.reviews:
            if review.product_id in embedded_hash and (
                review.product_content_sha256 != embedded_hash[review.product_id]
            ):
                problems.append(
                    f"{review.product_id}: product_content_sha256 does not match the "
                    "embedded manifest"
                )

    for review in evidence.reviews:
        for label, value in (("issue_fields", review.issue_fields), ("notes", review.notes)):
            if value and EMAIL_RE.search(value):
                problems.append(f"{review.product_id}: {label} contains an email-like string")
    if "@" in evidence.reviewer_handle:
        problems.append("reviewer_handle looks like an email address")
    try:
        datetime.strptime(evidence.recorded_at, "%Y-%m-%dT%H:%M:%SZ")  # noqa: DTZ007
    except ValueError:
        problems.append("recorded_at is not a valid UTC timestamp")
    if problems or embedded is None:
        raise EvidenceError("evidence verification failed: " + "; ".join(problems[:10]))
    matches = None if current_manifest is None else dict(current_manifest) == embedded
    return VerificationResult(embedded, matches)


@dataclass(frozen=True)
class ReviewStatus:
    state: str  # PENDING | RECORDED | ERROR
    reviewed: int
    total: int
    reviewers: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default_factory=tuple)
    source: str = "evidence"

    def describe(self) -> str:
        head = (
            f"review status ({self.source}): {self.state} ({self.reviewed}/{self.total} sampled "
            f"products have a human verdict recorded"
        )
        if self.reviewers:
            head += f"; reviewer: {', '.join(self.reviewers)}"
        text = head + ")"
        if self.problems:
            text += "\n  " + "\n  ".join(self.problems)
        return text


def evaluate_rows(
    manifest: Mapping, rows: Sequence[Mapping[str, Any]], *, source: str = "database"
) -> ReviewStatus:
    """Manifest-exact status from recorded rows (each row: product_id, verdict, reviewer,
    manifest_sha256, dataset_id, dataset_version, dataset_checksum_sha256,
    product_content_sha256). Anything inconsistent is ERROR; an incomplete clean batch is
    PENDING; only an exact, single-reviewer, fully matching batch is RECORDED."""
    expected = {p["product_id"]: p["content_sha256"] for p in manifest["products"]}
    total = len(expected)
    if not rows:
        return ReviewStatus("PENDING", 0, total, source=source)

    problems: list[str] = []
    reviewers = sorted({r["reviewer"] for r in rows})
    if len(reviewers) > 1:
        problems.append(f"rows from {len(reviewers)} different reviewers (one is required)")
    counts = Counter(r["product_id"] for r in rows)
    dupes = sorted(pid for pid, n in counts.items() if n > 1)
    if dupes:
        problems.append(f"more than one outcome for: {', '.join(dupes[:5])}")
    foreign = sorted(set(counts) - set(expected))
    if foreign:
        problems.append(f"products outside the sample: {', '.join(foreign[:5])}")
    allowed = {"accept", "needs_correction", "unsure"}
    for r in rows:
        pid = r["product_id"]
        if r["manifest_sha256"] != manifest["manifest_sha256"]:
            problems.append(f"{pid}: manifest_sha256 mismatch")
        if (
            r["dataset_id"] != manifest["dataset_id"]
            or r["dataset_version"] != manifest["dataset_version"]
            or r["dataset_checksum_sha256"] != manifest["dataset_checksum_sha256"]
        ):
            problems.append(f"{pid}: dataset/checksum mismatch")
        if pid in expected and r["product_content_sha256"] != expected[pid]:
            problems.append(f"{pid}: product_content_sha256 mismatch")
        if r["verdict"] not in allowed:
            problems.append(f"{pid}: invalid verdict")
    reviewed = len(set(counts) & set(expected))
    if problems:
        return ReviewStatus(
            "ERROR", reviewed, total, tuple(reviewers), tuple(problems[:10]), source
        )
    state = "RECORDED" if reviewed == total else "PENDING"
    return ReviewStatus(state, reviewed, total, tuple(reviewers), (), source)
