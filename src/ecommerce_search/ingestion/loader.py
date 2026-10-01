"""Read a JSONL catalog file: checksum, parse, validate and normalize (no database access)."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from ecommerce_search.catalog.quality import parameters as qp
from ecommerce_search.catalog.quality.findings import CheckOutcome, Finding, Severity
from ecommerce_search.catalog.schemas import (
    CatalogRecord,
    RecordValidationError,
    parse_raw,
    record_content_hash,
)
from ecommerce_search.catalog.taxonomy import TAXONOMY_VERSION, TRANSFORM_VERSION


class LoaderError(Exception):
    """The input cannot be read or does not match its provenance file."""


class Provenance(BaseModel):
    """Provenance file for one dataset version. `license_identifier` is nullable on purpose:
    rights are explained in `rights_note`, and no licence identifier is invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    # Positive integer without leading zeros: versions of a dataset are ordered numerically.
    dataset_version: str = Field(pattern=r"^[1-9][0-9]{0,8}$")
    source_description: str = Field(min_length=1)
    rights_note: str = Field(min_length=1)
    license_identifier: str | None = None
    authored_on: date
    generator: str = Field(min_length=1)
    is_synthetic: StrictBool
    transform_version: str
    taxonomy_version: str
    notes: str | None = None
    record_count: int = Field(ge=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_provenance(path: Path) -> Provenance:
    try:
        return Provenance.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise LoaderError(f"cannot read provenance file {path}: {exc}") from exc


@dataclass(frozen=True)
class SourceLine:
    line_number: int
    raw_line: str
    raw_sha256: str
    record: CatalogRecord
    content_sha256: str


@dataclass
class LoadedCatalog:
    path: Path
    checksum_sha256: str
    total_lines: int
    lines: list[SourceLine] = field(default_factory=list)
    outcomes: dict[str, CheckOutcome] = field(default_factory=dict)

    @property
    def records(self) -> list[CatalogRecord]:
        return [line.record for line in self.lines]


def scan_label_fields(raw: dict[str, Any]) -> list[str]:
    return sorted(k for k in raw if k.casefold() in qp.FORBIDDEN_LABEL_FIELDS)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_catalog(path: Path) -> LoadedCatalog:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"cannot read {path}: {exc}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError(f"{path} is not valid UTF-8: {exc}") from exc

    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    catalog = LoadedCatalog(
        path=path, checksum_sha256=hashlib.sha256(data).hexdigest(), total_lines=len(raw_lines)
    )
    schema_findings: list[Finding] = []
    label_findings: list[Finding] = []
    for number, raw_line in enumerate(raw_lines, start=1):
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            schema_findings.append(
                Finding(
                    "schema_validation", Severity.ERROR, f"invalid JSON: {exc.msg}", line=number
                )
            )
            continue
        if isinstance(raw, dict):
            for name in scan_label_fields(raw):
                label_findings.append(
                    Finding(
                        "evaluation_label_fields",
                        Severity.ERROR,
                        f"record contains forbidden label/review field {name!r}",
                        product_id=raw.get("product_id")
                        if isinstance(raw.get("product_id"), str)
                        else None,
                        field=name,
                        line=number,
                    )
                )
        try:
            record = parse_raw(raw)
        except RecordValidationError as exc:
            pid = raw.get("product_id") if isinstance(raw, dict) else None
            schema_findings.extend(
                Finding(
                    "schema_validation",
                    Severity.ERROR,
                    message,
                    product_id=pid if isinstance(pid, str) else None,
                    field=field_name,
                    line=number,
                )
                for field_name, message in exc.errors
            )
            continue
        catalog.lines.append(
            SourceLine(number, raw_line, sha256_text(raw_line), record, record_content_hash(record))
        )

    catalog.outcomes = {
        "schema_validation": CheckOutcome(len(raw_lines), tuple(schema_findings)),
        "evaluation_label_fields": CheckOutcome(len(raw_lines), tuple(label_findings)),
    }
    return catalog


def verify_against_provenance(catalog: LoadedCatalog, provenance: Provenance) -> None:
    """Refuse when the file does not match its provenance (checksum, count, versions)."""
    problems = []
    if catalog.checksum_sha256 != provenance.checksum_sha256:
        problems.append(
            f"checksum mismatch: file {catalog.checksum_sha256} vs provenance "
            f"{provenance.checksum_sha256}"
        )
    if catalog.total_lines != provenance.record_count:
        problems.append(
            f"record count mismatch: file has {catalog.total_lines}, provenance says "
            f"{provenance.record_count}"
        )
    if provenance.transform_version != TRANSFORM_VERSION:
        problems.append(
            f"transform_version {provenance.transform_version!r} != code {TRANSFORM_VERSION!r}"
        )
    if provenance.taxonomy_version != TAXONOMY_VERSION:
        problems.append(
            f"taxonomy_version {provenance.taxonomy_version!r} != code {TAXONOMY_VERSION!r}"
        )
    if problems:
        raise LoaderError("; ".join(problems))
