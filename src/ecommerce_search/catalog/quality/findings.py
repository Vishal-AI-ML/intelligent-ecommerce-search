from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    message: str
    product_id: str | None = None
    field: str | None = None
    line: int | None = None
    count: int | None = None  # aggregated (info) findings

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "product_id": self.product_id,
            "line": self.line,
            "field": self.field,
            "message": self.message,
            "count": self.count,
        }

    def sort_key(self) -> tuple:
        return (self.check, self.product_id or "", self.line or 0, self.field or "", self.message)


@dataclass(frozen=True)
class CheckOutcome:
    """Result of one check: how many records it evaluated and what it found."""

    evaluated: int
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    note: str | None = None
