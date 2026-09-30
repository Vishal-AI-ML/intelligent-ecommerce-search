"""Dependency checks. Failures return generic details; exception text is never exposed."""

import logging
import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None
    version: str | None = None


def check_database(session: Session) -> CheckResult:
    start = time.perf_counter()
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("database check failed: %s", type(exc).__name__)
        return CheckResult(ok=False, detail="database unreachable")
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return CheckResult(ok=True, latency_ms=latency_ms)


def check_pgvector(session: Session) -> CheckResult:
    try:
        version = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.error("pgvector check failed: %s", type(exc).__name__)
        return CheckResult(ok=False, detail="pgvector check failed")
    if version is None:
        return CheckResult(ok=False, detail="extension not installed")
    return CheckResult(ok=True, version=str(version))
