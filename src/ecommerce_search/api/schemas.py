from typing import Literal

from pydantic import BaseModel


class ComponentCheck(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float | None = None
    version: str | None = None
    detail: str | None = None


class HealthChecks(BaseModel):
    database: ComponentCheck
    pgvector: ComponentCheck


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    service: str
    version: str
    environment: str
    checks: HealthChecks
