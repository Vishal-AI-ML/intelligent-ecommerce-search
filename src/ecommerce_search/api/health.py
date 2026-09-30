from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from ecommerce_search import __version__
from ecommerce_search.api.dependencies import get_db_session, get_settings_dep
from ecommerce_search.api.schemas import ComponentCheck, HealthChecks, HealthResponse
from ecommerce_search.config import Settings
from ecommerce_search.db.checks import CheckResult, check_database, check_pgvector

router = APIRouter()

SERVICE_NAME = "ecommerce-search"


def _to_component(result: CheckResult) -> ComponentCheck:
    return ComponentCheck(
        status="ok" if result.ok else "error",
        latency_ms=result.latency_ms,
        version=result.version,
        detail=result.detail,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A required dependency is down"}},
    summary="Readiness check (database and pgvector)",
)
def health(
    response: Response,
    session: Annotated[Session, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> HealthResponse:
    database = check_database(session)
    if database.ok:
        pgvector = check_pgvector(session)
    else:
        pgvector = CheckResult(ok=False, detail="not checked")

    healthy = database.ok and pgvector.ok
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "unavailable",
        service=SERVICE_NAME,
        version=__version__,
        environment=settings.app_env,
        checks=HealthChecks(database=_to_component(database), pgvector=_to_component(pgvector)),
    )
