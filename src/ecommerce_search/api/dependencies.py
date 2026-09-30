from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ecommerce_search.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db_session(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
