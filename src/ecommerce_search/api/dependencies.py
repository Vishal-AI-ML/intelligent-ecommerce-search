from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ecommerce_search.config import Settings
from ecommerce_search.decision import DecisionProvider
from ecommerce_search.embeddings.provider import Embedder


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db_session(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_embedder(request: Request) -> Embedder | None:
    """The application's lazily loaded embedding provider (None when no models dir is known)."""
    return request.app.state.embedder


def get_decision_provider(request: Request) -> DecisionProvider:
    """The application's single decision provider (used by /search/hybrid and /search/filtered)."""
    return request.app.state.decision_provider
