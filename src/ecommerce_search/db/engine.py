from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ecommerce_search.config import Settings


def create_db_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url(),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        connect_args={
            "connect_timeout": settings.db_connect_timeout_seconds,
            # Value is a validated int, so nothing user-controlled reaches the options string.
            "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
        },
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
