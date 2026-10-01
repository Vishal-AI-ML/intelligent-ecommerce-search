import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ecommerce_search import __version__
from ecommerce_search.api.health import router as health_router
from ecommerce_search.api.search import router as search_router
from ecommerce_search.config import Settings, get_settings
from ecommerce_search.db.engine import create_db_engine, create_session_factory

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        engine = create_db_engine(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        logger.info(
            "starting ecommerce-search %s env=%s db=%s:%s",
            __version__,
            settings.app_env,
            settings.postgres_host,
            settings.postgres_port,
        )
        try:
            yield
        finally:
            engine.dispose()
            logger.info("engine disposed")

    app = FastAPI(title="Intelligent E-commerce Search", version=__version__, lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(search_router)
    return app
