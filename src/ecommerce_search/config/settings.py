from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    """Typed application configuration read from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "ecommerce_search"
    postgres_user: str = "ecommerce_search"
    postgres_password: SecretStr = Field(min_length=1)

    db_connect_timeout_seconds: int = Field(default=3, ge=1)
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_timeout_seconds: int = Field(default=10, ge=1)
    # PostgreSQL accepts statement_timeout as int32 milliseconds; 0 (disabled) is not allowed.
    db_statement_timeout_ms: int = Field(default=30_000, ge=1, le=2_147_483_647)

    # Lexical search (Milestone 3). `search_lexical_k` is the maximum accepted `top_k`.
    search_lexical_k: int = Field(default=50, ge=1, le=1000)
    search_default_top_k: int = Field(default=10, ge=1, le=1000)
    search_max_query_length: int = Field(default=200, ge=1, le=1000)

    @model_validator(mode="after")
    def _default_top_k_within_limit(self) -> "Settings":
        if self.search_default_top_k > self.search_lexical_k:
            raise ValueError("search_default_top_k must not exceed search_lexical_k")
        return self

    def database_url(self, database: str | None = None) -> URL:
        """SQLAlchemy URL for the configured server (optionally another database)."""
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=database or self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
