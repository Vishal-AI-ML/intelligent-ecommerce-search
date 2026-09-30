import pytest

from ecommerce_search.config import Settings

TEST_PASSWORD = "unit-test-password"  # noqa: S105 - test fixture value, not a real credential


@pytest.fixture
def make_settings():
    def _make(**overrides) -> Settings:
        values = {"postgres_password": TEST_PASSWORD, "app_env": "test"}
        values.update(overrides)
        # _env_file=None keeps a developer's local .env from leaking into tests.
        return Settings(_env_file=None, **values)

    return _make
