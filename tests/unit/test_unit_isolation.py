"""Regression tests: the non-integration suite is hermetic (no .env, no database password)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ecommerce_search.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[2]
CHILD_FLAG = "ECOMMERCE_SEARCH_HERMETIC_CHILD"


def test_settings_do_not_read_the_local_dotenv_or_ambient_environment():
    # The repository .env (if a developer has one) must not leak into unit tests.
    with pytest.raises(ValidationError):
        Settings()
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.skipif(os.environ.get(CHILD_FLAG) == "1", reason="already inside the hermetic child")
def test_whole_unit_suite_passes_without_a_database_password():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("POSTGRES_", "DB_", "APP_ENV", "LOG_LEVEL"))
    }
    env.update({CHILD_FLAG: "1", "POSTGRES_PASSWORD": ""})
    result = subprocess.run(  # noqa: S603 - fixed argv, sys.executable
        [sys.executable, "-m", "pytest", "-m", "not integration", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-1000:]
