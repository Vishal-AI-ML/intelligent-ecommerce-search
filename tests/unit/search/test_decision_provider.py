"""Milestone 6 DecisionProvider (ADR-003): protocol, deterministic pass-through, validation and
import isolation of the pure packages."""

import importlib
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ecommerce_search.decision import (
    DecisionProvider,
    DecisionResult,
    DeterministicDecisionProvider,
    QueryUnderstandingFailed,
    understand_normalized_query,
)
from ecommerce_search.query_understanding import (
    PARSER_VERSION,
    QueryUnderstanding,
    parse_normalized_query,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")

provider_module = importlib.import_module("ecommerce_search.decision.provider")

QUERY = "hp laptop 8gb 256 ssd under 40k"


class RecordingProvider:
    def __init__(self, result=None) -> None:
        self.calls: list[tuple[str, QueryUnderstanding]] = []
        self.result = result

    def understand(self, query: str, deterministic_result: QueryUnderstanding):
        self.calls.append((query, deterministic_result))
        if self.result is None:
            return DecisionResult(understanding=deterministic_result)
        return self.result


def test_deterministic_provider_conforms_to_the_protocol() -> None:
    assert isinstance(DeterministicDecisionProvider(), DecisionProvider)
    assert isinstance(RecordingProvider(), DecisionProvider)
    assert not isinstance(object(), DecisionProvider)


def test_deterministic_provider_returns_the_parse_unchanged() -> None:
    understanding = parse_normalized_query(QUERY)
    result = DeterministicDecisionProvider().understand(QUERY, understanding)
    assert result.understanding is understanding
    assert (result.provider, result.provider_version) == ("deterministic", PARSER_VERSION)


def test_decision_result_has_no_confidence_or_fallback_fields() -> None:
    assert set(DecisionResult.model_fields) == {"provider", "provider_version", "understanding"}
    understanding = parse_normalized_query(QUERY)
    with pytest.raises(ValidationError):
        DecisionResult(understanding=understanding, confidence=0.9)
    with pytest.raises(ValidationError):
        DecisionResult(understanding=understanding, provider="jev")
    result = DecisionResult(understanding=understanding)
    with pytest.raises(ValidationError):
        result.provider = "deterministic"  # type: ignore[misc]


def test_understand_parses_once_and_calls_the_provider_once(monkeypatch) -> None:
    parse_calls: list[str] = []

    def counting_parse(query: str) -> QueryUnderstanding:
        parse_calls.append(query)
        return parse_normalized_query(query)

    monkeypatch.setattr(provider_module, "parse_normalized_query", counting_parse)
    provider = RecordingProvider()
    result = understand_normalized_query(QUERY, provider)
    assert parse_calls == [QUERY]
    assert len(provider.calls) == 1
    query, deterministic = provider.calls[0]
    assert query == QUERY
    assert result.understanding is deterministic
    assert deterministic.model_dump_json() == parse_normalized_query(QUERY).model_dump_json()


def test_understand_with_the_deterministic_provider() -> None:
    result = understand_normalized_query(QUERY, DeterministicDecisionProvider())
    assert result.understanding.raw_query == QUERY
    assert result.understanding.max_price is not None
    assert result.model_dump_json() == (
        understand_normalized_query(QUERY, DeterministicDecisionProvider()).model_dump_json()
    )


@pytest.mark.parametrize(
    "bad_result",
    [
        None,
        {"provider": "deterministic", "provider_version": "qu-1"},
        "deterministic",
        parse_normalized_query(QUERY),
        DecisionResult(understanding=parse_normalized_query("dell laptop")),
    ],
    ids=["none", "dict", "str", "bare-understanding", "wrong-raw-query"],
)
def test_invalid_provider_results_raise_a_fixed_error(bad_result) -> None:
    class BadProvider:
        def understand(self, query: str, deterministic_result: QueryUnderstanding):
            return bad_result

    with pytest.raises(QueryUnderstandingFailed) as caught:
        understand_normalized_query(QUERY, BadProvider())
    message = str(caught.value)
    assert message == "query understanding failed"
    assert "hp" not in message and "laptop" not in message


def test_provider_exceptions_propagate_unchanged() -> None:
    # The helper validates results only; a raising provider is sanitized by the API boundary
    # (fixed 503), so the helper neither wraps nor swallows it and adds no fallback.
    class Boom(RuntimeError):
        pass

    class RaisingProvider:
        def understand(self, query: str, deterministic_result: QueryUnderstanding):
            raise Boom

    with pytest.raises(Boom):
        understand_normalized_query(QUERY, RaisingProvider())


def test_precondition_errors_propagate_before_the_provider() -> None:
    provider = RecordingProvider()
    with pytest.raises(ValueError):
        understand_normalized_query("hp  laptop", provider)
    assert provider.calls == []


_ISOLATION_PROBE = """
import sys
import ecommerce_search.decision
import ecommerce_search.query_understanding
banned = ("sqlalchemy", "psycopg", "fastapi", "starlette", "httpx", "sentence_transformers",
          "transformers", "torch", "numpy", "alembic", "pgvector", "uvicorn")
internal = ("ecommerce_search.api", "ecommerce_search.config", "ecommerce_search.db",
            "ecommerce_search.embeddings", "ecommerce_search.search", "ecommerce_search.models")
loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in banned or name.startswith(internal)
)
print(",".join(loaded))
"""


def test_pure_packages_import_no_db_web_or_model_stack() -> None:
    done = subprocess.run(  # noqa: S603 - fixed argv, sys.executable
        [sys.executable, "-c", _ISOLATION_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""
