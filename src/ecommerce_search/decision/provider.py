"""DecisionProvider interface and the deterministic provider (ADR-003, Milestone 6).

A provider receives the normalized query and the deterministic parse and returns a
`DecisionResult`. Only the deterministic provider exists in M6; it returns the parse unchanged.
Confidence, probability, gate and fallback fields are deliberately absent until a later
milestone needs them.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ecommerce_search.query_understanding.models import QueryUnderstanding
from ecommerce_search.query_understanding.parser import parse_normalized_query


class DecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["deterministic"] = "deterministic"
    provider_version: Literal["qu-1"] = "qu-1"
    understanding: QueryUnderstanding


@runtime_checkable
class DecisionProvider(Protocol):
    def understand(self, query: str, deterministic_result: QueryUnderstanding) -> DecisionResult:
        """Return a decision for `query`, given the deterministic parse of the same query."""
        ...


class DeterministicDecisionProvider:
    """Local, network-free provider: the deterministic parse is the decision."""

    def understand(self, query: str, deterministic_result: QueryUnderstanding) -> DecisionResult:
        return DecisionResult(understanding=deterministic_result)


class QueryUnderstandingFailed(Exception):
    """The provider returned an invalid result. The message never contains query text."""

    def __init__(self) -> None:
        super().__init__("query understanding failed")


def understand_normalized_query(
    normalized_query: str, provider: DecisionProvider
) -> DecisionResult:
    """Parse once, call the provider once and validate its result."""
    deterministic_result = parse_normalized_query(normalized_query)
    result = provider.understand(normalized_query, deterministic_result)
    if not isinstance(result, DecisionResult) or result.understanding.raw_query != normalized_query:
        raise QueryUnderstandingFailed()
    return result
