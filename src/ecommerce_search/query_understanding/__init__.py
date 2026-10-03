"""Deterministic query understanding (Milestone 6): pure, typed, informational parsing."""

from ecommerce_search.query_understanding.lexicon import LEXICON_VERSION
from ecommerce_search.query_understanding.models import (
    Ambiguity,
    AmbiguityReason,
    Conflict,
    ConflictReason,
    Intent,
    MatchedTerm,
    MatchRule,
    QueryAttributes,
    QueryUnderstanding,
    SourceSpan,
    UnderstandingField,
)
from ecommerce_search.query_understanding.parser import PARSER_VERSION, parse_normalized_query

__all__ = [
    "LEXICON_VERSION",
    "PARSER_VERSION",
    "Ambiguity",
    "AmbiguityReason",
    "Conflict",
    "ConflictReason",
    "Intent",
    "MatchRule",
    "MatchedTerm",
    "QueryAttributes",
    "QueryUnderstanding",
    "SourceSpan",
    "UnderstandingField",
    "parse_normalized_query",
]
