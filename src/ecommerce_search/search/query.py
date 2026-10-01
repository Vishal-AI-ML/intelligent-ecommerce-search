"""Query normalization and the V0 lexical constants. No parsing of constraints happens here:
every term is plain lexical input (RAM, price, brand and storage are not interpreted)."""

import unicodedata

SEARCH_VERSION = "v0_lexical"

# Initial query-time ts_rank weights, written as PostgreSQL's array order {D, C, B, A}.
# A starting value, not tuned. Normalization flag 0 = no document-length normalization.
RANK_WEIGHTS: dict[str, float] = {"D": 0.1, "C": 0.2, "B": 0.4, "A": 1.0}
RANK_WEIGHT_ARRAY: list[float] = [RANK_WEIGHTS[label] for label in ("D", "C", "B", "A")]
RANK_NORMALIZATION = 0


class QueryError(ValueError):
    """The query text is not acceptable (the API maps this to HTTP 422)."""


# Unicode policy (explicit decision):
# * rejected: surrogate code points (Cs: cannot be encoded as UTF-8 and would reach the database
#   driver as an unhandled error), control characters (Cc, NUL included) that are not ordinary
#   whitespace, and invisible format characters (Cf: zero-width space, bidirectional overrides,
#   byte-order mark, soft hyphen, ...) which can make the displayed query differ from the
#   executed one;
# * allowed: ordinary whitespace (collapsed to single spaces), every letter, digit, mark and
#   symbol of any script, private-use and unassigned code points (plain text, harmless), and
#   the two joiner characters ZWNJ (U+200C) and ZWJ (U+200D), which are part of correct
#   spelling in several scripts (for example Persian and Indic languages).
REJECTED_CATEGORIES = frozenset({"Cs", "Cc", "Cf"})
ALLOWED_JOINERS = frozenset({"\u200c", "\u200d"})  # ZWNJ, ZWJ


def normalize_query(raw: str, max_length: int) -> str:
    """Collapse whitespace; reject empty text, unsafe characters (see policy) and long text.

    Quotes, hyphens and other punctuation are kept: `plainto_tsquery` treats them as plain
    text and never raises on them. Error messages never repeat the input."""
    for char in raw:
        if char.isspace() or char in ALLOWED_JOINERS:
            continue
        if unicodedata.category(char) in REJECTED_CATEGORIES:
            raise QueryError(
                "query must not contain control, surrogate or invisible format characters"
            )
    normalized = " ".join(raw.split())
    if not normalized:
        raise QueryError("query must not be empty or whitespace only")
    if len(normalized) > max_length:
        raise QueryError(f"query must be at most {max_length} characters")
    return normalized
