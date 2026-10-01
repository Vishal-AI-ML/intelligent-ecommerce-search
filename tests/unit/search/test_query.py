import pytest

from ecommerce_search.search.query import (
    RANK_NORMALIZATION,
    RANK_WEIGHT_ARRAY,
    RANK_WEIGHTS,
    SEARCH_VERSION,
    QueryError,
    normalize_query,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")

MAX = 200


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("laptop", "laptop"),
        ("  hp   laptop \t 8gb\n", "hp laptop 8gb"),
        ("HP Laptop", "HP Laptop"),  # case is left to the FTS configuration
        ('"nike" -shoes', '"nike" -shoes'),  # quotes and hyphens are plain lexical input
        ("in-ear", "in-ear"),
        ("!!!", "!!!"),  # punctuation-only is valid; it simply matches nothing
        ("coding ke liye laptop", "coding ke liye laptop"),
        ("256gb ssd laptop", "256gb ssd laptop"),
    ],
)
def test_valid_queries(raw, expected):
    assert normalize_query(raw, MAX) == expected


@pytest.mark.parametrize("raw", ["", "   ", "\t\n", "  "])
def test_empty_and_whitespace_are_rejected(raw):
    with pytest.raises(QueryError, match="empty"):
        normalize_query(raw, MAX)


@pytest.mark.parametrize("raw", ["lap\x00top", "lap\x07top", "a\x1bb", "x\x7f"])
def test_control_characters_including_nul_are_rejected(raw):
    with pytest.raises(QueryError, match="control"):
        normalize_query(raw, MAX)


def test_length_limit_applies_after_whitespace_normalization():
    assert normalize_query("a" * MAX, MAX) == "a" * MAX
    assert normalize_query("a  " * 50, MAX).count("a") == 50
    with pytest.raises(QueryError, match="at most 200"):
        normalize_query("a" * (MAX + 1), MAX)
    assert normalize_query("a" * 10, 10)
    with pytest.raises(QueryError):
        normalize_query("a" * 11, 10)


def test_versioned_constants_and_initial_weights():
    assert SEARCH_VERSION == "v0_lexical"
    assert RANK_WEIGHTS == {"D": 0.1, "C": 0.2, "B": 0.4, "A": 1.0}
    assert RANK_WEIGHT_ARRAY == [0.1, 0.2, 0.4, 1.0]  # PostgreSQL order {D, C, B, A}
    assert RANK_NORMALIZATION == 0


UNSAFE_UNICODE = {
    "lone high surrogate": "\ud800",
    "lone low surrogate": "\udc00",
    "embedded surrogate": "lap\ud800top",
    "NUL": "lap\x00top",
    "unsafe control character": "lap\x07top",
    "zero-width space": "lap​top",
    "right-to-left override": "‮laptop",
    "byte-order mark": "﻿hp",
    "soft hyphen": "lap­top",
    "left-to-right isolate": "lap⁦top",
}


@pytest.mark.parametrize("raw", UNSAFE_UNICODE.values(), ids=UNSAFE_UNICODE.keys())
def test_invalid_or_invisible_unicode_is_rejected_without_echoing_the_input(raw):
    with pytest.raises(QueryError) as excinfo:
        normalize_query(raw, MAX)
    message = str(excinfo.value)
    assert "surrogate" in message and message.isascii()  # fixed text, never the input
    assert raw not in message


@pytest.mark.parametrize("raw", UNSAFE_UNICODE.values(), ids=UNSAFE_UNICODE.keys())
def test_whatever_is_accepted_is_utf8_encodable(raw):
    # The invariant the database driver needs, including text around the unsafe character.
    for candidate in (raw, "ok " + raw, raw + " ok"):
        try:
            accepted = normalize_query(candidate, MAX)
        except QueryError:
            continue
        accepted.encode("utf-8")


@pytest.mark.parametrize(
    "raw",
    [
        "नमस्ते",  # Devanagari
        "ラップトップ",  # Japanese
        "café naïve",  # Latin with accents
        "می‌خواهم",  # Persian with ZWNJ
        "क्‍ष",  # Devanagari with ZWJ
        "laptop \U0001f4bb",  # an emoji (a symbol, not a format character)
        "hp laptop",  # line separator: ordinary whitespace, collapsed
    ],
)
def test_legitimate_multilingual_text_is_accepted(raw):
    assert normalize_query(raw, MAX).encode("utf-8")


def test_line_separator_is_whitespace_and_joiners_are_kept():
    assert normalize_query("hp laptop", MAX) == "hp laptop"
    assert normalize_query("می‌خ", MAX).count("‌") == 1
