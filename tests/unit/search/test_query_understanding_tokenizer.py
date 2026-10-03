"""Milestone 6 tokenizer, source spans, gaps, Unicode visibility and structural bounds.

Spans are zero-based code-point offsets into the raw query with an exclusive end. No timing is
asserted anywhere: complexity is checked structurally with deterministic counts.
"""

import importlib
from decimal import Decimal

import pytest

from ecommerce_search.query_understanding import parse_normalized_query
from ecommerce_search.query_understanding.lexicon import MAX_PHRASE_TOKENS, PHRASES
from ecommerce_search.query_understanding.tokenizer import TokenKind, tokenize

pytestmark = pytest.mark.usefixtures("no_socket_connect")

parser_module = importlib.import_module("ecommerce_search.query_understanding.parser")


def _tokens(query: str) -> list[tuple[str, str, int, int, str]]:
    return [(t.kind.value, t.text, t.start, t.end, t.gap_before) for t in tokenize(query)]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "８ＧＢ ｒａｍ",
            [("NUMBER", "８", 0, 1, ""), ("WORD", "ＧＢ", 1, 3, ""), ("WORD", "ｒａｍ", 4, 7, " ")],
        ),
        ("ＨＰ", [("WORD", "ＨＰ", 0, 2, "")]),
        (
            "8gb-ram",
            [("NUMBER", "8", 0, 1, ""), ("WORD", "gb", 1, 3, ""), ("WORD", "ram", 4, 7, "-")],
        ),
        (
            "8gb,ram",
            [("NUMBER", "8", 0, 1, ""), ("WORD", "gb", 1, 3, ""), ("WORD", "ram", 4, 7, ",")],
        ),
        ("rs.40000", [("WORD", "rs", 0, 2, ""), ("NUMBER", "40000", 3, 8, ".")]),
        ("₹40000", [("CURRENCY", "₹", 0, 1, ""), ("NUMBER", "40000", 1, 6, "")]),
        (
            "₹ 40, 000",
            [
                ("CURRENCY", "₹", 0, 1, ""),
                ("NUMBER", "40", 2, 4, " "),
                ("NUMBER", "000", 6, 9, ", "),
            ],
        ),
        ("１２．５", [("NUMBER", "１２．５", 0, 4, "")]),
        ("8.", [("NUMBER", "8", 0, 1, "")]),
        (".5", [("NUMBER", "5", 1, 2, ".")]),
        (
            "20k – 40k",
            [
                ("NUMBER", "20", 0, 2, ""),
                ("WORD", "k", 2, 3, ""),
                ("NUMBER", "40", 6, 8, " – "),
                ("WORD", "k", 8, 9, ""),
            ],
        ),
        ("ﬁ ß", [("UNRESOLVED", "ﬁ", 0, 1, ""), ("UNRESOLVED", "ß", 2, 3, " ")]),
        ("hp😀laptop", [("WORD", "hp", 0, 2, ""), ("WORD", "laptop", 3, 9, "😀")]),
        ("a‍b", [("UNRESOLVED", "a‍b", 0, 3, "")]),
        ("!!!", []),
    ],
)
def test_exact_tokens_offsets_and_gaps(query: str, expected: list[tuple]) -> None:
    assert _tokens(query) == expected


@pytest.mark.parametrize(
    "query", ["i5", "rtx3050", "4k60hz", "1.2.3", "1.234k", "12345678901", "1.234", "hpश", "८"]
)
def test_unsupported_run_shapes_are_one_unresolved_token(query: str) -> None:
    (token,) = tokenize(query)
    assert (token.kind, token.start, token.end, token.text) == (
        TokenKind.UNRESOLVED,
        0,
        len(query),
        query,
    )
    understanding = parse_normalized_query(query)
    assert [(s.start, s.end, s.text) for s in understanding.unresolved] == [(0, len(query), query)]
    assert understanding.matched_terms == ()
    assert understanding.ambiguities == ()


def test_8gbssd_is_an_unresolved_number_and_word_that_sets_nothing() -> None:
    assert _tokens("8gbssd") == [("NUMBER", "8", 0, 1, ""), ("WORD", "gbssd", 1, 6, "")]
    understanding = parse_normalized_query("8gbssd")
    assert [(s.start, s.end, s.text) for s in understanding.unresolved] == [
        (0, 1, "8"),
        (1, 6, "gbssd"),
    ]
    assert understanding.storage_gb is None
    assert understanding.storage_type is None
    assert understanding.matched_terms == () and understanding.ambiguities == ()


@pytest.mark.parametrize(
    ("query", "unresolved"),
    [
        ("from20k to40k", [(0, 7, "from20k"), (8, 13, "to40k")]),
        ("rs-40000", [(0, 2, "rs"), (3, 8, "40000")]),
    ],
)
def test_false_positive_forms_set_no_price(query: str, unresolved: list[tuple]) -> None:
    understanding = parse_normalized_query(query)
    assert [(s.start, s.end, s.text) for s in understanding.unresolved] == unresolved
    assert understanding.min_price is None and understanding.max_price is None
    assert understanding.matched_terms == () and understanding.ambiguities == ()


@pytest.mark.parametrize(
    ("query", "field", "value", "span"),
    [
        ("８ＧＢ ｒａｍ", "ram_gb", "8", (0, 7, "８ＧＢ ｒａｍ")),
        ("ＨＰ", "brand", "HP", (0, 2, "ＨＰ")),
        ("8gb-ram", "ram_gb", "8", (0, 7, "8gb-ram")),
        ("8gb,ram", "ram_gb", "8", (0, 7, "8gb,ram")),
        ("50k ke andar", "max_price", "50000.00", (0, 12, "50k ke andar")),
        ("one plus", "brand", "OnePlus", (0, 8, "one plus")),
        ("hewlett-packard", "brand", "HP", (0, 15, "hewlett-packard")),
        ("coding ke liye", "semantic_intent", "coding", (0, 14, "coding ke liye")),
        ("under 50k ke andar", "max_price", "50000.00", (0, 18, "under 50k ke andar")),
        ("laptop ＳＳＤ", "storage_type", "SSD", (7, 10, "ＳＳＤ")),
    ],
)
def test_exact_evidence_spans(query: str, field: str, value: str, span: tuple) -> None:
    understanding = parse_normalized_query(query)
    (term,) = [t for t in understanding.matched_terms if t.field.value == field]
    assert term.value == value
    assert (term.span.start, term.span.end, term.span.text) == span
    assert understanding.unresolved == ()


def test_rs_dot_price_span_covers_the_gap() -> None:
    (ambiguity,) = parse_normalized_query("rs.40000").ambiguities
    assert (ambiguity.reason.value, ambiguity.value) == ("price_without_bound", "40000.00")
    assert (ambiguity.span.start, ambiguity.span.end, ambiguity.span.text) == (0, 8, "rs.40000")


@pytest.mark.parametrize("query", ["ﬁ", "ß", "K", "½", "²", "①", "Ⅷ"])
def test_unfolded_unicode_stays_unresolved(query: str) -> None:
    understanding = parse_normalized_query(f"laptop {query}")
    assert [(s.start, s.end, s.text) for s in understanding.unresolved] == [(7, 8, query)]
    assert [t.field.value for t in understanding.matched_terms] == ["category"]


@pytest.mark.parametrize(
    "query", ["८ जीबी रैम", "हिंदी लैपटॉप", "₹४००००", "४०k", "laptop ८gb ram", "रैम ८"]
)
def test_devanagari_digits_and_hindi_text_are_unresolved_and_set_nothing(query: str) -> None:
    understanding = parse_normalized_query(query)
    data = understanding.model_dump(mode="json")
    numeric = ("ram_gb", "storage_gb", "min_price", "max_price")
    assert all(data[name] is None for name in numeric)
    texts = [span.text for span in understanding.unresolved]
    for chunk in query.split(" "):
        if chunk not in ("laptop", "ram"):
            assert any(ch in "".join(texts) for ch in chunk)
    assert all(ord(ch) < 0x0900 or ch in "".join(texts) for ch in query if ch != " ")


def test_only_validated_ascii_numbers_reach_int_and_decimal(monkeypatch) -> None:
    calls: list[str] = []
    real_int = int

    def spy_int(value, *args):
        calls.append(value)
        return real_int(value, *args)

    def spy_decimal(value="0", *args):
        calls.append(value)
        return Decimal(value, *args)

    monkeypatch.setattr(parser_module, "int", spy_int, raising=False)
    monkeypatch.setattr(parser_module, "Decimal", spy_decimal)
    rejected = [
        "₹10000000000 laptop",
        "10000000000gb ram",
        "12345678901k",
        "１２３４５６７８９０１k",
        "1.234k",
        "₹1.234",
        "rs 0.001",
        "८gb ram",
        "₹४००००",
        "४०k",
        "८ ssd",
        "under ₹40kg",
        "rs 500mb",
    ]
    for query in rejected:
        parse_normalized_query(query)
    assert calls == []

    parse_normalized_query("８ＧＢ ｒａｍ ₹４０ｋ 256 ssd")
    assert sorted(calls) == ["256", "40", "8"]
    assert all(isinstance(value, str) and value.isascii() for value in calls)


ADVERSARIAL = {
    "digits": "9" * 200,
    "rupees": "₹" * 200,
    "prices": " ".join(["9k"] * 66),
    "punctuation": "!?.,;:-/" * 25,
    "devanagari": "क" * 200,
    "long_prices": " ".join(["9k"] * 3333),
    "long_mixed": " ".join(["laptop 8gb ram 256gb ssd under 50k wala"] * 250),
}

EXPECTED_COUNTS = {
    # name: (tokens, matched terms + ambiguities + conflict candidates, unresolved)
    "digits": (1, 0, 1),
    "rupees": (200, 0, 200),
    "prices": (132, 66, 0),
    "punctuation": (0, 0, 0),
    "devanagari": (1, 0, 1),
    "long_prices": (6666, 3333, 0),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_structural_bounds_on_adversarial_inputs(name: str) -> None:
    query = ADVERSARIAL[name]
    assert len(query) <= 10_000
    tokens = tokenize(query)
    understanding = parse_normalized_query(query)
    evidence = (
        len(understanding.matched_terms)
        + len(understanding.ambiguities)
        + sum(len(c.candidates) for c in understanding.conflicts)
    )
    assert len(tokens) <= len(query)
    assert evidence <= len(tokens)
    assert len(understanding.unresolved) <= len(tokens)
    if name in EXPECTED_COUNTS:
        assert (len(tokens), evidence, len(understanding.unresolved)) == EXPECTED_COUNTS[name]


def test_lexicon_phrases_are_at_most_three_tokens() -> None:
    assert MAX_PHRASE_TOKENS == 3
    assert max(len(phrase.words) for phrase in PHRASES) <= MAX_PHRASE_TOKENS
