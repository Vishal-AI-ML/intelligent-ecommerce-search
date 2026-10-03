"""Milestone 6 contract: evidence invariants I-1..I-8, determinism, closed enums, frozen models,
versions, lexicon consistency, no logging and the fixed precondition error."""

import logging
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ecommerce_search.catalog.taxonomy import BRAND_LOOKUP, Category
from ecommerce_search.query_understanding import (
    LEXICON_VERSION,
    PARSER_VERSION,
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
    parse_normalized_query,
)
from ecommerce_search.query_understanding.lexicon import (
    CUR,
    GLUE,
    LINK,
    PHRASES,
    SPACE,
    Role,
)
from ecommerce_search.query_understanding.tokenizer import tokenize

pytestmark = pytest.mark.usefixtures("no_socket_connect")

# The contract corpus is defined here, independently of the other test modules, so this file
# collects and imports on its own. It covers the rule-table, tokenizer, numeric and adversarial
# query shapes.
RULE_QUERIES = [
    "hp laptop 8gb 256 ssd under 40k",
    "laptop 8gb 256 ssd",
    "coding laptop",
    "8gb",
    "8 gb",
    "16gb",
    "256gb",
    "512gb",
    "1tb",
    "40k",
    "₹40000",
    "rs 40000",
    "50k ke andar",
    "phone 20k ke under",
    "coding ke liye laptop",
    "gaming ke liye laptop",
    "sasta phone",
    "laptop under 50k",
    "ssd laptop",
    "nvme laptop",
    "smartphones mobile",
    "earbuds earphones headphone",
    "shoe shoes",
    "boat earbuds",
    "one plus phone",
    "hewlett-packard",
    "hewlett packard",
    "notebook",
    "iphone 15",
    "for gaming laptop",
    "student",
    "lightweight",
    "premium",
    "comfort",
    "travel",
    "office",
    "laptop ke liye",
    "for laptop",
    "hdd",
    "ssd",
    "ram",
    "memory storage",
    "8gb ram 256gb ssd",
    "ram 8gb",
    "8gb ssd",
    "ram 8gb ssd",
    "8gb ram storage",
    "storage 512gb ssd",
    "memory 8gb",
    "8gb memory 256gb ssd",
    "8gb ram 16gb",
    "8gb-ram 1tb storage",
    "laptop 16gb",
    "8gb 256gb ssd",
    "8gb laptop",
    "512gb 256gb ssd",
    "8gb 16gb 256gb ssd",
    "8gb ram 16gb 256gb ssd",
    "8gb memory 16gb 256gb ssd",
    "8gb ram storage 16gb 256gb ssd",
    "8gb 256gb ssd 512gb ssd",
    "256 ram",
    "256 memory",
    "i5 ssd",
    "512 ssd",
    "₹256 ssd",
    "₹40,000",
    "rs 40,000",
    "inr 40,000",
    "under 40,000",
    "₹40, 000",
    "40,000",
    "rs 40 000",
    "20k-40k",
    "under 20k-40k",
    "20k – 40k",
    "20k~40k",
    "between 20k and 40k",
    "from 20k to 40k",
    "8/256gb",
    "8 256 ssd",
    "₹ 40,000 laptop under 50k",
    "₹40000 8gb laptop",
    "20k-40k ke andar",
    "between 20k",
    "20k and 40k",
    "laptop 8gb 16gb ssd",
    "8gbssd",
    "8ssd",
    "ram 8ssd",
    "16gaming laptop",
    "5hp",
    "50wala phone",
    "under ₹40kg",
    "rs 500mb",
    "from20k to40k",
    "rs-40000",
    "₹-40000",
    "under-40k",
    "inr 40000",
    "rs.40000",
    "rs. 40000",
    "₹ 40000",
    "₹40k",
    "4k",
    "rs",
    "inr",
    "from 20k",
    "above 20k",
    "over ₹20000",
    "below 30k",
    "upto 30k",
    "up to 30k",
    "within rs 30000",
    "under 50k ke andar",
    "from hp",
    "over ear",
    "under laptop",
    "ke andar",
    "under 40000",
    "40000",
    "above 20k under 40k",
    "above 40k under 40k",
    "above 50k under 40k",
    "under 30k under 40k",
    "from 20k ke andar",
    "laptop above ₹30000 ke under",
    "under 40k 40k ke andar",
    "nvme hdd",
    "nvme nvme",
    "nvme ssd",
    "ssd hdd",
    "hp dell laptop",
    "hp hewlett-packard",
    "gaming coding laptop",
    "laptop phone",
    "nike laptop",
    "ka",
    "wala",
    "xyz wala",
    "50k wala phone",
    "gaming wala laptop",
    "hp ka laptop",
    "256gb ssd wala",
    "nvme wala",
    "ssd hdd wala",
    "wala laptop",
    "laptop-wala",
    "2 lakh",
    "1 crore",
    "50 hazar",
    "40000rs",
    "128gb rom",
    "cheap",
    "sasti",
    "saste",
    "galaxy s24",
    "anc headphones",
    "red shoes size 9",
    "8gb ka ram",
]

ADVERSARIAL_QUERIES = [
    "9" * 200,
    "₹" * 200,
    " ".join(["9k"] * 66),
    "!?.,;:-/" * 25,
    "क" * 200,
    " ".join(["9k"] * 3333),
    " ".join(["laptop 8gb ram 256gb ssd under 50k wala"] * 250),
]

EXTRA_QUERIES = [
    "８ＧＢ ｒａｍ",
    "ＨＰ",
    "ﬁ ß K ½ ² ①",
    "८ जीबी रैम",
    "₹४००००",
    "i5 rtx3050 4k60hz 1.2.3 1.234k",
    "under ₹9999999999.99",
    "under 9999999.99k",
    "under 10000000k",
    "above 0k ₹0",
    "4097gb ram 5tb ram 65tb ssd",
    "64tb ssd 4096gb ram",
    "under ₹10000000000",
    "!!!",
    "a‍b laptop",
    "hp😀laptop",
    "laptop ＳＳＤ under ５０ｋ",
    "ram 8gb ssd wala",
    "nvme hdd wala ka",
    "8gb ram 256gb ssd under 50k ke andar gaming ke liye hp laptop sasta",
    "from 20k ke andar wala phone",
    "above 10k ke andar under 20k",
]

CORPUS = sorted(set(RULE_QUERIES) | set(EXTRA_QUERIES) | set(ADVERSARIAL_QUERIES))
STORAGE_TYPE_RULES = {
    MatchRule.STORAGE_TYPE_TERM,
    MatchRule.STORAGE_INTERFACE_TERM,
    MatchRule.STORAGE_TYPE_IMPLIED_BY_INTERFACE,
}


def _field_values(understanding: QueryUnderstanding) -> dict[str, object]:
    data = understanding.model_dump(mode="json")
    return {
        field.value: {**data, **data["attributes"]}[field.value] for field in UnderstandingField
    }


BOUND_RULES = {MatchRule.PRICE_LOWER_BOUND, MatchRule.PRICE_UPPER_BOUND}


Evidence = tuple[SourceSpan, MatchRule | None, str | None]


def _evidence(understanding: QueryUnderstanding) -> list[Evidence]:
    items: list[Evidence] = []
    items += [(term.span, term.rule, term.value) for term in understanding.matched_terms]
    items += [(ambiguity.span, None, ambiguity.value) for ambiguity in understanding.ambiguities]
    items += [
        (t.span, t.rule, t.value)
        for conflict in understanding.conflicts
        for t in conflict.candidates
    ]
    return items


def _dual_bound(a: Evidence, b: Evidence) -> bool:
    """One price construction with a lower prefix and an upper postfix marker (`from 20k ke
    andar`): one lower and one upper bound of equal value whose spans share exactly the price,
    the lower span starting first and the upper span ending last."""
    if {a[1], b[1]} != BOUND_RULES:
        return False
    lower, upper = (a, b) if a[1] is MatchRule.PRICE_LOWER_BOUND else (b, a)
    return lower[2] == upper[2] and lower[0].start < upper[0].start < lower[0].end < upper[0].end


@pytest.fixture(scope="module")
def parsed() -> list[QueryUnderstanding]:
    return [parse_normalized_query(query) for query in CORPUS]


def test_corpus_is_large_and_unique() -> None:
    assert len(CORPUS) >= 150


def test_i1_every_set_field_has_matching_evidence(parsed) -> None:
    for understanding in parsed:
        values = _field_values(understanding)
        for name, value in values.items():
            if value is None:
                continue
            assert any(
                term.field.value == name and term.value == str(value)
                for term in understanding.matched_terms
            ), (understanding.raw_query, name)


def test_i2_conflicted_fields_are_null_and_candidates_are_not_matched(parsed) -> None:
    for understanding in parsed:
        values = _field_values(understanding)
        for conflict in understanding.conflicts:
            assert values[conflict.field.value] is None
            assert all(term.field is conflict.field for term in conflict.candidates)
            assert not any(t.field is conflict.field for t in understanding.matched_terms)


def test_i3_every_token_is_evidence_or_exactly_one_unresolved_span(parsed) -> None:
    for understanding in parsed:
        evidence = [span for span, _, _ in _evidence(understanding)]
        for token in tokenize(understanding.raw_query):
            covered = any(s.start <= token.start and token.end <= s.end for s in evidence)
            unresolved = [
                s for s in understanding.unresolved if s.start < token.end and token.start < s.end
            ]
            assert covered != bool(unresolved), (understanding.raw_query, token.text)
            assert len(unresolved) <= 1
            if unresolved:
                assert (unresolved[0].start, unresolved[0].end) == (token.start, token.end)


def test_i4_overlapping_evidence_is_one_construction_or_a_shared_storage_anchor(parsed) -> None:
    for understanding in parsed:
        items = _evidence(understanding)
        for i, item_a in enumerate(items):
            for item_b in items[i + 1 :]:
                (a, rule_a, _), (b, rule_b, _) = item_a, item_b
                if a.end <= b.start or b.end <= a.start:
                    continue
                same_span = (a.start, a.end) == (b.start, b.end)
                anchor = (
                    rule_a in STORAGE_TYPE_RULES and b.start <= a.start and a.end <= b.end
                ) or (rule_b in STORAGE_TYPE_RULES and a.start <= b.start and b.end <= a.end)
                dual = _dual_bound(item_a, item_b)
                assert same_span or anchor or dual, (understanding.raw_query, a.text, b.text)


def test_i4_dual_bound_price_is_one_construction() -> None:
    understanding = parse_normalized_query("from 20k ke andar")
    lower, upper = _evidence(understanding)
    assert _dual_bound(lower, upper)
    assert (lower[0].text, upper[0].text) == ("from 20k", "20k ke andar")
    assert understanding.min_price == understanding.max_price == Decimal("20000.00")
    assert understanding.conflicts == () and understanding.unresolved == ()
    # Two separate, disjoint prices are not one construction.
    assert not _dual_bound(*_evidence(parse_normalized_query("above 20k under 20k")))


@pytest.mark.parametrize(
    ("query", "spans"),
    [
        (
            "nvme",
            {("storage_interface_term", "nvme"), ("storage_type_implied_by_interface", "nvme")},
        ),
        ("256gb ssd", {("storage_capacity", "256gb ssd"), ("storage_type_term", "ssd")}),
    ],
)
def test_i4_legitimate_multi_field_evidence_is_kept(query: str, spans: set) -> None:
    understanding = parse_normalized_query(query)
    assert {(t.rule.value, t.span.text) for t in understanding.matched_terms} == spans


def test_i5_unresolved_spans_are_ordered_and_disjoint(parsed) -> None:
    for understanding in parsed:
        spans = understanding.unresolved
        assert all(a.end <= b.start for a, b in zip(spans, spans[1:], strict=False))


def test_i6_every_span_text_is_the_raw_query_slice(parsed) -> None:
    for understanding in parsed:
        raw = understanding.raw_query
        spans = [span for span, _, _ in _evidence(understanding)] + list(understanding.unresolved)
        for span in spans:
            assert 0 <= span.start < span.end <= len(raw)
            assert raw[span.start : span.end] == span.text


def test_i7_collections_follow_the_declared_ordering(parsed) -> None:
    for u in parsed:
        terms = [(t.span.start, t.span.end, t.field.value, t.rule.value) for t in u.matched_terms]
        assert terms == sorted(terms)
        ambiguities = [(a.span.start, a.span.end, a.reason.value) for a in u.ambiguities]
        assert ambiguities == sorted(ambiguities)
        conflicts = [(c.field.value, c.reason.value) for c in u.conflicts]
        assert conflicts == sorted(conflicts)
        for conflict in u.conflicts:
            keys = [
                (t.span.start, t.span.end, t.field.value, t.rule.value) for t in conflict.candidates
            ]
            assert keys == sorted(keys)
        starts = [span.start for span in u.unresolved]
        assert starts == sorted(starts)


def test_i7_repeated_parses_are_byte_identical(parsed) -> None:
    for understanding in parsed:
        again = parse_normalized_query(understanding.raw_query)
        assert again.model_dump_json() == understanding.model_dump_json()


def test_i8_extra_fields_are_forbidden_on_every_model() -> None:
    span = {"start": 0, "end": 1, "text": "x"}
    term = {"rule": "brand_term", "field": "brand", "value": "HP", "span": span}
    cases = [
        (SourceSpan, span),
        (MatchedTerm, term),
        (Ambiguity, {"reason": "bare_capacity", "span": span, "value": "8"}),
        (Conflict, {"field": "brand", "reason": "repeated_different_values", "candidates": [term]}),
        (QueryAttributes, {}),
        (QueryUnderstanding, {"raw_query": "x"}),
    ]
    for model, data in cases:
        model.model_validate(data)
        with pytest.raises(ValidationError):
            model.model_validate({**data, "extra": 1})
    nested = [
        {"raw_query": "x", "attributes": {"extra": 1}},
        {"raw_query": "x", "unresolved": [{**span, "extra": 1}]},
        {"raw_query": "x", "matched_terms": [{**term, "extra": 1}]},
        {"raw_query": "x", "matched_terms": [{**term, "span": {**span, "extra": 1}}]},
    ]
    for data in nested:
        with pytest.raises(ValidationError):
            QueryUnderstanding.model_validate(data)


def test_models_are_frozen() -> None:
    understanding = parse_normalized_query("hp laptop 8gb ram")
    with pytest.raises(ValidationError):
        understanding.brand = "Dell"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        understanding.matched_terms[0].value = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        understanding.matched_terms[0].span.start = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        understanding.attributes.price_preference = "low"  # type: ignore[misc]
    assert isinstance(understanding.matched_terms, tuple)
    assert isinstance(understanding.unresolved, tuple)


def test_json_round_trip(parsed) -> None:
    for understanding in parsed:
        restored = QueryUnderstanding.model_validate_json(understanding.model_dump_json())
        assert restored == understanding


def test_span_validators() -> None:
    with pytest.raises(ValidationError):
        SourceSpan(start=-1, end=1, text="x")
    with pytest.raises(ValidationError):
        SourceSpan(start=1, end=1, text="")
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="ab", unresolved=(SourceSpan(start=0, end=1, text="b"),))
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="ab", unresolved=(SourceSpan(start=1, end=3, text="b"),))
    with pytest.raises(ValidationError):
        Conflict(field="brand", reason="repeated_different_values", candidates=())


def test_versions_and_retrieval_strategy(parsed) -> None:
    assert PARSER_VERSION == "qu-1"
    assert LEXICON_VERSION == "1"
    for understanding in parsed:
        assert understanding.parser_version == PARSER_VERSION
        assert understanding.lexicon_version == LEXICON_VERSION
        assert understanding.retrieval_strategy is None
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="x", parser_version="qu-2")
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="x", retrieval_strategy="dense")


def test_closed_enums_match_the_plan() -> None:
    assert [f.value for f in UnderstandingField] == [
        "category",
        "brand",
        "ram_gb",
        "storage_gb",
        "storage_type",
        "storage_interface",
        "min_price",
        "max_price",
        "semantic_intent",
        "price_preference",
    ]
    assert {r.value for r in MatchRule} == {
        "category_term",
        "brand_term",
        "intent_term",
        "storage_type_term",
        "storage_interface_term",
        "storage_type_implied_by_interface",
        "price_preference_term",
        "ram_capacity",
        "ram_capacity_paired",
        "storage_capacity",
        "storage_capacity_implicit_gb",
        "price_upper_bound",
        "price_lower_bound",
    }
    assert {r.value for r in AmbiguityReason} == {
        "bare_capacity",
        "ambiguous_memory_capacity",
        "capacity_multiple_roles",
        "price_without_bound",
        "unsupported_grouped_number",
        "unsupported_range",
        "unsupported_numeric_compound",
        "out_of_range_value",
    }
    assert {r.value for r in ConflictReason} == {
        "repeated_different_values",
        "min_price_exceeds_max_price",
        "dependent_on_conflicted_storage_type",
    }
    assert {i.value for i in Intent} == {
        "gaming",
        "coding",
        "student",
        "lightweight",
        "premium",
        "comfort",
        "travel",
        "office",
    }


def test_nike_laptop_keeps_both_values_without_conflict() -> None:
    understanding = parse_normalized_query("nike laptop")
    assert (understanding.brand, understanding.category) == ("Nike", Category.LAPTOP)
    assert understanding.conflicts == ()
    assert understanding.ambiguities == ()


def test_lexicon_categories_and_brands_come_from_the_taxonomy() -> None:
    categories = {p.value for p in PHRASES if p.role is Role.CATEGORY}
    assert categories == {c.value for c in Category}
    brand_phrases = [p for p in PHRASES if p.role is Role.BRAND]
    assert {p.value for p in brand_phrases} <= set(BRAND_LOOKUP.values())
    assert len(brand_phrases) == len(BRAND_LOOKUP)
    category_words = {p.words for p in PHRASES if p.role is Role.CATEGORY}
    assert ("notebook",) not in category_words
    assert ("iphone",) not in {p.words for p in brand_phrases}


def test_exact_brand_gaps_match_the_dictionary_keys() -> None:
    for phrase in (p for p in PHRASES if p.role is Role.BRAND):
        key = phrase.words[0] + "".join(
            g + w for g, w in zip(phrase.gaps, phrase.words[1:], strict=True)
        )
        assert BRAND_LOOKUP[key] == phrase.value
        tokens = tokenize(key)
        assert tuple(t.folded for t in tokens) == phrase.words
        assert tuple(t.gap_before for t in tokens[1:]) == phrase.gaps
    assert any(p.gaps == ("-",) and p.words == ("hewlett", "packard") for p in PHRASES)
    assert any(p.gaps == (" ",) and p.words == ("one", "plus") for p in PHRASES)


def test_no_phrase_has_two_roles() -> None:
    seen: dict[tuple, Role] = {}
    for phrase in PHRASES:
        key = (phrase.words, phrase.gaps)
        assert seen.setdefault(key, phrase.role) is phrase.role, key
    assert len(seen) == len(PHRASES)


def test_gap_sets_are_exact() -> None:
    assert frozenset({"", " "}) == GLUE
    assert frozenset({" "}) == SPACE
    assert frozenset({" ", "-", ",", ", "}) == LINK
    assert frozenset({" ", ".", ". "}) == CUR


def test_parser_emits_no_log_records(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    for query in CORPUS:
        parse_normalized_query(query)
    assert caplog.records == []


@pytest.mark.parametrize(
    "query",
    ["", " ", " secret laptop", "secret laptop ", "secret  laptop", "secret\tlaptop", "a\nb"],
)
def test_precondition_error_is_fixed_and_never_echoes_input(query: str) -> None:
    with pytest.raises(ValueError) as caught:
        parse_normalized_query(query)
    message = str(caught.value)
    assert message == "query understanding requires a non-empty, whitespace-normalized query"
    assert "secret" not in message


def test_price_fields_are_decimals(parsed) -> None:
    for understanding in parsed:
        for value in (understanding.min_price, understanding.max_price):
            assert value is None or (isinstance(value, Decimal) and value.as_tuple().exponent == -2)
