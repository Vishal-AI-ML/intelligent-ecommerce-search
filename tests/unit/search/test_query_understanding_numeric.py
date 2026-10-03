"""Milestone 6 numeric bounds, canonical forms and field validators (A17).

Authoritative bounds (docs/data-quality.md §3.1): price (0, 9999999999.99] with 2 decimals,
RAM 1-4096 GB, storage 1-65536 GB (1 TB = 1024 GB).
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from ecommerce_search.query_understanding import QueryUnderstanding, parse_normalized_query

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def _evidence(query: str) -> tuple[dict, list, list, list]:
    understanding = parse_normalized_query(query)
    data = understanding.model_dump(mode="json")
    fields = {
        name: data[name]
        for name in ("ram_gb", "storage_gb", "min_price", "max_price")
        if data[name] is not None
    }
    terms = [(t.rule.value, t.value, t.span.text) for t in understanding.matched_terms]
    ambiguities = [(a.reason.value, a.value, a.span.text) for a in understanding.ambiguities]
    unresolved = [span.text for span in understanding.unresolved]
    return fields, terms, ambiguities, unresolved


@pytest.mark.parametrize(
    ("query", "fields", "ambiguities", "unresolved"),
    [
        # Price upper bound, on / one cent above / one cent below.
        ("under ₹9999999999.99", {"max_price": "9999999999.99"}, [], []),
        ("under rs 9999999999.99", {"max_price": "9999999999.99"}, [], []),
        ("under ₹9999999999.98", {"max_price": "9999999999.98"}, [], []),
        ("under ₹10000000000", {}, [], ["under", "₹", "10000000000"]),
        ("₹9999999999.99", {}, [("price_without_bound", "9999999999.99", "₹9999999999.99")], []),
        ("under 9999999.99k", {"max_price": "9999999990.00"}, [], []),
        ("under 9999999.98k", {"max_price": "9999999980.00"}, [], []),
        (
            "under 10000000k",
            {},
            [("out_of_range_value", "10000000000.00", "10000000k")],
            ["under"],
        ),
        # Price lower bound: zero is out of range, one cent is the smallest valid price.
        ("above 0k", {}, [("out_of_range_value", "0.00", "0k")], ["above"]),
        ("above ₹0", {}, [("out_of_range_value", "0.00", "₹0")], ["above"]),
        ("above ₹0.00", {}, [("out_of_range_value", "0.00", "₹0.00")], ["above"]),
        ("above ₹0.01", {"min_price": "0.01"}, [], []),
        ("above 0.01k", {"min_price": "10.00"}, [], []),
        # 3 decimals and 11 integer digits are never numbers.
        ("under ₹0.001", {}, [], ["under", "₹", "0.001"]),
        ("under 12345678901k", {}, [], ["under", "12345678901k"]),
        # RAM 1-4096.
        ("4096gb ram", {"ram_gb": 4096}, [], []),
        ("4095gb ram", {"ram_gb": 4095}, [], []),
        ("4097gb ram", {}, [("out_of_range_value", "4097", "4097gb")], ["ram"]),
        ("1gb ram", {"ram_gb": 1}, [], []),
        ("0gb ram", {}, [("out_of_range_value", "0", "0gb")], ["ram"]),
        ("4tb ram", {"ram_gb": 4096}, [], []),
        ("5tb ram", {}, [("out_of_range_value", "5120", "5tb")], ["ram"]),
        # Storage 1-65536.
        ("64tb ssd", {"storage_gb": 65536}, [], []),
        ("65535gb storage", {"storage_gb": 65535}, [], []),
        ("65537gb storage", {}, [("out_of_range_value", "65537", "65537gb")], ["storage"]),
        ("65tb ssd", {}, [("out_of_range_value", "66560", "65tb")], []),
        ("0 ssd", {}, [("out_of_range_value", "0", "0")], []),
        ("65537 ssd", {}, [("out_of_range_value", "65537", "65537")], []),
        ("65536 ssd", {"storage_gb": 65536}, [], []),
        # Bare capacity range 1-65536.
        ("65536gb", {}, [("bare_capacity", "65536", "65536gb")], []),
        ("65537gb", {}, [("out_of_range_value", "65537", "65537gb")], []),
        ("0gb", {}, [("out_of_range_value", "0", "0gb")], []),
        # Decimal capacities are not capacities.
        ("1.5tb", {}, [], ["1.5", "tb"]),
        ("2.5 ssd", {}, [], ["2.5"]),
    ],
)
def test_bounds_and_out_of_range_values(
    query: str, fields: dict, ambiguities: list, unresolved: list
) -> None:
    got_fields, _, got_ambiguities, got_unresolved = _evidence(query)
    assert got_fields == fields
    assert got_ambiguities == ambiguities
    assert got_unresolved == unresolved


@pytest.mark.parametrize(
    ("query", "rule", "value"),
    [
        ("under 40k", "price_upper_bound", "40000.00"),
        ("under ₹40000", "price_upper_bound", "40000.00"),
        ("under ₹40000.5", "price_upper_bound", "40000.50"),
        ("under 40.5k", "price_upper_bound", "40500.00"),
        ("under 0040k", "price_upper_bound", "40000.00"),
        ("from rs 1", "price_lower_bound", "1.00"),
        ("1tb ssd", "storage_capacity", "1024"),
        ("08gb ram", "ram_capacity", "8"),
        ("256 ssd", "storage_capacity_implicit_gb", "256"),
    ],
)
def test_canonical_evidence_values(query: str, rule: str, value: str) -> None:
    understanding = parse_normalized_query(query)
    (term,) = [t for t in understanding.matched_terms if t.rule.value == rule]
    assert term.value == value
    assert "e" not in term.value.lower() and "," not in term.value
    assert not term.value.startswith(("+", "-"))


def test_price_fields_are_exact_two_decimal_decimals() -> None:
    understanding = parse_normalized_query("above 20.5k under ₹40000")
    assert understanding.min_price == Decimal("20500.00")
    assert understanding.max_price == Decimal("40000.00")
    assert str(understanding.min_price) == "20500.00"
    assert str(understanding.max_price) == "40000.00"
    assert '"min_price":"20500.00"' in understanding.model_dump_json()
    assert '"max_price":"40000.00"' in understanding.model_dump_json()


def test_price_validator_quantizes_to_cents() -> None:
    understanding = QueryUnderstanding(raw_query="x", max_price=Decimal("40000"))
    assert str(understanding.max_price) == "40000.00"


@pytest.mark.parametrize(
    ("price", "stored"),
    [
        (Decimal("4E+4"), "40000.00"),
        (Decimal("4.0E+4"), "40000.00"),
        (Decimal("1E-2"), "0.01"),
        (Decimal("9.99999999999E+9"), "9999999999.99"),
    ],
)
def test_exponent_form_input_is_stored_in_canonical_two_decimal_form(
    price: Decimal, stored: str
) -> None:
    # The contract fixes the stored and serialized form (quantized to 0.01, never an exponent),
    # not the spelling a direct constructor caller uses for an in-range value.
    understanding = QueryUnderstanding(raw_query="x", min_price=price)
    assert understanding.min_price == Decimal(stored)
    assert understanding.min_price.as_tuple().exponent == -2
    assert str(understanding.min_price) == stored
    assert f'"min_price":"{stored}"' in understanding.model_dump_json()


@pytest.mark.parametrize("price", [Decimal("1E+10"), Decimal("1E-3"), Decimal("-4E+4")])
def test_exponent_form_input_outside_the_bounds_is_rejected(price: Decimal) -> None:
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="x", max_price=price)


@pytest.mark.parametrize(
    "price",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("-1"),
        Decimal("0"),
        Decimal("0.00"),
        Decimal("1.001"),
        Decimal("10000000000.00"),
        Decimal("9999999999.991"),
        "nan",
        "inf",
    ],
)
@pytest.mark.parametrize("name", ["min_price", "max_price"])
def test_price_validators_reject_invalid_values(name: str, price: object) -> None:
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="x", **{name: price})


@pytest.mark.parametrize(
    ("name", "value"),
    [("ram_gb", 0), ("ram_gb", 4097), ("ram_gb", -8), ("storage_gb", 0), ("storage_gb", 65537)],
)
def test_capacity_validators_reject_out_of_range_values(name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        QueryUnderstanding(raw_query="x", **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [("ram_gb", 1), ("ram_gb", 4096), ("storage_gb", 1), ("storage_gb", 65536)],
)
def test_capacity_validators_accept_bounds(name: str, value: int) -> None:
    assert getattr(QueryUnderstanding(raw_query="x", **{name: value}), name) == value


@pytest.mark.parametrize(
    "query",
    [
        "under 10000000k laptop",
        "4097gb ram 256gb ssd",
        "65tb ssd under 40k",
        "above ₹0 under 30k",
        "under ₹10000000000",
        "8gb 70000gb ssd",
    ],
)
def test_rejected_numbers_never_partially_bind(query: str) -> None:
    understanding = parse_normalized_query(query)
    rejected = [a.span for a in understanding.ambiguities if a.reason.value == "out_of_range_value"]
    for term in understanding.matched_terms:
        for span in rejected:
            assert term.span.end <= span.start or span.end <= term.span.start
    data = understanding.model_dump(mode="json")
    for term in understanding.matched_terms:
        if term.field.value in ("ram_gb", "storage_gb", "min_price", "max_price"):
            assert str(data[term.field.value]) == term.value
