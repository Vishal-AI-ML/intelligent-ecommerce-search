"""Milestone 7 filter policy `fp-1`: a predeclared truth table over real `qu-1` parses, the closed
FilterSpec, contract violations, determinism, no logging and import isolation.

Expectations are written from the approved policy rules (plan §5; decisions B2–B4, I4, I5), not
copied from the implementation's output."""

import logging
import subprocess
import sys
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ecommerce_search.catalog.taxonomy import BRAND_CATEGORIES, StorageInterface, StorageType
from ecommerce_search.decision import (
    DecisionResult,
    DeterministicDecisionProvider,
    understand_normalized_query,
)
from ecommerce_search.filtering import (
    FILTER_POLICY_VERSION,
    AppliedFilter,
    FilterDerivation,
    FilterField,
    FilterOperator,
    FilterPolicyError,
    FilterSpec,
    IgnoredConstraint,
    IgnoreReason,
    derive_filters,
    policy,
)
from ecommerce_search.query_understanding import (
    Ambiguity,
    AmbiguityReason,
    Conflict,
    ConflictReason,
    MatchedTerm,
    MatchRule,
    QueryAttributes,
    QueryUnderstanding,
    SourceSpan,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")

EQ, GTE, LTE = FilterOperator.EQ, FilterOperator.GTE, FilterOperator.LTE
CONFLICT = IgnoreReason.CONFLICT
AMBIGUOUS = IgnoreReason.AMBIGUOUS_FAMILY
INFO = IgnoreReason.INFORMATIONAL_ONLY

# query -> (applied [(field, operator, value)], ignored [(field, reason)])
TRUTH_TABLE = {
    "hp laptop 8gb ram 256gb ssd under 40k": (
        [
            ("category", EQ, "laptop"),
            ("brand", EQ, "HP"),
            ("ram_gb", EQ, "8"),
            ("storage_gb", EQ, "256"),
            ("storage_type", EQ, "SSD"),
            ("max_price", LTE, "40000.00"),
        ],
        [],
    ),
    # MASTER_PLAN §1 example: R6-paired RAM and implicit-GB storage are applied (I4).
    "laptop 8gb 256 ssd": (
        [
            ("category", EQ, "laptop"),
            ("ram_gb", EQ, "8"),
            ("storage_gb", EQ, "256"),
            ("storage_type", EQ, "SSD"),
        ],
        [],
    ),
    "coding ke liye laptop": ([("category", EQ, "laptop")], [("semantic_intent", INFO)]),
    "sasta phone": ([("category", EQ, "phone")], [("price_preference", INFO)]),
    # `ssd` filters only the medium, so NVMe SSDs are not excluded (FR-QU-4).
    "ssd": ([("storage_type", EQ, "SSD")], []),
    "nvme": ([("storage_type", EQ, "SSD"), ("storage_interface", EQ, "NVME")], []),
    "nvme hdd": ([], [("storage_type", CONFLICT), ("storage_interface", CONFLICT)]),
    "ssd hdd": ([], [("storage_type", CONFLICT)]),
    "hp dell laptop": ([("category", EQ, "laptop")], [("brand", CONFLICT)]),
    "above 30k under 20k": ([], [("min_price", CONFLICT), ("max_price", CONFLICT)]),
    "from 20k ke andar": ([("min_price", GTE, "20000.00"), ("max_price", LTE, "20000.00")], []),
    "gaming laptop for coding": ([("category", EQ, "laptop")], [("semantic_intent", CONFLICT)]),
    # A price ambiguity suppresses the price family only.
    "₹40,000 laptop under 50k": ([("category", EQ, "laptop")], [("max_price", AMBIGUOUS)]),
    "laptop 8gb ram 50k": ([("category", EQ, "laptop"), ("ram_gb", EQ, "8")], []),
    # A capacity ambiguity suppresses the capacity family only.
    "laptop under 50k 16gb": ([("category", EQ, "laptop"), ("max_price", LTE, "50000.00")], []),
    # Unsupported numbers suppress both numeric families.
    "16gb ram laptop ₹40,000": ([("category", EQ, "laptop")], [("ram_gb", AMBIGUOUS)]),
    "10000000k laptop": ([("category", EQ, "laptop")], []),
    "ram 8gb ssd": ([("storage_type", EQ, "SSD")], []),
    "8 256 ssd": ([("storage_type", EQ, "SSD")], []),
    "laptop 16gb": ([("category", EQ, "laptop")], []),
    "memory 8gb": ([], []),
    "40k phone": ([("category", EQ, "phone")], []),
    "under 40000": ([], []),
    "from 20k to 40k": ([], []),
    # Brand and category are applied literally, even when the pair is unusual (I5).
    "nike laptop": ([("category", EQ, "laptop"), ("brand", EQ, "Nike")], []),
    "phone ssd": ([("category", EQ, "phone"), ("storage_type", EQ, "SSD")], []),
    "laptop laptops": ([("category", EQ, "laptop")], []),
    "1tb ssd": ([("storage_gb", EQ, "1024"), ("storage_type", EQ, "SSD")], []),
    "above 30k": ([("min_price", GTE, "30000.00")], []),
    "8ssd": ([], []),
    "!!!": ([], []),
    # Regression (Session B): an intent conflict suppresses neither category nor RAM.
    "gaming coding laptop 8gb ram": (
        [("category", EQ, "laptop"), ("ram_gb", EQ, "8")],
        [("semantic_intent", CONFLICT)],
    ),
    # A price ambiguity keeps category, brand, storage type and capacities.
    "hp ssd laptop 8gb ram 40k": (
        [
            ("category", EQ, "laptop"),
            ("brand", EQ, "HP"),
            ("ram_gb", EQ, "8"),
            ("storage_type", EQ, "SSD"),
        ],
        [],
    ),
    "hp laptop ssd above 30k 40k": (
        [("category", EQ, "laptop"), ("brand", EQ, "HP"), ("storage_type", EQ, "SSD")],
        [("min_price", AMBIGUOUS)],
    ),
    # A capacity ambiguity keeps category, brand, storage type and prices.
    "hp laptop ssd 16gb under 50k": (
        [
            ("category", EQ, "laptop"),
            ("brand", EQ, "HP"),
            ("storage_type", EQ, "SSD"),
            ("max_price", LTE, "50000.00"),
        ],
        [],
    ),
    "hp laptop memory 8gb 512gb ssd under 50k": (
        [
            ("category", EQ, "laptop"),
            ("brand", EQ, "HP"),
            ("storage_type", EQ, "SSD"),
            ("max_price", LTE, "50000.00"),
        ],
        [("storage_gb", AMBIGUOUS)],
    ),
}


def decide(query: str) -> DecisionResult:
    return understand_normalized_query(query, DeterministicDecisionProvider())


@pytest.mark.parametrize("query", list(TRUTH_TABLE))
def test_truth_table(query: str) -> None:
    applied, ignored = TRUTH_TABLE[query]
    derivation = derive_filters(decide(query))
    assert [(f.field.value, f.operator, f.value) for f in derivation.applied_filters] == applied
    assert [(i.field.value, i.reason) for i in derivation.ignored_constraints] == ignored
    assert derivation.policy_version == FILTER_POLICY_VERSION == "fp-1"
    assert derivation.spec.is_empty() == (applied == [])


@pytest.mark.parametrize("query", list(TRUTH_TABLE))
def test_applied_filters_follow_the_fixed_field_order_and_operators(query: str) -> None:
    derivation = derive_filters(decide(query))
    order = list(FilterField)
    positions = [order.index(f.field) for f in derivation.applied_filters]
    assert positions == sorted(set(positions))
    for applied in derivation.applied_filters:
        expected = {FilterField.MIN_PRICE: GTE, FilterField.MAX_PRICE: LTE}.get(applied.field, EQ)
        assert applied.operator is expected


@pytest.mark.parametrize("query", list(TRUTH_TABLE))
def test_same_input_gives_byte_identical_json(query: str) -> None:
    first = derive_filters(decide(query)).model_dump_json()
    assert derive_filters(decide(query)).model_dump_json() == first
    assert FilterDerivation.model_validate_json(first).model_dump_json() == first


def test_full_spec_values_and_serialization() -> None:
    spec = derive_filters(decide("hp laptop 8gb ram 256gb ssd under 40k")).spec
    assert spec.category == "laptop"
    assert spec.brand == "HP"
    assert (spec.ram_gb, spec.storage_gb) == (8, 256)
    assert spec.storage_type is StorageType.SSD
    assert spec.storage_interface is None
    assert spec.min_price is None
    assert spec.max_price == Decimal("40000.00")
    assert spec.model_dump(mode="json")["max_price"] == "40000.00"


def test_informational_fields_are_never_filters() -> None:
    names = {field.value for field in FilterField}
    assert {"semantic_intent", "price_preference", "retrieval_strategy"}.isdisjoint(names)
    assert set(FilterSpec.model_fields) == names | {"policy_version"}


# The approved family decision for every current parser ambiguity reason, written out
# explicitly. A reason may suppress no family; a new reason fails the coverage check below until
# it is classified here on purpose.
CAP, PRC = "capacity", "price"
AMBIGUITY_FAMILIES = {
    "bare_capacity": {CAP},
    "ambiguous_memory_capacity": {CAP},
    "capacity_multiple_roles": {CAP},
    "price_without_bound": {PRC},
    "unsupported_grouped_number": {CAP, PRC},
    "unsupported_range": {CAP, PRC},
    "unsupported_numeric_compound": {CAP, PRC},
    "out_of_range_value": {CAP, PRC},
}


def test_ambiguity_family_sets_are_exactly_the_approved_closed_sets() -> None:
    assert {
        AmbiguityReason.BARE_CAPACITY,
        AmbiguityReason.AMBIGUOUS_MEMORY_CAPACITY,
        AmbiguityReason.CAPACITY_MULTIPLE_ROLES,
        AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER,
        AmbiguityReason.UNSUPPORTED_RANGE,
        AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND,
        AmbiguityReason.OUT_OF_RANGE_VALUE,
    } == policy.CAPACITY_AMBIGUITIES
    assert {
        AmbiguityReason.PRICE_WITHOUT_BOUND,
        AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER,
        AmbiguityReason.UNSUPPORTED_RANGE,
        AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND,
        AmbiguityReason.OUT_OF_RANGE_VALUE,
    } == policy.PRICE_AMBIGUITIES
    assert {FilterField.RAM_GB, FilterField.STORAGE_GB} == policy.CAPACITY_FAMILY
    assert {FilterField.MIN_PRICE, FilterField.MAX_PRICE} == policy.PRICE_FAMILY


def test_every_current_ambiguity_reason_is_classified_explicitly() -> None:
    assert {reason.value for reason in AmbiguityReason} == set(AMBIGUITY_FAMILIES)
    for value, families in AMBIGUITY_FAMILIES.items():
        reason = AmbiguityReason(value)
        assert (reason in policy.CAPACITY_AMBIGUITIES) == (CAP in families), value
        assert (reason in policy.PRICE_AMBIGUITIES) == (PRC in families), value


@pytest.mark.parametrize("value", sorted(AMBIGUITY_FAMILIES))
def test_each_ambiguity_reason_suppresses_only_its_families(value: str) -> None:
    # Every filterable field is set with evidence; one ambiguity of `value` is added.
    raw = "hp laptop 8gb ram 256gb ssd above 20k under 40k x"
    terms = (
        _term(raw, "brand", "HP", MatchRule.BRAND_TERM, 0, 2),
        _term(raw, "category", "laptop", MatchRule.CATEGORY_TERM, 3, 9),
        _term(raw, "ram_gb", "8", MatchRule.RAM_CAPACITY, 10, 17),
        _term(raw, "storage_gb", "256", MatchRule.STORAGE_CAPACITY, 18, 27),
        _term(raw, "storage_type", "SSD", MatchRule.STORAGE_TYPE_TERM, 24, 27),
        _term(raw, "min_price", "20000.00", MatchRule.PRICE_LOWER_BOUND, 28, 37),
        _term(raw, "max_price", "40000.00", MatchRule.PRICE_UPPER_BOUND, 38, 47),
    )
    ambiguity = Ambiguity(reason=value, span=SourceSpan(start=48, end=49, text="x"))
    derivation = derive_filters(
        _decision(
            raw_query=raw,
            category="laptop",
            brand="HP",
            ram_gb=8,
            storage_gb=256,
            storage_type="SSD",
            min_price=Decimal("20000"),
            max_price=Decimal("40000"),
            matched_terms=terms,
            ambiguities=(ambiguity,),
        )
    )
    families = AMBIGUITY_FAMILIES[value]
    suppressed = (["ram_gb", "storage_gb"] if CAP in families else []) + (
        ["min_price", "max_price"] if PRC in families else []
    )
    applied = [f.field.value for f in derivation.applied_filters]
    assert applied == [
        f.value for f in FilterField if f.value not in suppressed + ["storage_interface"]
    ]
    assert {"category", "brand", "storage_type"} <= set(applied)
    assert [(i.field.value, i.reason) for i in derivation.ignored_constraints] == [
        (field, AMBIGUOUS) for field in suppressed
    ]


def test_closed_enumerations() -> None:
    assert [f.value for f in FilterField] == [
        "category",
        "brand",
        "ram_gb",
        "storage_gb",
        "storage_type",
        "storage_interface",
        "min_price",
        "max_price",
    ]
    assert [o.value for o in FilterOperator] == ["eq", "gte", "lte"]
    assert [r.value for r in IgnoreReason] == ["conflict", "ambiguous_family", "informational_only"]


# ---- FilterSpec validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "brand",
    [
        "HP'; DROP TABLE products; --",
        "hp",  # canonical spelling only
        "HP ",
        "Foo",
        "",
        "Nike' OR '1'='1",
    ],
)
def test_spec_rejects_unknown_or_injection_like_brands(brand: str) -> None:
    with pytest.raises(ValidationError):
        FilterSpec(brand=brand)


def test_spec_accepts_every_dictionary_brand() -> None:
    for brand in BRAND_CATEGORIES:
        assert FilterSpec(brand=brand).brand == brand


@pytest.mark.parametrize(
    "fields",
    [
        {"category": "Laptop"},
        {"category": "laptop'; --"},
        {"storage_type": "nvme"},
        {"storage_type": "NVME"},
        {"storage_interface": "SATA'"},
        {"ram_gb": 0},
        {"ram_gb": 4097},
        {"ram_gb": "8 OR 1=1"},
        {"storage_gb": 65537},
        {"min_price": Decimal("0")},
        {"min_price": Decimal("-1")},
        {"max_price": Decimal("NaN")},
        {"max_price": Decimal("Infinity")},
        {"max_price": Decimal("10000000000.00")},
        {"max_price": Decimal("1.001")},
        {"max_price": "forty"},
        # Strict SQL boundary (Session B decision on F1): no float, bool or string coercion.
        {"max_price": "1e3"},
        {"max_price": "40000.00"},
        {"max_price": 40000},
        {"max_price": 1.5},
        {"min_price": 40000.0},
        {"ram_gb": 8.0},
        {"ram_gb": "8"},
        {"ram_gb": True},
        {"storage_gb": 256.0},
        {"storage_gb": "256"},
        {"min_price": Decimal("50000"), "max_price": Decimal("40000")},
        {"storage_interface": StorageInterface.NVME},
        {"storage_interface": StorageInterface.NVME, "storage_type": StorageType.HDD},
        {"policy_version": "fp-2"},
        {"currency": "INR"},
    ],
)
def test_spec_rejects_invalid_values(fields: dict) -> None:
    with pytest.raises(ValidationError):
        FilterSpec(**fields)


def test_spec_boundaries_and_equal_bounds() -> None:
    spec = FilterSpec(
        ram_gb=4096,
        storage_gb=65536,
        min_price=Decimal("0.01"),
        max_price=Decimal("9999999999.99"),
    )
    assert spec.ram_gb == 4096
    equal = FilterSpec(min_price=Decimal("20000"), max_price=Decimal("20000.00"))
    assert [f.value for f in equal.applied_filters()] == ["20000.00", "20000.00"]
    nvme = FilterSpec(storage_type=StorageType.SSD, storage_interface=StorageInterface.NVME)
    assert nvme.storage_interface is StorageInterface.NVME


def test_strict_spec_json_round_trip_keeps_canonical_strings() -> None:
    spec = derive_filters(decide("hp laptop 8gb ram 256gb ssd above 20k under 40k")).spec
    dumped = spec.model_dump_json()
    assert '"min_price":"20000.00"' in dumped
    assert '"ram_gb":8' in dumped
    assert FilterSpec.model_validate_json(dumped) == spec
    with pytest.raises(ValidationError):
        FilterSpec.model_validate_json('{"ram_gb": "8"}')
    with pytest.raises(ValidationError):
        FilterSpec.model_validate_json('{"ram_gb": 8.0}')


def test_models_are_frozen_and_closed() -> None:
    spec = FilterSpec(category="laptop")
    with pytest.raises(ValidationError):
        spec.category = "phone"
    with pytest.raises(ValidationError):
        AppliedFilter(field="category", operator="eq", value="laptop", extra=1)
    with pytest.raises(ValidationError):
        IgnoredConstraint(field="brand", reason="low_confidence")
    with pytest.raises(ValidationError):
        AppliedFilter(field="semantic_intent", operator="eq", value="coding")
    with pytest.raises(ValidationError):
        AppliedFilter(field="ram_gb", operator="gt", value="8")


def test_derivation_rejects_applied_filters_not_derived_from_spec() -> None:
    spec = FilterSpec(category="laptop")
    with pytest.raises(ValidationError):
        FilterDerivation(spec=spec, applied_filters=(), ignored_constraints=())
    forged = AppliedFilter(field="brand", operator="eq", value="HP")
    with pytest.raises(ValidationError):
        FilterDerivation(
            spec=spec, applied_filters=(*spec.applied_filters(), forged), ignored_constraints=()
        )


def test_empty_spec() -> None:
    spec = FilterSpec()
    assert spec.is_empty()
    assert spec.applied_filters() == ()
    assert spec.policy_version == "fp-1"


# ---- contract violations (provider-shaped input) ----------------------------------------------


def _term(raw: str, field: str, value: str, rule: MatchRule, start: int, end: int) -> MatchedTerm:
    return MatchedTerm(
        rule=rule,
        field=field,
        value=value,
        span=SourceSpan(start=start, end=end, text=raw[start:end]),
    )


def _decision(**fields) -> DecisionResult:
    return DecisionResult(understanding=QueryUnderstanding(**fields))


def test_non_null_field_without_evidence_is_a_contract_violation() -> None:
    with pytest.raises(FilterPolicyError):
        derive_filters(_decision(raw_query="hp", brand="HP"))


def test_evidence_with_a_different_value_is_a_contract_violation() -> None:
    raw = "16gb ram"
    term = _term(raw, "ram_gb", "16", MatchRule.RAM_CAPACITY, 0, 8)
    with pytest.raises(FilterPolicyError):
        derive_filters(_decision(raw_query=raw, ram_gb=8, matched_terms=(term,)))


def test_evidence_for_another_field_is_not_enough() -> None:
    raw = "8gb ram"
    term = _term(raw, "storage_gb", "8", MatchRule.STORAGE_CAPACITY, 0, 7)
    with pytest.raises(FilterPolicyError):
        derive_filters(_decision(raw_query=raw, ram_gb=8, matched_terms=(term,)))


def test_brand_outside_the_dictionary_fails_even_with_evidence() -> None:
    raw = "foo"
    term = _term(raw, "brand", "Foo", MatchRule.BRAND_TERM, 0, 3)
    with pytest.raises(FilterPolicyError):
        derive_filters(_decision(raw_query=raw, brand="Foo", matched_terms=(term,)))


def test_interface_without_ssd_medium_fails() -> None:
    raw = "nvme"
    term = _term(raw, "storage_interface", "NVME", MatchRule.STORAGE_INTERFACE_TERM, 0, 4)
    decision = _decision(
        raw_query=raw,
        attributes=QueryAttributes(storage_interface="NVME"),
        matched_terms=(term,),
    )
    with pytest.raises(FilterPolicyError):
        derive_filters(decision)


def test_unconflicted_min_above_max_fails() -> None:
    raw = "above 50k under 40k"
    terms = (
        _term(raw, "min_price", "50000.00", MatchRule.PRICE_LOWER_BOUND, 0, 9),
        _term(raw, "max_price", "40000.00", MatchRule.PRICE_UPPER_BOUND, 10, 19),
    )
    decision = _decision(
        raw_query=raw,
        min_price=Decimal("50000"),
        max_price=Decimal("40000"),
        matched_terms=terms,
    )
    with pytest.raises(FilterPolicyError) as raised:
        derive_filters(decision)
    assert raised.value.__suppress_context__


def test_non_null_conflicted_field_is_ignored_not_applied() -> None:
    raw = "hp dell"
    hp = _term(raw, "brand", "HP", MatchRule.BRAND_TERM, 0, 2)
    dell = _term(raw, "brand", "Dell", MatchRule.BRAND_TERM, 3, 7)
    conflict = Conflict(
        field="brand", reason=ConflictReason.REPEATED_DIFFERENT_VALUES, candidates=(hp, dell)
    )
    derivation = derive_filters(
        _decision(raw_query=raw, brand="HP", matched_terms=(hp,), conflicts=(conflict,))
    )
    assert derivation.applied_filters == ()
    assert derivation.ignored_constraints == (IgnoredConstraint(field="brand", reason=CONFLICT),)


def test_conflicted_medium_blocks_a_provider_supplied_interface() -> None:
    raw = "nvme ssd hdd"
    nvme = _term(raw, "storage_interface", "NVME", MatchRule.STORAGE_INTERFACE_TERM, 0, 4)
    ssd = _term(raw, "storage_type", "SSD", MatchRule.STORAGE_TYPE_TERM, 5, 8)
    hdd = _term(raw, "storage_type", "HDD", MatchRule.STORAGE_TYPE_TERM, 9, 12)
    conflict = Conflict(
        field="storage_type", reason=ConflictReason.REPEATED_DIFFERENT_VALUES, candidates=(ssd, hdd)
    )
    derivation = derive_filters(
        _decision(
            raw_query=raw,
            attributes=QueryAttributes(storage_interface="NVME"),
            matched_terms=(nvme,),
            conflicts=(conflict,),
        )
    )
    assert derivation.applied_filters == ()
    assert [(i.field.value, i.reason) for i in derivation.ignored_constraints] == [
        ("storage_type", CONFLICT),
        ("storage_interface", CONFLICT),
    ]


def test_ambiguity_suppresses_the_whole_family() -> None:
    raw = "8gb ram 256gb ssd 16gb"
    terms = (
        _term(raw, "ram_gb", "8", MatchRule.RAM_CAPACITY, 0, 7),
        _term(raw, "storage_gb", "256", MatchRule.STORAGE_CAPACITY, 8, 17),
        _term(raw, "storage_type", "SSD", MatchRule.STORAGE_TYPE_TERM, 14, 17),
    )
    bare = Ambiguity(
        reason=AmbiguityReason.BARE_CAPACITY,
        span=SourceSpan(start=18, end=22, text="16gb"),
        value="16",
    )
    derivation = derive_filters(
        _decision(
            raw_query=raw,
            ram_gb=8,
            storage_gb=256,
            storage_type="SSD",
            matched_terms=terms,
            ambiguities=(bare,),
        )
    )
    assert [f.field.value for f in derivation.applied_filters] == ["storage_type"]
    assert [(i.field.value, i.reason) for i in derivation.ignored_constraints] == [
        ("ram_gb", AMBIGUOUS),
        ("storage_gb", AMBIGUOUS),
    ]


def test_non_decision_input_fails() -> None:
    with pytest.raises(FilterPolicyError):
        derive_filters(decide("hp laptop").understanding)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("provider", "jev"),
        ("provider_version", "qu-2"),
    ],
)
def test_unsupported_provider_or_version_fails(attribute: str, value: str) -> None:
    fields = {"provider": "deterministic", "provider_version": "qu-1"} | {attribute: value}
    forged = DecisionResult.model_construct(
        understanding=decide("hp laptop").understanding, **fields
    )
    with pytest.raises(FilterPolicyError):
        derive_filters(forged)


@pytest.mark.parametrize("attribute,value", [("parser_version", "qu-2"), ("lexicon_version", "2")])
def test_unsupported_parser_or_lexicon_version_fails(attribute: str, value: str) -> None:
    understanding = decide("hp laptop").understanding.model_copy(update={attribute: value})
    forged = DecisionResult.model_construct(
        provider="deterministic", provider_version="qu-1", understanding=understanding
    )
    with pytest.raises(FilterPolicyError):
        derive_filters(forged)


def test_error_message_is_fixed_and_never_echoes_query_text() -> None:
    marker = "zzquerymarkerzz"
    raw = f"hp {marker}"
    with pytest.raises(FilterPolicyError) as raised:
        derive_filters(_decision(raw_query=raw, brand="HP"))
    assert str(raised.value) == "filter policy failed"
    assert marker not in repr(raised.value)


def test_policy_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    for query in TRUTH_TABLE:
        derive_filters(decide(query))
    with pytest.raises(FilterPolicyError):
        derive_filters(_decision(raw_query="hp", brand="HP"))
    assert caplog.records == []


_ISOLATION_PROBE = """
import sys
import ecommerce_search.filtering
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


def test_filtering_package_imports_no_db_web_settings_or_model_stack() -> None:
    done = subprocess.run(  # noqa: S603 - fixed argv, sys.executable
        [sys.executable, "-c", _ISOLATION_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""
