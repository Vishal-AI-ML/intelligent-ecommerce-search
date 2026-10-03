"""Milestone 7 filtered retrieval SQL (pure): the closed fragment table, statement construction
for every field subset, bind parameters, value isolation, and the no-post-filter/no-retry path.

No database: statements are inspected as text, and the search functions run against a recording
fake session that returns rows which deliberately violate the filters."""

import dataclasses
import inspect
import itertools
import re
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ecommerce_search.catalog.taxonomy import Category, StorageInterface, StorageType
from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2
from ecommerce_search.filtering import FilterField, FilterSpec
from ecommerce_search.search import dense, filtered, hybrid, lexical
from ecommerce_search.search.filtered import (
    FILTER_FRAGMENTS,
    active_fields,
    dense_statement,
    filter_params,
    filtered_dense_search,
    filtered_lexical_search,
    lexical_statement,
    read_filtered_sources,
)

pytestmark = pytest.mark.usefixtures("no_socket_connect")

ALL_SUBSETS = [
    tuple(field for field, keep in zip(FilterField, mask, strict=True) if keep)
    for mask in itertools.product((False, True), repeat=len(FilterField))
]
BIND = re.compile(r"(?<![:\w]):(\w+)")
# Written by hand from fp-1 (expected parameter names, not read from the module).
EXPECTED_PARAMS = {
    FilterField.CATEGORY: "f_category",
    FilterField.BRAND: "f_brand",
    FilterField.RAM_GB: "f_ram_gb",
    FilterField.STORAGE_GB: "f_storage_gb",
    FilterField.STORAGE_TYPE: "f_storage_type",
    FilterField.STORAGE_INTERFACE: "f_storage_interface",
    FilterField.MIN_PRICE: "f_min_price",
    FilterField.MAX_PRICE: "f_max_price",
}
FULL_SPEC = FilterSpec(
    category=Category.LAPTOP,
    brand="HP",
    ram_gb=8,
    storage_gb=256,
    storage_type=StorageType.SSD,
    storage_interface=StorageInterface.NVME,
    min_price=Decimal("30000.00"),
    max_price=Decimal("40000.50"),
)
VECTOR = [1.0] + [0.0] * (ALL_MINILM_L6_V2.dimension - 1)


def squash(sql: str) -> str:
    return " ".join(sql.split())


def split_filters(sql: str, base_sql: str) -> str:
    """The filter lines of `sql`: what sits between the base WHERE predicates and ORDER BY."""
    base_where, base_tail = squash(base_sql).split(" ORDER BY ")
    body = squash(sql)
    assert body.startswith(base_where) and body.endswith(" ORDER BY " + base_tail)
    return body[len(base_where) : -len(" ORDER BY " + base_tail)].strip()


# ---------------------------------------------------------------- fragment table


def test_fragment_table_is_closed_and_in_fp1_field_order():
    assert tuple(FILTER_FRAGMENTS) == tuple(FilterField)
    assert {field: fragment.param for field, fragment in FILTER_FRAGMENTS.items()} == (
        EXPECTED_PARAMS
    )
    with pytest.raises(TypeError):
        FILTER_FRAGMENTS[FilterField.BRAND] = FILTER_FRAGMENTS[FilterField.CATEGORY]  # type: ignore[index]


def test_each_fragment_binds_only_its_own_parameter():
    for field, fragment in FILTER_FRAGMENTS.items():
        assert set(BIND.findall(fragment.sql)) == {EXPECTED_PARAMS[field]}
        assert "'" not in fragment.sql and "%" not in fragment.sql and ";" not in fragment.sql


@pytest.mark.parametrize(
    ("field", "required"),
    [
        (FilterField.CATEGORY, ["p.category = :f_category"]),
        (FilterField.BRAND, ["p.brand = :f_brand"]),
        (FilterField.RAM_GB, ["laptop_specs", "phone_specs", "s.ram_gb = :f_ram_gb", " OR "]),
        (
            FilterField.STORAGE_GB,
            ["laptop_specs", "phone_specs", "s.storage_gb = :f_storage_gb", " OR "],
        ),
        (FilterField.STORAGE_TYPE, ["laptop_specs", "s.storage_type = :f_storage_type"]),
        (
            FilterField.STORAGE_INTERFACE,
            ["laptop_specs", "s.storage_interface = :f_storage_interface"],
        ),
        (FilterField.MIN_PRICE, ["p.price >= :f_min_price"]),
        (FilterField.MAX_PRICE, ["p.price <= :f_max_price"]),
    ],
)
def test_fragment_semantics(field, required):
    sql = FILTER_FRAGMENTS[field].sql
    for piece in required:
        assert piece in sql
    if field in (FilterField.STORAGE_TYPE, FilterField.STORAGE_INTERFACE):
        assert "phone_specs" not in sql
    spec_fields = {
        FilterField.RAM_GB,
        FilterField.STORAGE_GB,
        FilterField.STORAGE_TYPE,
        FilterField.STORAGE_INTERFACE,
    }
    if field in spec_fields:  # correlated EXISTS: one row per product, never a join
        assert sql.count("EXISTS (SELECT 1 FROM") == sql.count("s.product_id = p.product_id") >= 1
        assert "JOIN" not in sql.upper()


# ---------------------------------------------------------------- statements


def test_empty_filters_reproduce_the_existing_statements_exactly():
    # Byte-for-byte, not just modulo whitespace: any edit to the committed statements fails here.
    assert lexical_statement(()).text == lexical.RETRIEVAL_SQL.text
    assert dense_statement(()).text == dense.RETRIEVAL_SQL.text
    assert active_fields(FilterSpec()) == ()
    assert filter_params(FilterSpec()) == {}


@pytest.mark.parametrize("fields", ALL_SUBSETS, ids=lambda f: "+".join(f) or "none")
def test_every_subset_builds_identical_ordered_filters_for_both_sources(fields):
    expected = " ".join(f"AND {squash(FILTER_FRAGMENTS[f].sql)}" for f in fields)
    lexical_sql = lexical_statement(fields).text
    dense_sql = dense_statement(fields).text
    # The same fragment text, in fp-1 order, appended to each unchanged base statement.
    assert split_filters(lexical_sql, lexical.RETRIEVAL_SQL.text) == expected
    assert split_filters(dense_sql, dense.RETRIEVAL_SQL.text) == expected
    for sql in (lexical_sql, dense_sql):
        body = squash(sql)
        assert body.index(" WHERE ") < body.index(" ORDER BY ") < body.index(" LIMIT :limit")
    filter_binds = BIND.findall(expected)
    assert set(filter_binds) == {EXPECTED_PARAMS[f] for f in fields}
    base_binds = set(BIND.findall(lexical.RETRIEVAL_SQL.text)) | set(
        BIND.findall(dense.RETRIEVAL_SQL.text)
    )
    assert not set(filter_binds) & base_binds  # filter names never collide with retrieval names


def test_statement_text_depends_only_on_the_set_of_fields():
    other = FilterSpec(
        category=Category.PHONE,
        brand="Samsung",
        ram_gb=12,
        storage_gb=1024,
        storage_type=StorageType.SSD,
        storage_interface=StorageInterface.NVME,
        min_price=Decimal("1.00"),
        max_price=Decimal("99999.99"),
    )
    assert active_fields(FULL_SPEC) == active_fields(other) == tuple(FilterField)
    assert lexical_statement(active_fields(FULL_SPEC)) is lexical_statement(active_fields(other))
    assert dense_statement(active_fields(FULL_SPEC)) is dense_statement(active_fields(other))


def test_parameters_are_complete_ordered_typed_and_values_stay_out_of_the_text():
    params = filter_params(FULL_SPEC)
    assert list(params) == [EXPECTED_PARAMS[f] for f in FilterField]
    assert params == {
        "f_category": "laptop",
        "f_brand": "HP",
        "f_ram_gb": 8,
        "f_storage_gb": 256,
        "f_storage_type": "SSD",
        "f_storage_interface": "NVME",
        "f_min_price": Decimal("30000.00"),
        "f_max_price": Decimal("40000.50"),
    }
    for name in ("f_category", "f_storage_type", "f_storage_interface"):
        assert type(params[name]) is str  # plain value, not an enum member
    for name in ("f_min_price", "f_max_price"):
        assert type(params[name]) is Decimal
    for name in ("f_ram_gb", "f_storage_gb"):
        assert type(params[name]) is int
    for sql, base in (
        (lexical_statement(tuple(FilterField)).text, lexical.RETRIEVAL_SQL.text),
        (dense_statement(tuple(FilterField)).text, dense.RETRIEVAL_SQL.text),
    ):
        filters = split_filters(sql, base)
        # No literal of any kind in the filter text: no quotes, no digits, no value words.
        assert not re.search(r"[0-9'\"%]", filters.replace("SELECT 1 FROM", ""))
        for value in ("HP", "SSD", "NVME", "'laptop'", " laptop "):
            assert value not in filters


def test_injection_like_values_are_bound_never_spliced():
    # NOT a supported input path: `model_construct` bypasses FilterSpec validation, which would
    # reject both values. The test only proves that, even then, values travel as bind
    # parameters and the statement text is exactly the one a valid spec with the same fields
    # gets; it makes no claim about how such values behave in the database.
    hostile = "HP'; DROP TABLE products; --"
    other = "x' OR '1'='1"
    spec = FilterSpec.model_construct(brand=hostile, category=other, min_price=Decimal("1"))
    valid = FilterSpec(brand="HP", category=Category.LAPTOP, min_price=Decimal("1.00"))
    fields = active_fields(spec)
    assert fields == active_fields(valid)
    assert fields == (FilterField.CATEGORY, FilterField.BRAND, FilterField.MIN_PRICE)
    for sql in (lexical_statement(fields).text, dense_statement(fields).text):
        assert "DROP" not in sql and "'1'" not in sql and "--" not in sql
    assert filter_params(spec) == {
        "f_category": other,
        "f_brand": hostile,
        "f_min_price": Decimal("1"),
    }
    session = RecordingSession([])
    filtered_lexical_search(session, "hp laptop", spec, 5)
    filtered_dense_search(session, VECTOR, ALL_MINILM_L6_V2, spec, 5)
    (lexical_sql, lexical_params), (dense_sql, dense_params) = session.calls[1], session.calls[2]
    assert lexical_sql == lexical_statement(active_fields(valid)).text
    assert dense_sql == dense_statement(active_fields(valid)).text
    for params in (lexical_params, dense_params):
        assert (params["f_brand"], params["f_category"]) == (hostile, other)


def test_non_filterspec_input_is_rejected():
    with pytest.raises(TypeError):
        active_fields({"brand": "HP"})  # type: ignore[arg-type]


# ---------------------------------------------------------------- execution path (fake session)


class RecordingSession:
    """Returns rows that violate every filter: the code must hand them back unchanged."""

    def __init__(self, rows, node_count=1):
        self.rows = rows
        self.node_count = node_count
        self.calls: list[tuple[str, dict]] = []

    def execute(self, statement, params):
        self.calls.append((statement.text, dict(params)))
        if statement is lexical.TSQUERY_SQL:
            return SimpleNamespace(one=lambda: ("'x'", self.node_count))
        return SimpleNamespace(all=lambda: list(self.rows))


def violating_row(pid, score):
    return SimpleNamespace(
        product_id=pid,
        title="Nike Shoe",
        brand="Nike",
        category="shoes",
        subcategory=None,
        description=None,
        price=Decimal("1.00"),
        currency="INR",
        rating=None,
        review_count=None,
        availability="in_stock",
        lexical_score=score,
        distance=score,
    )


ROWS = [violating_row("B", 0.5), violating_row("A", 0.25), violating_row("C", 0.75)]


def test_lexical_returns_exactly_the_sql_rows_with_positional_ranks():
    session = RecordingSession(ROWS)
    result = filtered_lexical_search(session, "hp laptop", FULL_SPEC, 3)
    assert [(h.rank, h.product_id, h.lexical_score) for h in result.hits] == [
        (1, "B", 0.5),
        (2, "A", 0.25),
        (3, "C", 0.75),
    ]
    assert len(session.calls) == 2  # tsquery check + one retrieval: no retry, no fallback
    sql, params = session.calls[1]
    assert sql == lexical_statement(tuple(FilterField)).text
    assert params == lexical.retrieval_params("hp laptop", 3) | filter_params(FULL_SPEC)
    assert params["q"] == "hp laptop"  # the query text reaches SQL unchanged


def test_lexical_empty_tsquery_skips_retrieval():
    session = RecordingSession(ROWS, node_count=0)
    assert filtered_lexical_search(session, "!!!", FULL_SPEC, 3).hits == []
    assert len(session.calls) == 1


def test_dense_returns_exactly_the_sql_rows_with_positional_ranks():
    session = RecordingSession(ROWS)
    result = filtered_dense_search(session, VECTOR, ALL_MINILM_L6_V2, FULL_SPEC, 3)
    assert [(h.rank, h.product_id, h.dense_score) for h in result.hits] == [
        (1, "B", 0.5),
        (2, "A", 0.75),
        (3, "C", 0.25),
    ]
    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert sql == dense_statement(tuple(FilterField)).text
    assert params == dense.retrieval_params(VECTOR, ALL_MINILM_L6_V2, 3) | filter_params(FULL_SPEC)


def test_empty_filters_bind_exactly_the_existing_parameters():
    session = RecordingSession([])
    filtered_lexical_search(session, "hp laptop", FilterSpec(), 7)
    filtered_dense_search(session, VECTOR, ALL_MINILM_L6_V2, FilterSpec(), 7)
    assert session.calls[1] == (
        lexical_statement(()).text,
        lexical.retrieval_params("hp laptop", 7),
    )
    assert session.calls[2] == (
        dense_statement(()).text,
        dense.retrieval_params(VECTOR, ALL_MINILM_L6_V2, 7),
    )


def _hit_fields(hit) -> dict:
    return dataclasses.asdict(hit)


def test_empty_filters_match_the_committed_reads_call_for_call_and_hit_for_hit():
    """Pins the copied base statements, parameters and row-to-hit mapping against the committed
    `lexical_search` / `dense_search`: any drift in either module fails here, offline."""
    for node_count in (1, 0):
        old, new = RecordingSession(ROWS, node_count), RecordingSession(ROWS, node_count)
        expected = lexical.lexical_search(old, "hp laptop", 3)
        actual = filtered_lexical_search(new, "hp laptop", FilterSpec(), 3)
        assert new.calls == old.calls
        assert actual.tsquery == expected.tsquery
        assert [_hit_fields(h) for h in actual.hits] == [_hit_fields(h) for h in expected.hits]
    old, new = RecordingSession(ROWS), RecordingSession(ROWS)
    expected = dense.dense_search(old, VECTOR, ALL_MINILM_L6_V2, 3)
    actual = filtered_dense_search(new, VECTOR, ALL_MINILM_L6_V2, FilterSpec(), 3)
    assert new.calls == old.calls
    assert [_hit_fields(h) for h in actual.hits] == [_hit_fields(h) for h in expected.hits]
    assert actual.hits  # non-vacuous: three mapped rows were compared


def test_read_filtered_sources_refuses_an_open_transaction_and_bad_filters():
    class Open:
        def in_transaction(self):
            return True

    class Closed:
        def in_transaction(self):
            return False

        def begin(self):  # pragma: no cover - must not be reached
            raise AssertionError("no transaction for invalid input")

    with pytest.raises(RuntimeError):
        read_filtered_sources(Open(), "q", VECTOR, ALL_MINILM_L6_V2, FilterSpec(), 5, 5)
    with pytest.raises(TypeError):
        read_filtered_sources(Closed(), "q", VECTOR, ALL_MINILM_L6_V2, {}, 5, 5)  # type: ignore[arg-type]


def test_module_reuses_existing_contracts_and_has_no_post_filter_path():
    assert filtered.SNAPSHOT_SQL is hybrid.SNAPSHOT_SQL
    assert filtered.LexicalHit is lexical.LexicalHit and filtered.DenseHit is dense.DenseHit
    functions = {
        name: value
        for name, value in vars(filtered).items()
        if inspect.isfunction(inspect.unwrap(value))
        and inspect.unwrap(value).__module__ == filtered.__name__
    }
    assert set(functions) == {
        "active_fields",
        "filter_params",
        "filter_clause",
        "lexical_statement",
        "dense_statement",
        "filtered_lexical_search",
        "filtered_dense_search",
        "read_filtered_sources",
    }
    # Nothing calls the unfiltered reads (no fallback) and no helper touches hits after SQL.
    referenced = set()
    for value in functions.values():
        referenced |= set(inspect.unwrap(value).__code__.co_names)
    assert not referenced & {"lexical_search", "dense_search", "read_sources", "fuse_rrf"}
