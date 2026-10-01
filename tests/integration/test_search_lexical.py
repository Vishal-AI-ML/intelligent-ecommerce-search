"""Lexical retrieval on the committed seed (a throwaway database, read-only).

These are provisional consistency checks, NOT relevance labels or quality metrics: they show the
PostgreSQL FTS results agree with an independent word-matching oracle and that ordering is
correct and deterministic.
"""

import json

import pytest
from search_support import (
    assert_ranked,
    oracle_matches,
    oracle_tokens,
    plan_node_types,
    query_terms,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.search.documents import DOCUMENT_VERSION
from ecommerce_search.search.lexical import (
    RETRIEVAL_SQL,
    lexical_search,
    retrieval_params,
)
from ecommerce_search.search.query import normalize_query

pytestmark = pytest.mark.integration

API_MAX = 50
UNBOUNDED = 10_000  # test-only limit on the internal retrieval function, never the public API

SMOKE_QUERIES = [
    "laptop",
    "hp laptop",
    "8gb laptop",
    "256gb ssd laptop",
    "apple phone",
    "iphone",
    "nike shoes",
    "wireless headphones",
    "anc headphones",
    "zzqxv",
    "HP   Laptop",
    "hp laptop!!",
    "LAPTOP",
]


@pytest.fixture(scope="module")
def tokens(seeded_engine):
    return oracle_tokens(seeded_engine)


def search(engine, query: str, limit: int):
    with Session(engine) as session:
        return lexical_search(session, normalize_query(query, 200), limit)


def test_the_gin_index_exists_on_the_search_vector(seeded_engine):
    with seeded_engine.connect() as conn:
        definition = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND "
                "indexname = 'ix_product_search_documents_search_vector'"
            )
        ).scalar_one()
    assert "USING gin (search_vector)" in definition


def test_one_current_document_per_product(seeded_engine):
    with seeded_engine.connect() as conn:
        total, versions, matching = conn.execute(
            text(
                "SELECT count(*), array_agg(DISTINCT d.document_version), "
                "count(*) FILTER (WHERE d.source_content_sha256 = p.content_sha256) "
                "FROM product_search_documents d JOIN products p USING (product_id)"
            )
        ).one()
    assert (total, versions, matching) == (240, [DOCUMENT_VERSION], 240)


@pytest.mark.parametrize("query", SMOKE_QUERIES)
def test_smoke_query_agrees_with_the_independent_oracle(seeded_engine, tokens, query):
    expected = oracle_matches(tokens, query)
    internal = search(seeded_engine, query, UNBOUNDED)
    assert_ranked(internal.hits)
    # The internal (unbounded) retrieval must return exactly the oracle's full set.
    assert {h.product_id for h in internal.hits} == expected
    # Every returned row contains every query term (checked by the oracle's own tokens).
    terms = set(query_terms(query))
    assert all(terms <= tokens[h.product_id] for h in internal.hits)

    # The public contract caps results at top_k. Compare the complete set only when it fits.
    public = search(seeded_engine, query, API_MAX)
    assert_ranked(public.hits)
    assert len(public.hits) == min(len(expected), API_MAX)
    assert all(h.product_id in expected for h in public.hits)
    if len(expected) <= API_MAX:
        assert {h.product_id for h in public.hits} == expected
    else:
        # Broad query: only the first API_MAX of the ranked list are returned (never "all").
        assert [h.product_id for h in public.hits] == [
            h.product_id for h in internal.hits[:API_MAX]
        ]


def test_broad_query_exceeds_the_public_limit_and_is_not_claimed_complete(seeded_engine, tokens):
    expected = oracle_matches(tokens, "laptop")
    assert len(expected) == 80 > API_MAX
    assert len(search(seeded_engine, "laptop", API_MAX).hits) == API_MAX


def test_complete_sets_for_the_specific_queries_fit_the_limit(seeded_engine, tokens):
    sizes = {q: len(oracle_matches(tokens, q)) for q in SMOKE_QUERIES}
    for query in (
        "hp laptop",
        "256gb ssd laptop",
        "apple phone",
        "nike shoes",
        "wireless headphones",
        "anc headphones",
    ):
        assert 0 < sizes[query] <= API_MAX, (query, sizes[query])
    assert sizes["nike shoes"] == 10 and sizes["anc headphones"] == 20


def test_iphone_returns_nothing_because_the_term_is_not_in_the_catalog(seeded_engine, tokens):
    # A documented lexical limitation of the synthetic catalog, not a success.
    assert not any("iphone" in bag for bag in tokens.values())
    result = search(seeded_engine, "iphone", API_MAX)
    assert result.hits == [] and result.tsquery == "'iphone'"


def test_apple_phone_needs_the_category_term_not_just_the_title(seeded_engine):
    hits = search(seeded_engine, "apple phone", UNBOUNDED).hits
    assert hits and {h.category for h in hits} == {"phone"} and {h.brand for h in hits} == {"Apple"}


def test_anc_matches_exactly_the_products_whose_anc_flag_is_true(seeded_engine):
    hits = search(seeded_engine, "anc headphones", UNBOUNDED).hits
    with seeded_engine.connect() as conn:
        flagged = set(
            conn.execute(text("SELECT product_id FROM headphone_specs WHERE anc IS TRUE")).scalars()
        )
        wireless_false = set(
            conn.execute(
                text("SELECT product_id FROM headphone_specs WHERE anc IS NOT TRUE")
            ).scalars()
        )
    ids = {h.product_id for h in hits}
    assert ids == flagged and not ids & wireless_false  # anc=false never emits the token


def test_seller_style_variants_are_retrievable_through_canonical_tokens(seeded_engine):
    with seeded_engine.connect() as conn:
        spaced = conn.execute(
            text(
                "SELECT p.product_id, s.ram_gb FROM raw_catalog_records r "
                "JOIN products p ON p.raw_record_id = r.id "
                "JOIN laptop_specs s ON s.product_id = p.product_id "
                'WHERE r.raw_line LIKE \'%"ram_gb":"% GB"%\''
            )
        ).all()
        terabyte = conn.execute(
            text(
                "SELECT p.product_id FROM raw_catalog_records r "
                "JOIN products p ON p.raw_record_id = r.id WHERE r.raw_line LIKE '%TB\"%'"
            )
        ).scalars()
        terabyte = list(terabyte)
        lowercase = conn.execute(
            text(
                "SELECT p.product_id, p.brand FROM raw_catalog_records r "
                "JOIN products p ON p.raw_record_id = r.id "
                'WHERE r.raw_line LIKE \'%"brand":" %\''
            )
        ).all()
    assert spaced and terabyte and lowercase  # the seed does contain these variants
    for pid, ram in spaced:
        assert pid in {h.product_id for h in search(seeded_engine, f"{ram}gb", UNBOUNDED).hits}
    for pid in terabyte:
        assert pid in {h.product_id for h in search(seeded_engine, "1tb", UNBOUNDED).hits}
    for pid, brand in lowercase:
        assert pid in {h.product_id for h in search(seeded_engine, brand.lower(), UNBOUNDED).hits}


@pytest.mark.parametrize("variant", ["hp laptop", "HP LAPTOP", "Hp   Laptop", "  hp\tlaptop\n"])
def test_case_and_whitespace_variants_return_identical_results(seeded_engine, variant):
    baseline = search(seeded_engine, "hp laptop", UNBOUNDED)
    other = search(seeded_engine, variant, UNBOUNDED)
    assert [(h.product_id, h.lexical_score) for h in other.hits] == [
        (h.product_id, h.lexical_score) for h in baseline.hits
    ]
    assert other.tsquery == baseline.tsquery == "'hp' & 'laptop'"


@pytest.mark.parametrize(
    ("query", "tsquery"),
    [
        ("8gb laptop", "'8gb' & 'laptop'"),
        ("256gb ssd laptop", "'256gb' & 'ssd' & 'laptop'"),
        ("nike -shoes", "'nike' & 'shoes'"),  # the hyphen is plain input, not an exclusion
        ('"nike" shoes', "'nike' & 'shoes'"),
        ("in-ear", "'in-ear' & 'in' & 'ear'"),
        ("!!!", ""),
    ],
)
def test_generated_tsquery_for_technical_and_punctuated_input(seeded_engine, query, tsquery):
    assert search(seeded_engine, query, 5).tsquery == tsquery


def test_hyphen_is_not_an_exclusion_operator(seeded_engine):
    assert {h.product_id for h in search(seeded_engine, "nike -shoes", UNBOUNDED).hits} == {
        h.product_id for h in search(seeded_engine, "nike shoes", UNBOUNDED).hits
    }


def test_split_capacity_text_is_a_different_query_than_the_canonical_token(seeded_engine):
    # "8 gb" tokenizes to '8' & 'gb', so it is not equivalent to "8gb" (V0 limitation, no parsing).
    assert search(seeded_engine, "8 gb laptop", 5).tsquery == "'8' & 'gb' & 'laptop'"


def test_punctuation_only_and_nonsense_return_empty_lists_cleanly(seeded_engine):
    for query in ("!!!", '"', "-", "???", "zzqxv", "qwertyuiop asdfgh"):
        result = search(seeded_engine, query, 10)
        assert result.hits == []


def test_filler_and_hinglish_words_are_ordinary_required_terms_under_strict_and(
    seeded_engine, tokens
):
    # Recorded V0 limitation (addressed by query understanding in M6): every term must match, so
    # a word the catalog does not contain empties the result, whatever its intent.
    for query in ("best laptop", "coding ke liye laptop", "sasta laptop"):
        assert search(seeded_engine, query, 10).hits == [], query
    # "for" and "coding" do occur together in some laptop descriptions, so this one matches
    # only those products (as plain words, not as an interpreted "coding intent").
    expected = oracle_matches(tokens, "laptop for coding")
    got = {h.product_id for h in search(seeded_engine, "laptop for coding", UNBOUNDED).hits}
    assert got == expected and 0 < len(got) < 80


def test_repeated_runs_return_the_same_ranked_list(seeded_engine):
    for query in ("laptop", "wireless headphones", "hp laptop"):
        first = search(seeded_engine, query, API_MAX)
        for _ in range(3):
            again = search(seeded_engine, query, API_MAX)
            assert [(h.product_id, h.lexical_score) for h in again.hits] == [
                (h.product_id, h.lexical_score) for h in first.hits
            ]


def test_ties_are_ordered_by_product_id(seeded_engine):
    hits = search(seeded_engine, "laptop", UNBOUNDED).hits
    scores = [h.lexical_score for h in hits]
    assert len(set(scores)) < len(scores), "the seed should produce exact score ties"
    assert_ranked(hits)


def test_top_k_is_respected_and_limit_one_is_the_best_hit(seeded_engine):
    full = search(seeded_engine, "laptop", UNBOUNDED).hits
    for limit in (1, 3, 10, 50):
        part = search(seeded_engine, "laptop", limit).hits
        assert [h.product_id for h in part] == [h.product_id for h in full[:limit]]


def test_subcategory_forms_are_searchable_through_the_taxonomy_section(seeded_engine):
    hits = search(seeded_engine, "ultrabook", UNBOUNDED).hits
    assert hits and all(h.category == "laptop" and h.subcategory == "ultrabook" for h in hits)


def test_results_carry_catalog_fields_and_decimal_money(seeded_engine):
    hit = search(seeded_engine, "hp laptop", 1).hits[0]
    assert hit.brand == "HP" and hit.category == "laptop"
    assert str(hit.price).count(".") == 1 and hit.currency == "INR"


def test_the_retrieval_query_is_a_full_text_query_and_explain_is_capturable(seeded_engine):
    sql = str(RETRIEVAL_SQL)
    assert "@@" in sql and "plainto_tsquery" in sql and "search_vector" in sql
    assert "ORDER BY lexical_score DESC, p.product_id ASC" in sql
    with seeded_engine.connect() as conn:
        raw = conn.execute(
            text("EXPLAIN (FORMAT JSON) " + sql), retrieval_params("hp laptop", 10)
        ).scalar_one()
    plan = raw if isinstance(raw, list) else json.loads(raw)
    nodes = plan_node_types(plan)
    assert nodes  # the planner's real choice is recorded by the benchmark; no index claim here


def test_gin_index_capability_check_with_seqscan_disabled_test_only(seeded_engine):
    """CAPABILITY CHECK ONLY. `enable_seqscan` and `enable_indexscan` are disabled for this one transaction to prove the
    planner CAN use the GIN index. It says nothing about the plan PostgreSQL chooses normally at
    240 rows (often a sequential scan), and production code never changes planner settings."""
    with seeded_engine.connect() as conn, conn.begin():
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        # With only seqscan off, PostgreSQL still chose a full pk Index Scan on the search table
        # (observed), not the GIN index. Plain index scans are disabled as well so the only
        # remaining access path to the vector is the GIN bitmap scan.
        conn.execute(text("SET LOCAL enable_indexscan = off"))
        raw = conn.execute(
            text("EXPLAIN (FORMAT JSON) " + str(RETRIEVAL_SQL)), retrieval_params("hp laptop", 10)
        ).scalar_one()
    plan = raw if isinstance(raw, list) else json.loads(raw)
    nodes = plan_node_types(plan)
    assert any("ix_product_search_documents_search_vector" in n for n in nodes), nodes


def test_the_database_under_test_is_a_scratch_database_not_the_development_one(
    seeded_engine, settings
):
    name = seeded_engine.url.database
    assert name.startswith("ecommerce_search_test_") and name != settings.postgres_db


# ---- I1/I6 D: Unicode at the query layer (normalize_query + the real retrieval query) -------


@pytest.mark.parametrize(
    "unsafe",
    [
        "\ud800",
        "\udc00",
        "lap\ud800top",
        "lap\x00top",
        "lap\x07top",
        "lap​top",
        "‮laptop",
    ],
)
def test_unsafe_text_is_rejected_before_it_can_reach_postgresql(seeded_engine, unsafe):
    from ecommerce_search.search.query import QueryError

    with pytest.raises(QueryError):
        search(seeded_engine, unsafe, 10)


@pytest.mark.parametrize(
    "text_value",
    ["नमस्ते", "می‌خ", "café", "\U0001f4bb"],
)
def test_valid_non_ascii_text_runs_in_postgresql_and_matches_nothing(seeded_engine, text_value):
    result = search(seeded_engine, text_value, 10)
    assert result.hits == []  # accepted and executed: no encoding error, no match
