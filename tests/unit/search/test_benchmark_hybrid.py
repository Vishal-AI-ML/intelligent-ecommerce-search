"""scripts/benchmark_hybrid.py: frozen protocol, scoring, selection, guards, derivation, cleanup.

Pure tests only: no database, no model, no network (strict socket guard)."""

import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION
from ecommerce_search.search.hybrid import fuse_rrf

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def _load():
    name = "benchmark_hybrid_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "benchmark_hybrid.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()
IDENTITY = bench.check_protocol_identity()
CORPUS_DIGEST = bench.embedding_corpus_digest()

# The reviewed copy of the frozen protocol. Any edit to the harness table must fail here.
REVIEWED_QUERIES = (
    "laptop",
    "hp laptop",
    "8gb laptop",
    "iphone",
    "zzqxv",
    "coding ke liye laptop",
    "coding laptop",
    "student laptop",
    "lightweight laptop",
    "premium phone",
    "running shoes",
    "noise cancelling headphones",
    "256gb ssd laptop",
    "apple phone",
    "nike shoes",
    "wireless headphones",
    "anc headphones",
)
REVIEWED_EXPECTATIONS = {
    "laptop": [("category", "laptop")],
    "hp laptop": [("brand", "HP"), ("category", "laptop")],
    "8gb laptop": [("category", "laptop")],
    "iphone": [],
    "zzqxv": [],
    "coding ke liye laptop": [("category", "laptop")],
    "coding laptop": [("category", "laptop")],
    "student laptop": [("category", "laptop")],
    "lightweight laptop": [("category", "laptop")],
    "premium phone": [("category", "phone")],
    "running shoes": [("category", "shoes")],
    "noise cancelling headphones": [("anc", True), ("category", "headphones")],
    "256gb ssd laptop": [("category", "laptop")],
    "apple phone": [("brand", "Apple"), ("category", "phone")],
    "nike shoes": [("brand", "Nike"), ("category", "shoes")],
    "wireless headphones": [("category", "headphones")],
    "anc headphones": [("anc", True), ("category", "headphones")],
}
QUERY_SET_SHA256 = "ea1df08db89a8bceb0e5b6437f18e75a30aed997fb70d727d080b441ebbd78d3"
EXPECTATION_TABLE_SHA256 = "ac5647876b33397bc0b0cc2d60e60357af9285daf65f5973e5c675f824c8d643"


def protocol_copy():
    return [json.loads(json.dumps(entry)) for entry in bench.QUERY_PROTOCOL]


# ---- frozen protocol -----------------------------------------------------------------------------


def test_exact_query_order_count_and_hash():
    assert bench.QUERIES == REVIEWED_QUERIES
    assert len(bench.QUERIES) == 17 == IDENTITY["count"]
    assert tuple(bench.QUERIES[:12]) == tuple(sys.modules["embedding_model_selection"].QUERIES)
    canonical = json.dumps(list(REVIEWED_QUERIES), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == QUERY_SET_SHA256
    assert IDENTITY["query_set_sha256"] == bench.QUERY_SET_SHA256 == QUERY_SET_SHA256


def test_exact_expectation_table_and_hash():
    table = {
        e["query"]: [(x["field"], x["value"]) for x in e["expectations"]]
        for e in IDENTITY["expectation_table"]
    }
    assert table == REVIEWED_EXPECTATIONS
    assert IDENTITY["expectation_table_sha256"] == bench.EXPECTATION_TABLE_SHA256
    assert bench.EXPECTATION_TABLE_SHA256 == EXPECTATION_TABLE_SHA256
    assert IDENTITY["diagnostic_only_queries"] == ["iphone", "zzqxv"]
    assert len(IDENTITY["checked_queries"]) == 15
    assert bench.APPROVED_TERMS == {
        "laptop": ("category", "laptop"),
        "phone": ("category", "phone"),
        "shoes": ("category", "shoes"),
        "headphones": ("category", "headphones"),
        "hp": ("brand", "HP"),
        "apple": ("brand", "Apple"),
        "nike": ("brand", "Nike"),
        "anc": ("anc", True),
        "noise cancelling": ("anc", True),
    }
    assert frozenset({"noise cancelling"}) == bench.APPROVED_PHRASES


def _reviewed(query, *expectations):
    status = "checked" if expectations else "diagnostic_only"
    return {
        "query": query,
        "status": status,
        "expectations": [{"term": t, "field": f, "value": v} for t, f, v in expectations],
    }


# The complete reviewed table, written out independently of QUERY_PROTOCOL and its helpers.
REVIEWED_TABLE = [
    _reviewed("laptop", ("laptop", "category", "laptop")),
    _reviewed("hp laptop", ("hp", "brand", "HP"), ("laptop", "category", "laptop")),
    _reviewed("8gb laptop", ("laptop", "category", "laptop")),
    _reviewed("iphone"),
    _reviewed("zzqxv"),
    _reviewed("coding ke liye laptop", ("laptop", "category", "laptop")),
    _reviewed("coding laptop", ("laptop", "category", "laptop")),
    _reviewed("student laptop", ("laptop", "category", "laptop")),
    _reviewed("lightweight laptop", ("laptop", "category", "laptop")),
    _reviewed("premium phone", ("phone", "category", "phone")),
    _reviewed("running shoes", ("shoes", "category", "shoes")),
    _reviewed(
        "noise cancelling headphones",
        ("noise cancelling", "anc", True),
        ("headphones", "category", "headphones"),
    ),
    _reviewed("256gb ssd laptop", ("laptop", "category", "laptop")),
    _reviewed("apple phone", ("apple", "brand", "Apple"), ("phone", "category", "phone")),
    _reviewed("nike shoes", ("nike", "brand", "Nike"), ("shoes", "category", "shoes")),
    _reviewed("wireless headphones", ("headphones", "category", "headphones")),
    _reviewed("anc headphones", ("anc", "anc", True), ("headphones", "category", "headphones")),
]


def _canonical_sha256(obj):
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_independently_reconstructed_table_recomputes_the_reviewed_hashes():
    assert _canonical_sha256(REVIEWED_TABLE) == (
        "ac5647876b33397bc0b0cc2d60e60357af9285daf65f5973e5c675f824c8d643"
    )
    assert _canonical_sha256([e["query"] for e in REVIEWED_TABLE]) == (
        "ea1df08db89a8bceb0e5b6437f18e75a30aed997fb70d727d080b441ebbd78d3"
    )
    assert IDENTITY["expectation_table"] == REVIEWED_TABLE
    assert [e["query"] for e in REVIEWED_TABLE] == list(REVIEWED_QUERIES)


def test_identity_is_deterministic_and_says_it_is_a_proxy():
    assert bench.protocol_identity() == IDENTITY
    label = IDENTITY["proxy_label"]
    assert label.startswith("provisional deterministic attribute-consistency proxy")
    assert "Not relevance, not quality ground truth, not human labels" in label


def _with_query(index, query, expectations=None, status=None):
    protocol = protocol_copy()
    protocol[index]["query"] = query
    if expectations is not None:
        protocol[index]["expectations"] = expectations
    if status is not None:
        protocol[index]["status"] = status
    return protocol


@pytest.mark.parametrize(
    ("protocol", "message"),
    [
        (protocol_copy()[:16], "expected 17"),
        (
            _with_query(16, "laptop", [{"term": "laptop", "field": "category", "value": "laptop"}]),
            "duplicates",
        ),
        (
            _with_query(
                16,
                "HP Laptop",
                [
                    {"term": "hp", "field": "brand", "value": "HP"},
                    {"term": "laptop", "field": "category", "value": "laptop"},
                ],
            ),
            "normalize to the same",
        ),
        (_with_query(16, "anc  headphones"), "normalized form"),
        (_with_query(0, "hp laptop"), "M4 queries"),
    ],
)
def test_bad_query_sets_are_refused(protocol, message):
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.protocol_identity(protocol)


def _expect(query_index, expectations, status="checked"):
    protocol = protocol_copy()
    protocol[query_index]["expectations"] = expectations
    protocol[query_index]["status"] = status
    return protocol


@pytest.mark.parametrize(
    ("protocol", "message"),
    [
        # inferred, not explicit: "iphone" does not contain the term "apple"
        (_expect(3, [{"term": "apple", "field": "brand", "value": "Apple"}]), "explicit term"),
        # a non-approved expectation type (RAM) for "8gb laptop"
        (
            _expect(
                2,
                [
                    {"term": "8gb", "field": "ram_gb", "value": 8},
                    {"term": "laptop", "field": "category", "value": "laptop"},
                ],
            ),
            "not an approved",
        ),
        # an approved term mapped to another value
        (_expect(0, [{"term": "laptop", "field": "category", "value": "phone"}]), "differs"),
        # anc must be the boolean True, not a truthy 1
        (
            _expect(
                16,
                [
                    {"term": "anc", "field": "anc", "value": 1},
                    {"term": "headphones", "field": "category", "value": "headphones"},
                ],
            ),
            "differs",
        ),
        # an explicit approved term left out
        (_expect(1, [{"term": "laptop", "field": "category", "value": "laptop"}]), "exactly"),
        # wrong status for a query with expectations / without
        (
            _expect(
                0, [{"term": "laptop", "field": "category", "value": "laptop"}], "diagnostic_only"
            ),
            "status",
        ),
        (_expect(4, [], "checked"), "status"),
        (_expect(0, [{"term": "laptop", "field": "category", "value": "laptop", "x": 1}]), "keys"),
    ],
)
def test_expectations_must_be_exactly_the_approved_explicit_rules(protocol, message):
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.protocol_identity(protocol)


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("noise cancelling headphones", {"noise cancelling", "headphones"}),
        ("Noise Cancelling headphones", {"noise cancelling", "headphones"}),
        ("anc headphones", {"anc", "headphones"}),
        # not explicit: hyphenated, inflected, partial or glued forms and synonyms
        ("noise-cancelling headphones", {"headphones"}),
        ("noise cancellation headphones", {"headphones"}),
        ("noise canceling headphones", {"headphones"}),
        ("active noise control headphones", {"headphones"}),
        ("noise cancellingx headphones", {"headphones"}),
        ("xnoise cancelling headphones", {"headphones"}),
        ("cancelling headphones", {"headphones"}),
        ("ancient laptop", {"laptop"}),
        ("iphone", set()),
    ],
)
def test_explicit_terms_are_whole_tokens_or_the_exact_approved_phrase(query, terms):
    assert bench.explicit_terms(query) == terms


def test_noise_cancelling_must_expect_anc():
    protocol = protocol_copy()
    assert protocol[11]["query"] == "noise cancelling headphones"
    assert protocol[11]["expectations"] == [
        {"term": "noise cancelling", "field": "anc", "value": True},
        {"term": "headphones", "field": "category", "value": "headphones"},
    ]
    protocol[11]["expectations"] = protocol[11]["expectations"][1:]
    with pytest.raises(bench.BenchmarkError, match="exactly"):
        bench.protocol_identity(protocol)
    protocol = protocol_copy()
    protocol[11]["expectations"][0]["term"] = "anc"  # "anc" is not a token of query 12
    with pytest.raises(bench.BenchmarkError, match="explicit term"):
        bench.protocol_identity(protocol)


def test_only_query_12_changed_from_the_first_reviewed_table():
    first_review = dict(REVIEWED_EXPECTATIONS)
    first_review["noise cancelling headphones"] = [("category", "headphones")]
    changed = [q for q in REVIEWED_QUERIES if first_review[q] != REVIEWED_EXPECTATIONS[q]]
    assert changed == ["noise cancelling headphones"]
    assert QUERY_SET_SHA256 == "ea1df08db89a8bceb0e5b6437f18e75a30aed997fb70d727d080b441ebbd78d3"


def test_pinned_hash_mismatch_is_refused():
    protocol = protocol_copy()
    protocol[0], protocol[1] = protocol[1], protocol[0]
    with pytest.raises(bench.BenchmarkError):
        bench.check_protocol_identity(protocol)
    with pytest.raises(bench.ProvenanceError, match="QUERY_SET_SHA256"):
        bench.check_protocol_identity(query_sha="0" * 64)
    with pytest.raises(bench.ProvenanceError, match="expectation table"):
        bench.check_protocol_identity(table_sha="0" * 64)


# ---- predicate and consistency -------------------------------------------------------------------

LAPTOP = [{"term": "laptop", "field": "category", "value": "laptop"}]
HP_LAPTOP = [{"term": "hp", "field": "brand", "value": "HP"}, *LAPTOP]
ANC_HEADPHONES = [
    {"term": "anc", "field": "anc", "value": True},
    {"term": "headphones", "field": "category", "value": "headphones"},
]


def test_expectation_predicate():
    hp = {"category": "laptop", "brand": "HP", "anc": None}
    assert bench.satisfies(hp, HP_LAPTOP) and bench.satisfies(hp, LAPTOP)
    assert not bench.satisfies({**hp, "brand": "hp"}, HP_LAPTOP)  # exact stored value
    assert not bench.satisfies({**hp, "category": "phone"}, HP_LAPTOP)
    assert not bench.satisfies(None, LAPTOP)  # product without facts
    head = {"category": "headphones", "brand": "Sony", "anc": True}
    assert bench.satisfies(head, ANC_HEADPHONES)
    assert not bench.satisfies({**head, "anc": False}, ANC_HEADPHONES)
    assert not bench.satisfies({**head, "anc": None}, ANC_HEADPHONES)
    assert not bench.satisfies({**head, "anc": 1}, ANC_HEADPHONES)


def test_consistency_uses_a_fixed_top10_denominator_and_skips_diagnostic_queries():
    facts = {"a": {"category": "laptop", "brand": "HP", "anc": None}}
    facts["b"] = {"category": "phone", "brand": "HP", "anc": None}
    assert bench.consistency(["a", "b", "x"], facts, LAPTOP) == {
        "satisfied": 1,
        "denominator": 10,
        "score": "1/10",
    }
    assert bench.consistency(["a"] * 12, facts, LAPTOP)["satisfied"] == 10  # top 10 only
    assert bench.consistency([], facts, LAPTOP)["score"] == "0"
    assert bench.consistency(["a"], facts, []) is None
    assert bench.mean_fraction([Fraction(1, 2), Fraction(1, 3)]) == Fraction(5, 12)
    with pytest.raises(bench.BenchmarkError):
        bench.mean_fraction([])


# ---- selection -------------------------------------------------------------------------------------


def _agg(**values):
    base = {k: Fraction(1, 2) for k in bench.RRF_K_GRID}
    base.update({int(k[1:]): Fraction(v) for k, v in values.items()})
    return base


def test_grid_and_tie_rule_are_predeclared():
    assert bench.RRF_K_GRID == (1, 5, 10, 20, 40, 60, 100)
    assert bench.PREDECLARED["S"]["runs"] == 2 and bench.PREDECLARED["S"]["top_k"] == 10


@pytest.mark.parametrize("winner", [1, 5, 10, 20, 40, 60, 100])
def test_a_unique_maximum_wins_for_every_grid_value(winner):
    decision = bench.decide_rrf_k(_agg(**{f"k{winner}": "9/10"}), deterministic=True)
    assert decision["selected_rrf_k"] == winner and decision["tied_rrf_k"] == [winner]
    assert decision["status"] == "candidate_proposed_pending_review"
    assert decision["best_mean_consistency"] == {"fraction": "9/10", "float": 0.9}
    assert "never changed" in decision["label"]


@pytest.mark.parametrize(
    ("tied", "expected"), [((1, 5), 5), ((5, 40, 20), 40), ((1, 100), 100), ((10, 60), 60)]
)
def test_an_exact_tie_goes_to_the_largest_tied_value(tied, expected):
    decision = bench.decide_rrf_k(_agg(**{f"k{k}": "3/4" for k in tied}), deterministic=True)
    assert decision["selected_rrf_k"] == expected
    assert decision["tied_rrf_k"] == sorted(tied)


def test_all_equal_selects_the_largest_grid_value():
    decision = bench.decide_rrf_k(_agg(), deterministic=True)
    assert decision["selected_rrf_k"] == 100 and decision["tied_rrf_k"] == list(bench.RRF_K_GRID)


def test_exact_fractions_distinguish_values_floats_would_tie():
    near = Fraction(1, 3) + Fraction(1, 10**18)
    assert float(near) == float(Fraction(1, 3))
    aggregates = dict.fromkeys(bench.RRF_K_GRID, Fraction(1, 3)) | {5: near}
    decision = bench.decide_rrf_k(aggregates, deterministic=True)
    assert decision["selected_rrf_k"] == 5


def test_nondeterminism_makes_no_selection():
    decision = bench.decide_rrf_k(_agg(k5="9/10"), deterministic=False)
    assert decision["selected_rrf_k"] is None
    assert decision["status"] == "no_selection_nondeterministic"


def test_aggregates_must_cover_exactly_the_grid():
    aggregates = _agg()
    del aggregates[60]
    with pytest.raises(bench.BenchmarkError, match="grid"):
        bench.decide_rrf_k(aggregates, deterministic=True)
    with pytest.raises(bench.BenchmarkError, match="grid"):
        bench.decide_rrf_k({**_agg(), 7: Fraction(1)}, deterministic=True)


# ---- fusion oracle, overlap and composition ------------------------------------------------------


def _hit(rank, pid, kind):
    return SimpleNamespace(rank=rank, product_id=pid, lexical_score=0.1, dense_score=0.1)


@pytest.mark.parametrize("rrf_k", [1, 5, 60, 100])
def test_oracle_matches_the_production_fusion(rrf_k):
    lexical = ["p3", "p1", "p7", "p2"]
    dense = ["p1", "p9", "p3", "p4", "p5"]
    production = fuse_rrf(
        [_hit(i, p, "lex") for i, p in enumerate(lexical, 1)],
        [_hit(i, p, "den") for i, p in enumerate(dense, 1)],
        rrf_k,
        50,
    )
    assert bench.rrf_oracle(lexical, dense, rrf_k) == [c.product_id for c in production]
    exact = bench.rrf_scores(lexical, dense, rrf_k)
    assert [exact[c.product_id] for c in production] == [c.rrf_score for c in production]


def test_oracle_breaks_exact_ties_by_product_id():
    assert bench.rrf_oracle(["b"], ["a"], 60) == ["a", "b"]


def test_overlap_and_composition():
    assert bench.overlap(["a", "b", "c"], ["a", "c", "d"]) == {
        "unordered": 2,
        "same_position": 1,
        "identical_order": False,
    }
    assert bench.overlap(["a"], ["a"])["identical_order"] is True
    assert bench.overlap([], ["a"]) == {
        "unordered": 0,
        "same_position": 0,
        "identical_order": False,
    }
    assert bench.composition([1, None, 3], [2, 1, None]) == {
        "both": 1,
        "lexical_only": 1,
        "dense_only": 1,
    }
    with pytest.raises(bench.BenchmarkError):
        bench.composition([None], [None])
    assert bench.has_score_tie([0.5, 0.4, 0.5]) and not bench.has_score_tie([0.5, None, None])


def test_endpoint_rotation_schedule():
    assert [bench.endpoint_order(r) for r in (1, 2, 3)] == [
        ("search", "dense", "hybrid"),
        ("dense", "hybrid", "search"),
        ("hybrid", "search", "dense"),
    ]
    for position in range(3):  # each endpoint once in each position
        assert {bench.endpoint_order(r)[position] for r in (1, 2, 3)} == set(bench.ENDPOINTS)
    with pytest.raises(bench.BenchmarkError):
        bench.endpoint_order(4)


def test_latency_protocol_is_predeclared():
    cl = bench.PREDECLARED["CL"]
    assert (cl["runs"], cl["cold_per_query"], cl["warmup_per_query"], cl["samples_per_query"]) == (
        3,
        1,
        20,
        200,
    )
    assert cl["top_k"] == 10 and cl["percentile_method"] == "nearest-rank"
    assert set(bench.ENDPOINTS) == {"search", "dense", "hybrid"}
    assert bench.ENDPOINTS["hybrid"]["stages"] == (
        "lexical_ms",
        "model_load_ms",
        "query_embedding_ms",
        "vector_ms",
        "rrf_ms",
    )


def test_reuses_the_committed_m3_m4_helpers():
    lexical, dense = sys.modules["benchmark_lexical"], sys.modules["benchmark_dense"]
    assert bench.summarize is lexical.summarize and bench.dumps is lexical.dumps
    assert bench.read_samples is lexical.read_samples
    assert bench.source_fingerprint is lexical.source_fingerprint
    for name in (
        "assert_scratch_name",
        "new_scratch_name",
        "scratch_database",
        "install_network_guard",
        "check_model",
        "power_events",
        "committed_catalog_identity",
        "embedding_corpus_digest",
        "ids_sha256",
    ):
        assert getattr(bench, name) is getattr(dense, name)


def test_nearest_rank_percentiles():
    s = bench.summarize([float(v) for v in range(1, 201)])
    assert (s["n"], s["p50"], s["p95"], s["p99"], s["max"]) == (200, 100.0, 190.0, 198.0, 200.0)


# ---- synthetic catalog and records -----------------------------------------------------------------

CATEGORIES = ("laptop", "phone", "shoes", "headphones")
BRANDS = {"laptop": "HP", "phone": "Apple", "shoes": "Nike", "headphones": "Sony"}


def make_facts():
    facts = {}
    for i in range(240):
        category = CATEGORIES[i % 4]
        facts[f"P{i:03d}"] = {
            "category": category,
            "brand": BRANDS[category] if i % 3 == 0 else "Other",
            "anc": (i % 8 == 3) if category == "headphones" else None,
        }
    return facts


FACTS = make_facts()
EXPECT = {e["query"]: e["expectations"] for e in IDENTITY["expectation_table"]}


def source_lists(query, variant=0):
    expectations = EXPECT[query] or LAPTOP
    match = [p for p, f in FACTS.items() if bench.satisfies(f, expectations)]
    other = [p for p, f in FACTS.items() if not bench.satisfies(f, expectations)]
    lexical = match[variant : variant + 10]
    dense = other[:5] + match[variant + 5 : variant + 10] + other[5:20]
    return lexical, dense


def run_record(run, phase):
    catalog = bench.committed_catalog_identity()
    return {
        "record": "run",
        "phase": phase,
        "endpoint_order": list(bench.endpoint_order(run)) if phase == "CL" else [],
        "catalog_datasets": [{k: catalog[k] for k in bench.CATALOG_KEYS}],
        "product_count": 240,
        "embedding_rows": 240,
    }


PINNED = {
    "S": "f7cb2757e669cb58b281c9ac3d8a37b6da7e1a7a",
    "CL": "86f7b68f39155a002390bffc110036ad75dbb326",
}
HEADER_SETTINGS = {
    "S": {
        "search_rrf_k": 60,
        "search_candidate_k": 50,
        "search_lexical_k": 50,
        "search_dense_k": 50,
    },
    "CL": {
        "search_rrf_k": 100,
        "search_candidate_k": 50,
        "search_lexical_k": 50,
        "search_dense_k": 50,
        "rrf_k_status": "provisional",
    },
}


def model_identity():
    from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2 as spec

    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dimension": spec.dimension,
        "max_seq_length": spec.max_seq_length,
        "normalize": spec.normalize,
        "config_sha256": spec.config_sha256(),
        "manifest_problems": [],
    }


def make_header(phase, protocol=None, experiment_id=None):
    return {
        "experiment_id": experiment_id or f"hybrid-m5-{phase.lower()}-20261002T100000Z-0123abcd",
        "generated_at_utc": "2026-10-02T10:00:00+00:00",
        "provenance": {
            "harness_head": "f" * 40,
            "pinned_commit": PINNED[phase],
            "guarded": list(bench.GUARDED_PATHS),
        },
        "model": model_identity(),
        "settings": dict(HEADER_SETTINGS[phase]),
        "source": {"git_working_tree_dirty": False},
        "catalog": bench.committed_catalog_identity(),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_corpus_digest": CORPUS_DIGEST,
        "not_applicable": dict(bench.NOT_APPLICABLE),
        "protocol_identity": bench.check_protocol_identity(),
        "protocol": protocol or bench.PREDECLARED[phase],
    }


def _stream(header, per_run):
    """Numbered records, deep-copied so a test can never mutate the shared synthetic data."""
    records = []
    for run, items in per_run.items():
        for sequence, item in enumerate(items):
            records.append(
                {"experiment_id": header["experiment_id"], "run": run, "sequence": sequence, **item}
            )
    return json.loads(json.dumps(records))


def hits(ids, score_name, score):
    return [
        SimpleNamespace(product_id=p, rank=i, **{score_name: score}) for i, p in enumerate(ids, 1)
    ]


class FakeFetch:
    """Stands in for `source_fetcher`: records every retrieval call."""

    def __init__(self, variants=None, run=1, load_ms=None):
        self.variants, self.run, self.load_ms = variants or {}, run, load_ms
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        lexical, dense = source_lists(query, self.variants.get((self.run, query), 0))
        return (
            SimpleNamespace(hits=hits(lexical, "lexical_score", 0.5)),
            SimpleNamespace(hits=hits(dense, "dense_score", 0.4)),
            self.load_ms,
        )


def s_run_items(run, fetch):
    items = [
        run_record(run, "S"),
        {"record": "facts", "facts": FACTS, "facts_sha256": bench.facts_sha256(FACTS)},
    ]
    bench.collect_phase_s(fetch, bench.QUERIES, items.append, 50)
    return items


def make_s_records(header, variants=None):
    """Phase S records produced by the real collector (production fuse_rrf, fake retrieval)."""
    per_run = {run: s_run_items(run, FakeFetch(variants, run)) for run in (1, 2)}
    return _stream(header, per_run)


SMALL_CL = {**bench.PREDECLARED["CL"], "warmup_per_query": 0, "samples_per_query": 3}
START = datetime(2026, 10, 2, 10, 0, tzinfo=UTC)


def make_cl_records(header, gap_at=None, status="provisional"):
    samples = header["protocol"]["samples_per_query"]
    per_run = {}
    clock = START
    for run in (1, 2, 3):
        items = [
            run_record(run, "CL"),
            {"record": "facts", "facts": FACTS, "facts_sha256": bench.facts_sha256(FACTS)},
        ]
        for endpoint in bench.endpoint_order(run):
            stages = bench.ENDPOINTS[endpoint]["stages"]
            for qi, query in enumerate(bench.QUERIES):
                lexical, dense = source_lists(query)
                hybrid = bench.rrf_oracle(lexical, dense, 100)[:10]
                ids = {"search": lexical, "dense": dense[:10], "hybrid": hybrid}[endpoint]
                result = {
                    "record": "results",
                    "endpoint": endpoint,
                    "query": query,
                    "query_index": qi,
                    "ids": ids,
                    "lexical_scores": (
                        [0.5, 0.5] + [0.1 * i for i in range(len(ids) - 2)]
                        if endpoint == "search"
                        else [0.5 if p in lexical else None for p in ids]
                    ),
                }
                if endpoint == "hybrid":
                    result.update(
                        {
                            "dense_scores": [0.3 if p in dense else None for p in ids],
                            "lexical_ranks": [
                                lexical.index(p) + 1 if p in lexical else None for p in ids
                            ],
                            "dense_ranks": [
                                dense.index(p) + 1 if p in dense else None for p in ids
                            ],
                            "rrf_scores": [0.0] * len(ids),
                            "fusion": {
                                "method": "rrf",
                                "rrf_k": 100,
                                "rrf_k_status": status,
                                "lexical_k": 50,
                                "dense_k": 50,
                                "candidate_k": 50,
                            },
                        }
                    )
                items.append(result)
                for phase, count in (("cold", 1), ("timed", samples)):
                    for i in range(count):
                        clock += timedelta(milliseconds=5)
                        if gap_at == (run, endpoint, qi, phase, i):
                            clock += timedelta(seconds=30)
                        items.append(
                            {
                                "record": "sample",
                                "endpoint": endpoint,
                                "query_index": qi,
                                "query": query,
                                "phase": phase,
                                "sample_index": i,
                                "timestamp_utc": clock.isoformat(timespec="microseconds"),
                                "status_code": 200,
                                "result_count": len(ids),
                                "result_ids_sha256": bench.ids_sha256(ids),
                                "request_ms": 10.0 + i + (100.0 if phase == "cold" else 0),
                                "app_total_ms": 8.0 + i,
                                "serialization_ms": 0.25,
                                "stages": {
                                    s: (
                                        (900.0 if phase == "cold" and qi == 0 else None)
                                        if s == "model_load_ms"
                                        else 1.0 + i
                                    )
                                    for s in stages
                                },
                            }
                        )
        per_run[run] = items
    return _stream(header, per_run)


# ---- Phase S derivation ------------------------------------------------------------------------------


def independent_aggregates():
    """The test's own scoring of the synthetic lists (no harness scoring helpers)."""
    out = {}
    for k in bench.RRF_K_GRID:
        scores = []
        for query in bench.QUERIES:
            if not REVIEWED_EXPECTATIONS[query]:
                continue
            lexical, dense = source_lists(query)
            top = bench.rrf_oracle(lexical, dense, k)[:10]
            good = sum(
                1
                for p in top
                if all(
                    FACTS[p][f] == v and type(FACTS[p][f]) is type(v)
                    for f, v in REVIEWED_EXPECTATIONS[query]
                )
            )
            scores.append(Fraction(good, 10))
        out[k] = sum(scores, Fraction(0)) / len(scores)
    return out


def test_phase_s_derivation_selects_from_checked_queries_only():
    header = make_header("S")
    derived = bench.derive(header, make_s_records(header))
    expected = independent_aggregates()
    assert len(set(expected.values())) > 1  # the synthetic lists make rrf_k matter
    assert derived["decision"] == bench.decide_rrf_k(expected, deterministic=True)
    assert derived["determinism"]["runs_identical"] is True
    run = derived["runs"]["1"]
    assert run["checked_query_count"] == 15
    assert run["per_query"]["zzqxv"]["status"] == "diagnostic_only"
    assert run["per_query"]["zzqxv"]["hybrid"]["60"]["consistency"] == "diagnostic_only"
    entry = run["per_query"]["laptop"]["hybrid"]["1"]
    assert sum(entry["composition"].values()) == 10 == entry["result_count"]
    assert entry["composition"]["both"] <= 5  # five products are in both synthetic lists
    assert run["per_query"]["laptop"]["lexical"]["count"] == 10
    assert run["per_query"]["laptop"]["dense"]["zero_results"] is False


def test_diagnostic_queries_never_change_the_aggregate():
    header = make_header("S")
    baseline = bench.derive(header, make_s_records(header))
    variants = {(run, "zzqxv"): 3 for run in (1, 2)} | {(run, "iphone"): 5 for run in (1, 2)}
    changed = bench.derive(header, make_s_records(header, variants))
    assert changed["decision"] == baseline["decision"]
    assert changed["runs"]["1"]["per_query"]["zzqxv"] != baseline["runs"]["1"]["per_query"]["zzqxv"]


def test_a_determinism_mismatch_refuses_selection():
    header = make_header("S")
    derived = bench.derive(header, make_s_records(header, {(2, "nike shoes"): 1}))
    assert derived["determinism"] == {
        "runs_identical": False,
        "mismatched_queries": ["nike shoes"],
        "rule": derived["determinism"]["rule"],
    }
    assert derived["decision"]["selected_rrf_k"] is None
    assert derived["decision"]["status"] == "no_selection_nondeterministic"


def _mutate(records, predicate, change):
    for record in records:
        if predicate(record):
            change(record)
            return records
    raise AssertionError("no record matched")


@pytest.mark.parametrize(
    ("predicate", "change", "message"),
    [
        (
            lambda r: r["record"] == "hybrid" and r["rrf_k"] == 20,
            lambda r: r["ids"].reverse(),
            "RRF oracle",
        ),
        (
            lambda r: r["record"] == "hybrid",
            lambda r: r["fusion"].update(rrf_k=61),
            "fusion settings",
        ),
        (
            lambda r: r["record"] == "hybrid",
            lambda r: r["dense_ranks"].__setitem__(0, 99),
            "source ranks",
        ),
        (
            lambda r: r["record"] == "hybrid",
            lambda r: r["counts"].update(overlap_count=0),
            "counts",
        ),
        (
            lambda r: r["record"] == "hybrid" and r["rrf_k"] == 5,
            lambda r: r.update(sources_sha256="0" * 64),
            "different source lists",
        ),
        (
            lambda r: r["record"] == "sources",
            lambda r: r["dense"]["scores"].__setitem__(0, 0.9),
            "differ from their hash",
        ),
        (
            lambda r: r["record"] == "hybrid" and r["rrf_k"] == 10,
            lambda r: r["rrf_score_fractions"].__setitem__(0, "1/3"),
            "rrf scores",
        ),
        (
            lambda r: r["record"] == "hybrid" and r["rrf_k"] == 40,
            lambda r: r["rrf_scores"].__setitem__(0, r["rrf_scores"][0] + 1e-12),
            "rrf scores",
        ),
        (
            lambda r: r["record"] == "sources",
            lambda r: r["lexical"]["ranks"].__setitem__(0, 2),
            "ranks 1..n",
        ),
        (
            lambda r: r["record"] == "facts",
            lambda r: r["facts"]["P000"].update(brand="Dell"),
            "facts",
        ),
        (
            lambda r: r["record"] == "run",
            lambda r: r.update(product_count=239),
            "240",
        ),
    ],
)
def test_inconsistent_phase_s_records_are_refused(predicate, change, message):
    header = make_header("S")
    records = _mutate(make_s_records(header), predicate, change)
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.derive(header, records)


# ---- Phase CL derivation -----------------------------------------------------------------------------


def test_phase_cl_recomputes_latency_comparison_and_determinism():
    header = make_header("CL", SMALL_CL)
    records = make_cl_records(header, gap_at=(2, "dense", 4, "timed", 1))
    derived = bench.derive(header, records)
    run1 = derived["runs"]["1"]
    assert run1["endpoint_order"] == ["search", "dense", "hybrid"]
    assert derived["runs"]["3"]["endpoint_order"] == ["hybrid", "search", "dense"]
    hybrid = run1["latency"]["hybrid"]
    assert hybrid["metrics"] == [
        "request_ms",
        "app_total_ms",
        "serialization_ms",
        "lexical_ms",
        "query_embedding_ms",
        "vector_ms",
        "rrf_ms",
    ]
    request = hybrid["overall"]["request_ms"]
    assert request == bench.summarize([10.0, 11.0, 12.0] * 17)  # cold samples excluded
    assert hybrid["per_query"]["laptop"]["cold"]["request_ms"] == 110.0
    assert hybrid["model_load_ms_cold"] == [{"query": "laptop", "ms": 900.0}]
    assert "model_load_ms" not in run1["latency"]["dense"]["overall"]
    assert run1["latency"]["search"]["metrics"][-1] == "lexical_ms"
    assert derived["runs"]["2"]["max_sampling_gap_s"] == pytest.approx(30.005)
    assert derived["runs"]["1"]["max_sampling_gap_s"] == pytest.approx(0.005)
    laptop = run1["comparison"]["per_query"]["laptop"]
    assert laptop["result_count"] == {"search": 10, "dense": 10, "hybrid": 10}
    assert laptop["zero_results"] == {"search": False, "dense": False, "hybrid": False}
    assert laptop["overlap"]["search|dense"]["unordered"] == 5
    assert sum(laptop["hybrid_composition"].values()) == 10
    assert laptop["consistency"]["search"] == {"satisfied": 10, "denominator": 10, "score": "1"}
    assert laptop["consistency"]["dense"]["satisfied"] == 5  # five matching dense neighbours
    assert run1["comparison"]["per_query"]["iphone"]["consistency"]["hybrid"] == "diagnostic_only"
    assert run1["comparison"]["mean_consistency"]["search"] == {"fraction": "1", "float": 1.0}
    assert run1["comparison"]["top10_with_tied_lexical_scores"]["search"] == 17
    assert derived["determinism"]["across_runs_identical"] is True
    assert derived["hybrid_fusion"] == {"rrf_k": 100, "rrf_k_status": "provisional"}
    assert derived["label"] == "provisional, non-authoritative"


@pytest.mark.parametrize(
    ("predicate", "change", "message"),
    [
        (
            lambda r: r["record"] == "sample" and r["phase"] == "timed",
            lambda r: r.update(result_ids_sha256="0" * 64),
            "results changed",
        ),
        (
            lambda r: r["record"] == "sample" and r["endpoint"] == "hybrid",
            lambda r: r["stages"].pop("rrf_ms"),
            "stage fields",
        ),
        (
            lambda r: r["record"] == "sample" and r["phase"] == "timed",
            lambda r: r.update(request_ms=None),
            "not numeric",
        ),
        (
            lambda r: r["record"] == "sample" and r["phase"] == "timed",
            lambda r: r.update(sample_index=7),
            "0..n-1",
        ),
        (
            lambda r: r["record"] == "run" and r["run"] == 2,
            lambda r: r.update(endpoint_order=["search", "dense", "hybrid"]),
            "endpoint order",
        ),
        (
            lambda r: r["record"] == "results" and r["endpoint"] == "hybrid" and r["run"] == 3,
            lambda r: r["fusion"].update(rrf_k=10),
            "fusion settings changed",
        ),
    ],
)
def test_inconsistent_phase_cl_records_are_refused(predicate, change, message):
    header = make_header("CL", SMALL_CL)
    records = _mutate(make_cl_records(header), predicate, change)
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.derive(header, records)


def test_cross_run_result_differences_are_reported_not_hidden():
    header = make_header("CL", SMALL_CL)
    records = make_cl_records(header)
    for record in records:
        if (
            record["run"] == 3
            and record.get("endpoint") == "dense"
            and record.get("query") == "zzqxv"
        ):
            if record["record"] == "results":
                record["ids"] = list(reversed(record["ids"]))
            if record["record"] == "sample":
                record["result_ids_sha256"] = bench.ids_sha256(
                    list(reversed(source_lists("zzqxv")[1][:10]))
                )
    derived = bench.derive(header, records)
    assert derived["determinism"]["across_runs_identical"] is False
    assert derived["determinism"]["mismatched_queries"] == ["zzqxv"]


# ---- record stream integrity -------------------------------------------------------------------------


def test_missing_duplicated_reordered_and_foreign_records_are_refused():
    header = make_header("S")
    records = make_s_records(header)
    with pytest.raises(bench.BenchmarkError, match="missing, duplicated or reordered"):
        bench.derive(header, records[:5] + records[6:])
    with pytest.raises(bench.BenchmarkError, match="missing, duplicated or reordered"):
        bench.derive(header, records[:6] + [records[5]] + records[6:])
    swapped = list(records)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    with pytest.raises(bench.BenchmarkError, match="missing, duplicated or reordered"):
        bench.derive(header, swapped)
    with pytest.raises(bench.BenchmarkError, match="2 runs"):
        bench.derive(header, [r for r in records if r["run"] == 1])
    with pytest.raises(bench.BenchmarkError, match="experiment_id"):
        bench.derive(header, [{**records[0], "experiment_id": "other"}, *records[1:]])
    with pytest.raises(bench.BenchmarkError, match="record type"):
        bench.derive(header, [{**records[0], "record": "sample"}, *records[1:]])


@pytest.mark.parametrize("phase", ["S", "CL"])
def test_an_extra_well_numbered_record_is_refused(phase):
    header = make_header(phase, SMALL_CL if phase == "CL" else None)
    records = make_s_records(header) if phase == "S" else make_cl_records(header)
    last_run1 = max(i for i, r in enumerate(records) if r["run"] == 1)
    extra = {**records[last_run1], "sequence": records[last_run1]["sequence"] + 1}
    extra["query"] = "unlisted query"
    records.insert(last_run1 + 1, extra)
    with pytest.raises(bench.BenchmarkError, match="run 1: expected"):
        bench.derive(header, records)


def test_dropping_a_whole_query_is_refused_even_with_renumbered_sequences():
    header = make_header("S")
    records = [
        r for r in make_s_records(header) if not (r["run"] == 1 and r.get("query") == "nike shoes")
    ]
    sequence = 0
    for record in records:
        if record["run"] == 1:
            record["sequence"] = sequence
            sequence += 1
    with pytest.raises(bench.BenchmarkError, match="run 1: expected 138 records, found 130"):
        bench.derive(header, records)


def test_header_identity_is_required():
    header = make_header("S")
    records = make_s_records(header)
    for key in ("catalog", "embedding_corpus_digest", "not_applicable"):
        broken = {k: v for k, v in header.items() if k != key}
        with pytest.raises(bench.BenchmarkError, match=key):
            bench.derive(broken, records)
    tampered = json.loads(json.dumps(header))
    tampered["protocol_identity"]["expectation_table"][0]["expectations"] = []
    with pytest.raises(bench.BenchmarkError, match="pinned protocol"):
        bench.derive(tampered, records)


# ---- artifacts and --verify --------------------------------------------------------------------------


def write(tmp_path, phase="S"):
    header = make_header(phase, SMALL_CL if phase == "CL" else None)
    records = make_s_records(header) if phase == "S" else make_cl_records(header)
    return bench.write_artifacts(tmp_path, header, records, {"power_events": {}}), header, records


def verify(json_path, phase="S"):
    return bench.verify_artifacts(json_path, SMALL_CL if phase == "CL" else None)


@pytest.mark.parametrize("phase", ["S", "CL"])
def test_artifacts_round_trip_deterministically_and_verify(tmp_path, phase):
    json_path, header, records = write(tmp_path, phase)
    assert verify(json_path, phase) == []
    raw = json_path.with_name(f"{header['experiment_id']}.samples.jsonl")
    body = raw.read_bytes()
    assert b"\r" not in body
    raw_header, raw_records = bench.read_samples(raw)
    assert {k: v for k, v in raw_header.items() if k != "record"} == header
    assert raw_records == json.loads(json.dumps(records))
    bench.write_artifacts(tmp_path, header, list(reversed(records)), {"power_events": {}})
    assert raw.read_bytes() == body  # same bytes whatever the input order
    markdown = json_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "proxy" in markdown and "Not relevance" in markdown


def test_verify_refuses_a_non_predeclared_protocol(tmp_path):
    json_path, _, _ = write(tmp_path, "CL")
    assert "the protocol differs from the predeclared protocol" in bench.verify_artifacts(json_path)


def _raw_lines(json_path):
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    raw = json_path.with_name(summary["raw_samples"]["file"])
    return summary, raw, raw.read_text(encoding="utf-8").splitlines()


def _forge(json_path, raw, lines, summary):
    """Rewrite the raw file and update its recorded hash (a forged-hash tamper)."""
    body = "\n".join(lines) + "\n"
    raw.write_bytes(body.encode("utf-8"))
    summary["raw_samples"]["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    json_path.write_text(json.dumps(summary), encoding="utf-8")


def test_verify_detects_a_tampered_sample_with_and_without_a_forged_hash(tmp_path):
    json_path, _, _ = write(tmp_path, "CL")
    summary, raw, lines = _raw_lines(json_path)
    index = next(i for i, line in enumerate(lines) if '"request_ms":12.0' in line)
    lines[index] = lines[index].replace('"request_ms":12.0', '"request_ms":11.0')
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = verify(json_path, "CL")
    assert "raw samples file hash differs from the summary" in problems
    assert any("does not recompute" in p for p in problems)
    _forge(json_path, raw, lines, summary)
    assert any("does not recompute" in p for p in verify(json_path, "CL"))


@pytest.mark.parametrize("edit", ["drop", "duplicate", "swap"])
def test_verify_detects_missing_duplicated_and_reordered_records(tmp_path, edit):
    json_path, _, _ = write(tmp_path)
    summary, raw, lines = _raw_lines(json_path)
    if edit == "drop":
        del lines[10]
    elif edit == "duplicate":
        lines.insert(10, lines[10])
    else:
        lines[10], lines[11] = lines[11], lines[10]
    _forge(json_path, raw, lines, summary)
    problems = verify(json_path)
    assert any("incomplete or inconsistent" in p for p in problems)


def test_verify_detects_a_tampered_decision_and_selected_value(tmp_path):
    json_path, _, _ = write(tmp_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    summary["derived"]["decision"]["selected_rrf_k"] = 60
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    assert any("does not recompute" in p for p in verify(json_path))


def test_verify_detects_a_changed_summary_header(tmp_path):
    json_path, _, _ = write(tmp_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    summary["header"]["provenance"]["pinned_commit"] = "0" * 40
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    assert "raw header differs from the summary header" in verify(json_path)


def test_cli_verify_exit_codes(tmp_path, capsys):
    json_path, _, _ = write(tmp_path)
    assert bench.main(["--verify", str(json_path)]) == 0
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    summary["derived"]["determinism"]["runs_identical"] = False
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    assert bench.main(["--verify", str(json_path)]) == 1
    assert "does not recompute" in capsys.readouterr().out


# ---- phase-specific header identity ------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["S", "CL"])
def test_a_correct_phase_header_has_no_problems(phase):
    assert bench.check_phase_header(make_header(phase)) == []


def test_header_settings_keep_the_historical_phase_s_shape_and_record_the_cl_status():
    fixed_s = {**APPROVED_SETTINGS, "rrf_k_status": "candidate_pending_selection"}
    fixed_cl = {**CL_APPROVED_SETTINGS, "rrf_k_status": "provisional"}
    assert bench.header_settings("S", fixed_s) == HEADER_SETTINGS["S"]
    assert bench.header_settings("CL", fixed_cl) == HEADER_SETTINGS["CL"]


def _edited(phase, path, value):
    header = make_header(phase)
    *parents, key = path
    target = header
    for part in parents:
        target = target[part]
    if value is KeyError:
        target.pop(key)
    else:
        target[key] = value
    return header


@pytest.mark.parametrize(
    ("phase", "path", "value", "message"),
    [
        ("CL", ("provenance", "pinned_commit"), KeyError, "pinned commit"),
        ("CL", ("provenance", "pinned_commit"), "0" * 40, "pinned commit"),
        ("CL", ("provenance", "pinned_commit"), PINNED["S"], "pinned commit"),
        ("S", ("provenance", "pinned_commit"), PINNED["CL"], "pinned commit"),
        ("CL", ("provenance", "guarded"), ["src/"], "guarded paths"),
        ("CL", ("provenance", "harness_head"), KeyError, "harness HEAD"),
        ("CL", ("settings", "search_rrf_k"), 60, "settings"),
        ("CL", ("settings", "search_candidate_k"), 40, "settings"),
        ("CL", ("settings", "rrf_k_status"), "candidate_pending_selection", "settings"),
        ("CL", ("settings", "rrf_k_status"), KeyError, "settings"),
        ("S", ("settings", "search_rrf_k"), 100, "settings"),
        ("S", ("settings", "rrf_k_status"), "provisional", "settings"),
        ("CL", ("experiment_id",), "hybrid-m5-s-20261002T100000Z-0123abcd", "experiment id"),
        ("S", ("experiment_id",), "hybrid-m5-cl-20261002T100000Z-0123abcd", "experiment id"),
        ("CL", ("model", "revision"), "0" * 40, "model identity"),
        ("CL", ("model",), KeyError, "model identity"),
        ("CL", ("protocol",), {"phase": "X"}, "unknown protocol phase"),
    ],
)
def test_a_wrong_or_swapped_phase_header_is_refused(phase, path, value, message):
    problems = bench.check_phase_header(_edited(phase, path, value))
    assert any(message in p for p in problems), problems


@pytest.mark.parametrize(("phase", "other"), [("S", "CL"), ("CL", "S")])
def test_an_artifact_relabelled_as_the_other_phase_does_not_verify(tmp_path, phase, other):
    json_path, _, _ = write(tmp_path, phase)
    summary, raw, lines = _raw_lines(json_path)
    header = json.loads(lines[0])
    header["protocol"] = SMALL_CL if other == "CL" else bench.PREDECLARED["S"]
    lines[0] = bench.dumps(header)
    summary["header"] = {k: v for k, v in header.items() if k != "record"}
    _forge(json_path, raw, lines, summary)
    problems = verify(json_path, other)
    assert any(f"not the Phase {other} commit" in p for p in problems), problems
    assert any(f"approved Phase {other} settings" in p for p in problems), problems
    assert any(f"not a Phase {other} harness id" in p for p in problems), problems


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rrf_k": 60}, "fusion rrf_k or status"),
        ({"rrf_k_status": "candidate_pending_selection"}, "fusion rrf_k or status"),
    ],
)
def test_cl_responses_with_another_fusion_setting_do_not_verify(tmp_path, change, message):
    header = make_header("CL", SMALL_CL)
    records = make_cl_records(header)
    for record in records:
        if record["record"] == "results" and record["endpoint"] == "hybrid":
            record["fusion"].update(change)
    json_path = bench.write_artifacts(tmp_path, header, records, {"power_events": {}})
    assert any(message in p for p in verify(json_path, "CL"))


# ---- provenance and safety -----------------------------------------------------------------------------

HEAD = "e" * 40


def fake_git(status=b"", diff=b"", ancestor=True, calls=None, ls_files=b"", ignored=True):
    def git(root, *args):
        if calls is not None:
            calls.append(args)
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return (HEAD + "\n").encode()
        if args[0] == "merge-base":
            if not ancestor:
                raise bench.GitError("not an ancestor")
            return b""
        if args[0] == "diff":
            return diff
        if args[0] == "ls-files":
            return ls_files
        if args[0] == "check-ignore":
            if not ignored:
                raise bench.GitError("not ignored")
            return b""
        raise AssertionError(args)

    return git


def test_core_commit_and_guarded_paths_are_pinned():
    assert bench.CORE_COMMIT == "f7cb2757e669cb58b281c9ac3d8a37b6da7e1a7a"
    assert bench.RUNTIME_COMMIT == "86f7b68f39155a002390bffc110036ad75dbb326"
    assert bench.PHASE_COMMITS == PINNED
    assert bench.GUARDED_PATHS == (
        "src/",
        "migrations/",
        "data/seed/",
        "pyproject.toml",
        "uv.lock",
        "docker-compose.yml",
        "alembic.ini",
        "Dockerfile",
        ".env.example",
    )


def test_clean_tree_with_identical_guarded_paths_is_accepted():
    calls = []
    result = bench.check_provenance(ROOT, "S", git=fake_git(calls=calls))
    assert result == {
        "harness_head": HEAD,
        "pinned_commit": bench.CORE_COMMIT,
        "guarded": list(bench.GUARDED_PATHS),
    }
    diff_call = next(c for c in calls if c[0] == "diff")
    assert diff_call[:4] == ("diff", "--name-only", bench.CORE_COMMIT, HEAD)
    assert diff_call[5:] == bench.GUARDED_PATHS


@pytest.mark.parametrize("status", [b" M src/x.py\n", b"?? scripts/new.py\n", b"M  docs/a.md\n"])
def test_a_dirty_tree_is_refused(status):
    with pytest.raises(bench.ProvenanceError, match="not clean"):
        bench.check_provenance(ROOT, "S", git=fake_git(status=status))


@pytest.mark.parametrize(
    "changed",
    [
        "src/ecommerce_search/search/hybrid.py",
        "src/ecommerce_search/api/hybrid.py",
        "src/ecommerce_search/config/settings.py",
        "migrations/versions/0005_x.py",
        "data/seed/catalog_seed_v1.jsonl",
        "pyproject.toml",
        "uv.lock",
        "docker-compose.yml",
        "alembic.ini",
        "Dockerfile",
        ".env.example",
    ],
)
def test_every_protected_production_path_difference_is_refused(changed):
    with pytest.raises(bench.ProvenanceError, match="runtime-affecting"):
        bench.check_provenance(ROOT, "S", git=fake_git(diff=f"{changed}\n".encode()))


def test_the_pinned_commit_must_be_an_ancestor():
    with pytest.raises(bench.ProvenanceError, match="not an ancestor"):
        bench.check_provenance(ROOT, "S", git=fake_git(ancestor=False))


def test_phase_cl_is_checked_against_the_reviewed_runtime_commit():
    calls = []
    result = bench.check_provenance(ROOT, "CL", git=fake_git(calls=calls))
    assert result == {
        "harness_head": HEAD,
        "pinned_commit": "86f7b68f39155a002390bffc110036ad75dbb326",
        "guarded": list(bench.GUARDED_PATHS),
    }
    assert [c[0] for c in calls] == ["status", "rev-parse", "merge-base", "diff"]
    assert calls[2] == ("merge-base", "--is-ancestor", PINNED["CL"], HEAD)
    diff_call = calls[3]
    assert diff_call[:4] == ("diff", "--name-only", PINNED["CL"], HEAD)
    assert diff_call[5:] == bench.GUARDED_PATHS
    with pytest.raises(bench.BenchmarkError, match="unknown phase"):
        bench.check_provenance(ROOT, "X", git=fake_git())


@pytest.mark.parametrize(
    ("git", "message"),
    [
        (fake_git(ancestor=False), "missing or not an ancestor"),
        (fake_git(status=b"?? scripts/new.py\n"), "not clean"),
        (fake_git(status=b" M scripts/benchmark_hybrid.py\n"), "not clean"),
        (fake_git(diff=b"src/ecommerce_search/search/hybrid.py\n"), "runtime-affecting"),
        (fake_git(diff=b"src/ecommerce_search/config/settings.py\n"), "runtime-affecting"),
        (fake_git(diff=b"uv.lock\n"), "runtime-affecting"),
    ],
)
def test_phase_cl_refuses_a_non_ancestor_dirty_or_runtime_different_tree(git, message):
    with pytest.raises(bench.ProvenanceError, match=message):
        bench.check_provenance(ROOT, "CL", git=git)


def test_a_missing_runtime_pin_is_refused_before_any_git_call(monkeypatch):
    monkeypatch.setattr(bench, "PHASE_COMMITS", {**bench.PHASE_COMMITS, "CL": None})
    calls = []
    with pytest.raises(bench.ProvenanceError, match="no pinned runtime commit"):
        bench.check_provenance(ROOT, "CL", git=fake_git(calls=calls))
    assert calls == []


# ---- fixed Phase-S settings and the shared preflight (fakes only) ----------------------------------

APPROVED_SETTINGS = {
    "search_lexical_k": 50,
    "search_dense_k": 50,
    "search_candidate_k": 50,
    "search_rrf_k": 60,
}


def fake_settings(models_dir, **overrides):
    from ecommerce_search.embeddings.spec import ALL_MINILM_L6_V2

    values = {**APPROVED_SETTINGS, **overrides}
    return SimpleNamespace(
        **values,
        postgres_db="ecommerce_search",
        database_url=lambda database=None: f"unused://{database}",
        resolved_models_dir=lambda: models_dir,
        embedding_spec=lambda: ALL_MINILM_L6_V2,
    )


# The model manifest check is shared with the dense harness and raises its ProvenanceError.
PROVENANCE_ERRORS = (bench.ProvenanceError, sys.modules["benchmark_dense"].ProvenanceError)


class SpyVerify:
    def __init__(self, problems=()):
        self.problems, self.calls = list(problems), []

    def __call__(self, models_dir, model_id, revision):
        self.calls.append((models_dir, model_id, revision))
        return self.problems


@pytest.fixture
def pre_selection_status(monkeypatch):
    """Phase S was approved for the pre-selection runtime; reproduce its status for these fakes."""
    import ecommerce_search.search.hybrid as hybrid

    monkeypatch.setattr(hybrid, "RRF_K_STATUS", bench.PHASE_S_RRF_K_STATUS)


def test_the_approved_phase_s_settings_are_pinned_to_the_production_names():
    from ecommerce_search.config import Settings

    assert bench.PHASE_S_SETTINGS == APPROVED_SETTINGS
    assert set(APPROVED_SETTINGS) <= set(Settings.model_fields)
    assert bench.PHASE_S_FROZEN == {
        "top_k": 10,
        "source_depth": 50,
        "rrf_k_grid": [1, 5, 10, 20, 40, 60, 100],
    }
    assert bench.PHASE_S_RRF_K_STATUS == "candidate_pending_selection"


def test_phase_s_refuses_the_adopted_provisional_runtime_defaults(monkeypatch):
    from ecommerce_search.config import Settings
    from ecommerce_search.search.hybrid import RRF_K_STATUS

    for name in APPROVED_SETTINGS:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-password")
    defaults = Settings(_env_file=None)
    assert {n: getattr(defaults, n) for n in APPROVED_SETTINGS} == {
        **APPROVED_SETTINGS,
        "search_rrf_k": 100,
    }
    assert RRF_K_STATUS == "provisional"
    with pytest.raises(bench.ProvenanceError, match="approved settings: search_rrf_k=100"):
        bench.check_phase_settings("S", defaults, RRF_K_STATUS)
    with pytest.raises(bench.ProvenanceError, match="only before selection"):
        bench.check_phase_settings("S", fake_settings(None), RRF_K_STATUS)


def test_approved_phase_s_settings_pass(tmp_path):
    result = bench.check_phase_settings("S", fake_settings(tmp_path), "candidate_pending_selection")
    assert result == {**APPROVED_SETTINGS, "rrf_k_status": "candidate_pending_selection"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("search_lexical_k", 49),
        ("search_lexical_k", 100),
        ("search_dense_k", 10),
        ("search_dense_k", 51),
        ("search_candidate_k", 10),
        ("search_candidate_k", 100),
        ("search_rrf_k", 1),
        ("search_rrf_k", 40),
        ("search_rrf_k", 61),
        ("search_rrf_k", "60"),
        ("search_rrf_k", None),
    ],
)
def test_every_non_approved_phase_s_setting_is_refused(tmp_path, name, value):
    settings = fake_settings(tmp_path, **{name: value})
    with pytest.raises(bench.ProvenanceError, match=f"approved settings: {name}="):
        bench.check_phase_settings("S", settings, "candidate_pending_selection")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SEARCH_LEXICAL_K", "60"),
        ("SEARCH_DENSE_K", "40"),
        ("SEARCH_CANDIDATE_K", "20"),
        ("SEARCH_RRF_K", "20"),
    ],
)
def test_non_default_environment_values_are_refused_not_recorded(monkeypatch, variable, value):
    from ecommerce_search.config import Settings

    for name, approved_value in APPROVED_SETTINGS.items():
        monkeypatch.setenv(name.upper(), str(approved_value))
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-password")
    approved = Settings(_env_file=None)
    bench.check_phase_settings("S", approved, "candidate_pending_selection")
    monkeypatch.setenv(variable, value)
    changed = Settings(_env_file=None)
    with pytest.raises(bench.ProvenanceError, match=variable.lower()):
        bench.check_phase_settings("S", changed, "candidate_pending_selection")


def test_a_selected_rrf_k_status_is_refused(tmp_path):
    with pytest.raises(bench.ProvenanceError, match="only before selection"):
        bench.check_phase_settings("S", fake_settings(tmp_path), "provisional")


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("TOP_K", 20),
        ("SOURCE_DEPTH", 40),
        ("RRF_K_GRID", (1, 5, 10, 20, 40, 60)),
        ("RRF_K_GRID", (1, 5, 10, 20, 40, 60, 100, 200)),
        ("PREDECLARED", {**bench.PREDECLARED, "S": {**bench.PREDECLARED["S"], "top_k": 5}}),
        (
            "PREDECLARED",
            {**bench.PREDECLARED, "S": {**bench.PREDECLARED["S"], "rrf_k_grid": [60]}},
        ),
    ],
)
def test_a_changed_phase_s_protocol_constant_is_refused(tmp_path, monkeypatch, attribute, value):
    monkeypatch.setattr(bench, attribute, value)
    with pytest.raises(bench.ProvenanceError, match="differs from approval"):
        bench.check_phase_settings("S", fake_settings(tmp_path), "candidate_pending_selection")


CL_APPROVED_SETTINGS = {
    "search_lexical_k": 50,
    "search_dense_k": 50,
    "search_candidate_k": 50,
    "search_rrf_k": 100,
}


def cl_settings(models_dir, **overrides):
    return fake_settings(models_dir, **{**CL_APPROVED_SETTINGS, **overrides})


def test_the_approved_phase_cl_settings_and_protocol_are_pinned():
    from ecommerce_search.config import Settings

    assert bench.PHASE_CL_SETTINGS == CL_APPROVED_SETTINGS
    assert set(CL_APPROVED_SETTINGS) <= set(Settings.model_fields)
    assert bench.PHASE_CL_RRF_K_STATUS == "provisional"
    assert bench.PHASE_CL_FROZEN == {
        "top_k": 10,
        "runs": 3,
        "cold_per_query": 1,
        "warmup_per_query": 20,
        "samples_per_query": 200,
        "endpoint_schedule": {
            "1": ["search", "dense", "hybrid"],
            "2": ["dense", "hybrid", "search"],
            "3": ["hybrid", "search", "dense"],
        },
    }


def test_phase_cl_accepts_the_reviewed_runtime_defaults(monkeypatch):
    from ecommerce_search.config import Settings
    from ecommerce_search.search.hybrid import RRF_K_STATUS

    for name in CL_APPROVED_SETTINGS:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-password")
    defaults = Settings(_env_file=None)
    result = bench.check_phase_settings("CL", defaults, RRF_K_STATUS)
    assert result == {**CL_APPROVED_SETTINGS, "rrf_k_status": "provisional"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("search_lexical_k", 49),
        ("search_lexical_k", 100),
        ("search_dense_k", 10),
        ("search_dense_k", 51),
        ("search_candidate_k", 10),
        ("search_candidate_k", 100),
        ("search_rrf_k", 60),
        ("search_rrf_k", 99),
        ("search_rrf_k", 101),
        ("search_rrf_k", "100"),
        ("search_rrf_k", None),
    ],
)
def test_every_non_approved_phase_cl_setting_is_refused(tmp_path, name, value):
    with pytest.raises(
        bench.ProvenanceError, match=f"Phase CL requires the approved settings: {name}="
    ):
        bench.check_phase_settings("CL", cl_settings(tmp_path, **{name: value}), "provisional")


@pytest.mark.parametrize("status", ["candidate_pending_selection", "final", ""])
def test_phase_cl_refuses_any_other_rrf_k_status(tmp_path, status):
    with pytest.raises(bench.ProvenanceError, match="Phase CL runs only at the adopted runtime"):
        bench.check_phase_settings("CL", cl_settings(tmp_path), status)


def test_phase_s_settings_never_pass_phase_cl_and_the_reverse(tmp_path):
    with pytest.raises(bench.ProvenanceError, match="search_rrf_k=60"):
        bench.check_phase_settings("CL", fake_settings(tmp_path), "provisional")
    with pytest.raises(bench.ProvenanceError, match="search_rrf_k=100"):
        bench.check_phase_settings("S", cl_settings(tmp_path), "candidate_pending_selection")
    with pytest.raises(bench.BenchmarkError, match="unknown phase"):
        bench.check_phase_settings("X", cl_settings(tmp_path), "provisional")


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("TOP_K", 20),
        ("ENDPOINT_SCHEDULE", {**bench.ENDPOINT_SCHEDULE, 2: ("search", "dense", "hybrid")}),
        ("PREDECLARED", {**bench.PREDECLARED, "CL": {**bench.PREDECLARED["CL"], "runs": 2}}),
        (
            "PREDECLARED",
            {**bench.PREDECLARED, "CL": {**bench.PREDECLARED["CL"], "warmup_per_query": 5}},
        ),
        (
            "PREDECLARED",
            {**bench.PREDECLARED, "CL": {**bench.PREDECLARED["CL"], "samples_per_query": 100}},
        ),
        (
            "PREDECLARED",
            {**bench.PREDECLARED, "CL": {**bench.PREDECLARED["CL"], "cold_per_query": 2}},
        ),
        ("PREDECLARED", {**bench.PREDECLARED, "CL": {**bench.PREDECLARED["CL"], "top_k": 5}}),
        (
            "PREDECLARED",
            {
                **bench.PREDECLARED,
                "CL": {**bench.PREDECLARED["CL"], "endpoint_schedule": {"1": ["search"]}},
            },
        ),
    ],
)
def test_a_changed_phase_cl_protocol_constant_is_refused(tmp_path, monkeypatch, attribute, value):
    monkeypatch.setattr(bench, attribute, value)
    with pytest.raises(bench.ProvenanceError, match="Phase CL .* differs from approval"):
        bench.check_phase_settings("CL", cl_settings(tmp_path), "provisional")


@pytest.fixture
def adopted_status(monkeypatch):
    import ecommerce_search.search.hybrid as hybrid

    monkeypatch.setattr(hybrid, "RRF_K_STATUS", bench.PHASE_CL_RRF_K_STATUS)


def test_preflight_passes_phase_cl_with_the_approved_runtime(tmp_path, adopted_status):
    calls, verify = [], SpyVerify()
    result = bench.preflight("CL", cl_settings(tmp_path), git=fake_git(calls=calls), verify=verify)
    assert result["provenance"]["pinned_commit"] == PINNED["CL"]
    assert result["settings"] == {**CL_APPROVED_SETTINGS, "rrf_k_status": "provisional"}
    assert result["model"] == model_identity()
    assert [c[0] for c in calls] == ["status", "rev-parse", "merge-base", "diff"]
    assert len(verify.calls) == 1


@pytest.mark.parametrize(
    ("git", "overrides", "status", "problems", "message"),
    [
        (fake_git(ancestor=False), {}, "provisional", (), "not an ancestor"),
        (fake_git(diff=b"src/ecommerce_search/api/hybrid.py\n"), {}, "provisional", (), "runtime"),
        (fake_git(), {"search_rrf_k": 60}, "provisional", (), "approved settings"),
        (fake_git(), {}, "candidate_pending_selection", (), "adopted runtime"),
        (fake_git(), {}, "provisional", ("sha256 mismatch",), "failed verification"),
    ],
)
def test_preflight_refuses_each_failed_phase_cl_invariant(
    tmp_path, monkeypatch, git, overrides, status, problems, message
):
    import ecommerce_search.search.hybrid as hybrid

    monkeypatch.setattr(hybrid, "RRF_K_STATUS", status)
    verify = SpyVerify(problems)
    with pytest.raises(PROVENANCE_ERRORS, match=message):
        bench.preflight("CL", cl_settings(tmp_path, **overrides), git=git, verify=verify)
    assert len(verify.calls) == (1 if problems else 0)  # the manifest is checked last


def test_preflight_passes_only_with_every_invariant(tmp_path, pre_selection_status):
    calls, verify = [], SpyVerify()
    result = bench.preflight("S", fake_settings(tmp_path), git=fake_git(calls=calls), verify=verify)
    assert result["provenance"]["pinned_commit"] == bench.CORE_COMMIT
    assert result["identity"] == IDENTITY
    assert result["settings"]["search_rrf_k"] == 60
    assert result["model"]["manifest_problems"] == []
    assert [c[0] for c in calls] == ["status", "rev-parse", "merge-base", "diff"]
    assert len(verify.calls) == 1 and verify.calls[0][0] == tmp_path


def test_preflight_refuses_phase_s_at_the_adopted_runtime(tmp_path, adopted_status):
    verify = SpyVerify()
    with pytest.raises(bench.ProvenanceError, match="approved settings: search_rrf_k=100"):
        bench.preflight("S", cl_settings(tmp_path), git=fake_git(), verify=verify)
    with pytest.raises(bench.ProvenanceError, match="only before selection"):
        bench.preflight("S", fake_settings(tmp_path), git=fake_git(), verify=verify)
    assert verify.calls == []


@pytest.mark.parametrize(
    ("git", "overrides", "problems", "message"),
    [
        (fake_git(status=b" M src/x.py\n"), {}, (), "not clean"),
        (fake_git(ancestor=False), {}, (), "not an ancestor"),
        (fake_git(diff=b"src/ecommerce_search/search/hybrid.py\n"), {}, (), "runtime-affecting"),
        (fake_git(), {"search_rrf_k": 40}, (), "approved settings"),
        (fake_git(), {}, ("sha256 mismatch",), "failed verification"),
    ],
)
def test_preflight_refuses_each_failed_invariant(
    tmp_path, pre_selection_status, git, overrides, problems, message
):
    verify = SpyVerify(problems)
    with pytest.raises(PROVENANCE_ERRORS, match=message):
        bench.preflight("S", fake_settings(tmp_path, **overrides), git=git, verify=verify)
    assert len(verify.calls) == (1 if problems else 0)  # the manifest is checked last


def test_preflight_refuses_changed_protocol_hashes(tmp_path, monkeypatch):
    original = bench.check_protocol_identity.__defaults__
    monkeypatch.setattr(
        bench.check_protocol_identity, "__defaults__", (original[0], original[1], "0" * 64)
    )
    verify = SpyVerify()
    with pytest.raises(bench.ProvenanceError, match="expectation table"):
        bench.preflight("S", fake_settings(tmp_path), git=fake_git(), verify=verify)
    assert verify.calls == []


# ---- direct child invocation repeats every applicable refusal (fakes only) -----------------------


class SetupReached(Exception):
    """Raised by a stub at the first database or model step: preflight let the child through."""


@pytest.fixture
def child_env(monkeypatch, tmp_path, pre_selection_status):
    import sqlalchemy

    import ecommerce_search.config as config
    import ecommerce_search.embeddings.fetch as fetch
    import ecommerce_search.embeddings.sentence_transformers_provider as provider

    def setup_reached(*args, **kwargs):
        raise SetupReached

    env = SimpleNamespace(
        git_calls=[], verify=SpyVerify(), guard=[], settings=fake_settings(tmp_path)
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(bench, "install_network_guard", lambda: env.guard.append(True))
    monkeypatch.setattr(bench, "_git", fake_git(calls=env.git_calls))
    monkeypatch.setattr(config, "get_settings", lambda: env.settings)
    monkeypatch.setattr(fetch, "verify_manifest", env.verify)
    monkeypatch.setattr(sqlalchemy, "create_engine", setup_reached)
    monkeypatch.setattr(provider.SentenceTransformerEmbedder, "load", setup_reached)
    return env


def child_argv(tmp_path, phase="S", **overrides):
    args = {
        "--child-phase": phase,
        "--run": "1",
        "--database": "ecommerce_search_bench_0123456789ab",
        "--experiment-id": f"hybrid-m5-{phase.lower()}-20261002T120000Z-0123abcd",
        "--out": str(tmp_path / "run1.jsonl"),
    }
    if phase == "CL":
        args.update({"--warmup": "20", "--samples": "200"})
    args.update(overrides)
    return ["--child", *(x for k, v in args.items() if v is not None for x in (k, v))]


@pytest.fixture
def cl_child_env(child_env, tmp_path, adopted_status):
    child_env.settings = cl_settings(tmp_path)
    return child_env


def test_an_approved_direct_child_cl_reaches_setup_only_after_every_check(cl_child_env, tmp_path):
    with pytest.raises(SetupReached):
        bench.main(child_argv(tmp_path, "CL"))
    assert cl_child_env.guard == [True]
    calls = cl_child_env.git_calls
    assert [c[0] for c in calls] == ["status", "rev-parse", "merge-base", "diff"]
    assert calls[2] == ("merge-base", "--is-ancestor", PINNED["CL"], HEAD)
    assert calls[3][2] == PINNED["CL"]
    assert len(cl_child_env.verify.calls) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"git": fake_git(status=b"?? scripts/new.py\n")}, "not clean"),
        ({"git": fake_git(ancestor=False)}, "missing or not an ancestor"),
        ({"git": fake_git(diff=b"src/ecommerce_search/search/hybrid.py\n")}, "runtime-affecting"),
        ({"pin": None}, "no pinned runtime commit"),
        ({"settings": {"search_lexical_k": 49}}, "search_lexical_k"),
        ({"settings": {"search_dense_k": 49}}, "search_dense_k"),
        ({"settings": {"search_candidate_k": 49}}, "search_candidate_k"),
        ({"settings": {"search_rrf_k": 60}}, "search_rrf_k"),
        ({"status": "candidate_pending_selection"}, "adopted runtime"),
        ({"problems": ["sha256 mismatch"]}, "failed verification"),
        ({"table_sha": "0" * 64}, "expectation table"),
        ({"query_sha": "0" * 64}, "QUERY_SET_SHA256"),
    ],
)
def test_direct_child_cl_cannot_bypass_provenance_protocol_settings_or_model_checks(
    cl_child_env, tmp_path, monkeypatch, change, message
):
    import ecommerce_search.search.hybrid as hybrid

    if "git" in change:
        monkeypatch.setattr(bench, "_git", change["git"])
    if "pin" in change:
        monkeypatch.setattr(bench, "PHASE_COMMITS", {**bench.PHASE_COMMITS, "CL": change["pin"]})
    if "settings" in change:
        cl_child_env.settings = cl_settings(tmp_path, **change["settings"])
    if "status" in change:
        monkeypatch.setattr(hybrid, "RRF_K_STATUS", change["status"])
    if "problems" in change:
        cl_child_env.verify.problems = change["problems"]
    if "table_sha" in change or "query_sha" in change:
        protocol, query_sha, table_sha = bench.check_protocol_identity.__defaults__
        monkeypatch.setattr(
            bench.check_protocol_identity,
            "__defaults__",
            (protocol, change.get("query_sha", query_sha), change.get("table_sha", table_sha)),
        )
    with pytest.raises(PROVENANCE_ERRORS, match=message):
        bench.main(child_argv(tmp_path, "CL"))
    assert not (tmp_path / "run1.jsonl").exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ({"--run": "0"}, "child run"),
        ({"--run": "4"}, "child run"),
        ({"--warmup": "0", "--samples": "0"}, "sampling"),
        ({"--warmup": "19"}, "sampling"),
        ({"--experiment-id": "hybrid-m5-s-20261002T120000Z-0123abcd"}, "experiment id"),
        ({"--out": str(ROOT / "run1.jsonl")}, "outside the repository"),
        ({"--database": "ecommerce_search"}, "development database"),
        ({"--database": "ecommerce_search_bench_xyz"}, "scratch database names"),
    ],
)
def test_direct_child_cl_arguments_are_refused_before_setup(cl_child_env, tmp_path, argv, message):
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.main(child_argv(tmp_path, "CL", **argv))
    assert not (tmp_path / "run1.jsonl").exists()


def test_an_approved_child_reaches_setup_only_after_every_check(child_env, tmp_path):
    with pytest.raises(SetupReached):
        bench.main(child_argv(tmp_path))
    assert child_env.guard == [True]
    assert [c[0] for c in child_env.git_calls] == ["status", "rev-parse", "merge-base", "diff"]
    assert len(child_env.verify.calls) == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ({"--run": "0"}, "child run"),
        ({"--run": "3"}, "child run"),
        ({"--warmup": "20", "--samples": "200"}, "sampling"),
        ({"--child-phase": "X"}, "known phase"),
        ({"--experiment-id": "hybrid-m5-cl-20261002T120000Z-0123abcd"}, "experiment id"),
        ({"--experiment-id": None}, "experiment id"),
        ({"--out": str(ROOT / "run1.jsonl")}, "outside the repository"),
        ({"--out": None}, "outside the repository"),
        ({"--database": "ecommerce_search"}, "development database"),
        ({"--database": "ecommerce_search_bench_xyz"}, "scratch database names"),
        ({"--database": None}, "scratch database names"),
    ],
)
def test_direct_child_arguments_are_refused_before_setup(child_env, tmp_path, argv, message):
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.main(child_argv(tmp_path, **argv))


def test_direct_child_cl_sampling_must_be_predeclared(child_env, tmp_path):
    with pytest.raises(bench.BenchmarkError, match="sampling"):
        bench.main(child_argv(tmp_path, "CL", **{"--samples": "5"}))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"git": fake_git(status=b"?? scripts/new.py\n")}, "not clean"),
        ({"git": fake_git(ancestor=False)}, "not an ancestor"),
        ({"git": fake_git(diff=b"src/ecommerce_search/config/settings.py\n")}, "runtime-affecting"),
        ({"settings": {"search_lexical_k": 49}}, "search_lexical_k"),
        ({"settings": {"search_dense_k": 49}}, "search_dense_k"),
        ({"settings": {"search_candidate_k": 49}}, "search_candidate_k"),
        ({"settings": {"search_rrf_k": 40}}, "search_rrf_k"),
        ({"status": "provisional"}, "only before selection"),
        ({"problems": ["sha256 mismatch"]}, "failed verification"),
        ({"table_sha": "0" * 64}, "expectation table"),
        ({"query_sha": "0" * 64}, "QUERY_SET_SHA256"),
    ],
)
def test_direct_child_s_cannot_bypass_provenance_protocol_settings_or_model_checks(
    child_env, tmp_path, monkeypatch, change, message
):
    import ecommerce_search.search.hybrid as hybrid

    if "git" in change:
        monkeypatch.setattr(bench, "_git", change["git"])
    if "settings" in change:
        child_env.settings = fake_settings(tmp_path, **change["settings"])
    if "status" in change:
        monkeypatch.setattr(hybrid, "RRF_K_STATUS", change["status"])
    if "problems" in change:
        child_env.verify.problems = change["problems"]
    if "table_sha" in change or "query_sha" in change:
        protocol, query_sha, table_sha = bench.check_protocol_identity.__defaults__
        monkeypatch.setattr(
            bench.check_protocol_identity,
            "__defaults__",
            (protocol, change.get("query_sha", query_sha), change.get("table_sha", table_sha)),
        )
    with pytest.raises(PROVENANCE_ERRORS, match=message):
        bench.main(child_argv(tmp_path))
    assert not (tmp_path / "run1.jsonl").exists()


def test_output_directory_must_be_ignored_untracked_and_under_data_processed():
    good = ROOT / "data" / "processed" / "hybrid_benchmark"
    assert bench.check_output_dir(ROOT, good, git=fake_git()) == good.resolve()
    with pytest.raises(bench.BenchmarkError, match="inside data/processed"):
        bench.check_output_dir(ROOT, ROOT / "docs", git=fake_git())
    with pytest.raises(bench.BenchmarkError, match="inside data/processed"):
        bench.check_output_dir(ROOT, ROOT / "data" / "processed", git=fake_git())
    with pytest.raises(bench.BenchmarkError, match="tracked"):
        bench.check_output_dir(ROOT, good, git=fake_git(ls_files=b"data/processed/x.json\0"))
    with pytest.raises(bench.BenchmarkError, match="not git-ignored"):
        bench.check_output_dir(ROOT, good, git=fake_git(ignored=False))
    assert good == bench.OUTPUT_DIR


def test_the_real_output_directory_is_ignored_by_this_repository():
    assert bench.check_output_dir(ROOT, bench.OUTPUT_DIR) == bench.OUTPUT_DIR.resolve()


@pytest.mark.parametrize("name", ["ecommerce_search", "postgres", "ecommerce_search_bench_xyz"])
def test_development_and_non_scratch_databases_are_refused(name):
    with pytest.raises(bench.BenchmarkError):
        bench.assert_scratch_name(name, "ecommerce_search")
    lookalike = "ecommerce_search_bench_0123456789ab"
    with pytest.raises(bench.BenchmarkError, match="development database"):
        bench.assert_scratch_name(lookalike, development_db=lookalike)


class FakeConn:
    def __init__(self, log):
        self.log = log

    def execute(self, statement, params=None):
        self.log.append(str(statement))
        return self


class FakeEngine:
    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def connect(self):
        yield FakeConn(self.log)


@pytest.mark.parametrize("failure", [RuntimeError("child failed"), KeyboardInterrupt()])
def test_scratch_database_is_dropped_after_child_failure_or_interrupt(failure):
    admin = FakeEngine()
    name = bench.new_scratch_name()
    with pytest.raises(type(failure)), bench.dropped_after(admin, name, "ecommerce_search"):
        raise failure
    assert admin.log == [f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)']


def test_child_side_scratch_database_is_created_and_always_dropped():
    admin = FakeEngine()
    name = bench.new_scratch_name()
    with pytest.raises(KeyboardInterrupt), bench.scratch_database(admin, name, "ecommerce_search"):
        raise KeyboardInterrupt
    assert admin.log == [
        f'CREATE DATABASE "{name}"',
        f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)',
    ]


def test_cleanup_refuses_the_development_database_before_any_sql():
    admin = FakeEngine()
    with (
        pytest.raises(bench.BenchmarkError),
        bench.dropped_after(admin, "ecommerce_search", "ecommerce_search"),
    ):
        pass
    assert admin.log == []


# ---- collectors (fake clients only) ----------------------------------------------------------------


class FakeResponse:
    def __init__(self, body, status=200):
        self.status_code, self._body = status, body

    def json(self):
        return json.loads(json.dumps(self._body))


def _body(path, query, top_k, rrf_k=60):
    lexical, dense = source_lists(query)
    if path == "/search":
        rows = [
            {"product_id": p, "rank": i, "lexical_score": 0.5} for i, p in enumerate(lexical, 1)
        ]
        return {"results": rows[:top_k]}
    if path == "/search/dense":
        rows = [{"product_id": p, "rank": i, "dense_score": 0.4} for i, p in enumerate(dense, 1)]
        return {"results": rows[:top_k]}
    fused = bench.rrf_oracle(lexical, dense, rrf_k)
    scores = bench.rrf_scores(lexical, dense, rrf_k)
    top = fused[:top_k]
    return {
        "fusion": {
            "method": "rrf",
            "rrf_k": rrf_k,
            "rrf_k_status": "candidate_pending_selection",
            "lexical_k": 50,
            "dense_k": 50,
            "candidate_k": 50,
        },
        "results": [
            {
                "product_id": p,
                "rank": i,
                "rrf_score": float(scores[p]),
                "lexical_rank": lexical.index(p) + 1 if p in lexical else None,
                "dense_rank": dense.index(p) + 1 if p in dense else None,
            }
            for i, p in enumerate(top, 1)
        ],
        "lexical_hit_count": len(lexical),
        "dense_hit_count": len(dense),
        "overlap_count": len(set(lexical) & set(dense)),
        "fused_count": len(fused),
        "candidate_count": min(len(fused), 50),
        "result_count": len(top),
    }


class FakeClient:
    def __init__(self, rrf_k=60, fail=None):
        self.rrf_k, self.fail, self.calls = rrf_k, fail, []

    def get(self, path, params):
        self.calls.append((path, params["q"], params["top_k"]))
        if self.fail and self.fail == (path, params["q"]):
            return FakeResponse({}, status=503)
        return FakeResponse(_body(path, params["q"], params["top_k"], self.rrf_k))


class SpyFuse:
    """Delegates to production fuse_rrf and records exactly what each call consumed."""

    def __init__(self):
        self.calls = []

    def __call__(self, lexical, dense, rrf_k, candidate_k):
        self.calls.append(
            {
                "lexical": lexical,
                "dense": dense,
                "inputs": (
                    [(h.product_id, h.rank) for h in lexical],
                    [(h.product_id, h.rank) for h in dense],
                ),
                "rrf_k": rrf_k,
            }
        )
        return fuse_rrf(lexical, dense, rrf_k, candidate_k)


def test_sources_are_retrieved_exactly_once_per_query_per_run():
    header = make_header("S")
    per_run, fetches = {}, {}
    for run in (1, 2):
        fetches[run] = FakeFetch(run=run)
        per_run[run] = s_run_items(run, fetches[run])
        assert fetches[run].calls == list(bench.QUERIES)  # once each, in protocol order
    stream = _stream(header, per_run)
    assert len(stream) == 2 * (2 + 17 + 17 * 7)
    assert sum(1 for r in stream if r["record"] == "sources") == 2 * 17
    assert bench.derive(header, stream)["determinism"]["runs_identical"] is True


def test_every_grid_value_consumes_the_identical_source_lists():
    spy, emitted = SpyFuse(), []
    bench.collect_phase_s(FakeFetch(), bench.QUERIES, emitted.append, 50, fuse=spy)
    assert len(spy.calls) == 17 * 7
    for qi in range(17):
        calls = spy.calls[qi * 7 : (qi + 1) * 7]
        assert [c["rrf_k"] for c in calls] == list(bench.RRF_K_GRID)
        assert all(c["lexical"] is calls[0]["lexical"] for c in calls)  # the same objects
        assert all(c["dense"] is calls[0]["dense"] for c in calls)
        assert all(c["inputs"] == calls[0]["inputs"] for c in calls)
        assert isinstance(calls[0]["lexical"], tuple) and isinstance(calls[0]["dense"], tuple)
    for query in bench.QUERIES:
        source = [r for r in emitted if r["record"] == "sources" and r["query"] == query]
        fused = [r for r in emitted if r["record"] == "hybrid" and r["query"] == query]
        assert len(source) == 1 and len(fused) == 7
        assert {r["sources_sha256"] for r in fused} == {source[0]["sources_sha256"]}


def test_changing_rrf_k_changes_only_fusion_never_retrieval():
    fetch, emitted = FakeFetch(), []
    bench.collect_phase_s(fetch, bench.QUERIES, emitted.append, 50)
    assert fetch.calls == list(bench.QUERIES)  # retrieval receives the query only, never rrf_k
    laptop = [r for r in emitted if r["record"] == "hybrid" and r["query"] == "laptop"]
    assert len({tuple(r["ids"]) for r in laptop}) > 1  # fusion output does change with rrf_k
    assert len({r["sources_sha256"] for r in laptop}) == 1  # its inputs never do
    assert [r["fusion"]["rrf_k"] for r in laptop] == list(bench.RRF_K_GRID)
    for record in laptop:
        assert record["fusion"]["implementation"] == "ecommerce_search.search.hybrid.fuse_rrf"
        assert record["rrf_score_fractions"][0].count("/") == 1


def test_fusion_that_mutates_its_inputs_is_refused():
    def mutating(lexical, dense, rrf_k, candidate_k):
        result = fuse_rrf(lexical, dense, rrf_k, candidate_k)
        lexical[0].lexical_score = 0.123
        return result

    with pytest.raises(bench.BenchmarkError, match="altered the source lists"):
        bench.collect_phase_s(FakeFetch(), bench.QUERIES, [].append, 50, fuse=mutating)


def test_source_fetcher_uses_one_loaded_embedder_and_production_depths():
    class Embedder:
        def __init__(self):
            self.loads = 0

        def load(self):
            self.loads += 1
            return 12.5 if self.loads == 1 else None

    embedder, sessions, reads = Embedder(), [], []

    @contextlib.contextmanager
    def open_session():
        sessions.append("open")
        yield f"session{len(sessions)}"
        sessions.append("closed")

    def encode(emb, spec, query, location):
        assert emb is embedder and spec == "spec"
        return [0.1], emb.load(), 1.0

    def read(session, query, vector, spec, lexical_k, dense_k):
        reads.append((session, query, lexical_k, dense_k))
        return SimpleNamespace(hits=[]), SimpleNamespace(hits=[])

    assert embedder.load() == 12.5  # the child loads the provider once before Phase S
    fetch = bench.source_fetcher(embedder, "spec", open_session, 50, read=read, encode=encode)
    assert [fetch(q)[2] for q in ("laptop", "nike shoes")] == [None, None]  # never reloaded
    assert reads == [("session1", "laptop", 50, 50), ("session3", "nike shoes", 50, 50)]
    assert sessions == ["open", "closed", "open", "closed"]


def test_source_fetcher_defaults_to_the_production_functions():
    import ecommerce_search.api.dense as api_dense
    import ecommerce_search.search.hybrid as hybrid

    seen = {}

    def spy_read(*args):
        seen["read"] = True
        raise RuntimeError("stop")

    original = hybrid.read_sources
    try:
        hybrid.read_sources = spy_read
        fetch = bench.source_fetcher(
            None, None, contextlib.nullcontext, 50, encode=lambda *a: ([0.0], None, 0.0)
        )
        with pytest.raises(RuntimeError, match="stop"):
            fetch("laptop")
    finally:
        hybrid.read_sources = original
    assert seen == {"read": True}
    assert api_dense.encode_query.__module__ == "ecommerce_search.api.dense"


def test_a_model_load_during_retrieval_is_refused():
    header = make_header("S")
    per_run = {run: s_run_items(run, FakeFetch(run=run, load_ms=850.0)) for run in (1, 2)}
    with pytest.raises(bench.BenchmarkError, match=r"the model was \(re\)loaded"):
        bench.derive(header, _stream(header, per_run))


def test_phase_cl_collector_aborts_on_a_non_200_response():
    client = LatencyClient(fail=("/search/hybrid", "laptop"))
    schemas = dict.fromkeys(bench.ENDPOINTS, FakeSchema)
    with pytest.raises(bench.BenchmarkError, match="503"):
        bench.collect_phase_cl(client, schemas, ["laptop"], 3, [].append, 1, 1)


class FakeSchema:
    @staticmethod
    def model_validate(body):
        return SimpleNamespace(model_dump_json=lambda: json.dumps(body))


class LatencyClient(FakeClient):
    def get(self, path, params):
        response = super().get(path, params)
        if response.status_code != 200:
            return response
        body = response._body
        body["result_count"] = len(body["results"])
        body["latency_ms"] = {
            "total_ms": 1.0,
            "lexical_ms": 0.5,
            "model_load_ms": None,
            "query_embedding_ms": 0.2,
            "vector_ms": 0.3,
            "rrf_ms": 0.01,
        }
        return response


def test_phase_cl_collector_records_cold_and_timed_but_not_warmups():
    client, emitted = LatencyClient(), []
    schemas = dict.fromkeys(bench.ENDPOINTS, FakeSchema)
    bench.collect_phase_cl(client, schemas, bench.QUERIES[:2], 2, emitted.append, 3, 4)
    paths = [c[0] for c in client.calls]
    assert paths[0] == "/search/dense" and paths[-1] == "/search"
    assert len(client.calls) == 3 * 2 * (1 + 3 + 4)
    samples = [r for r in emitted if r["record"] == "sample"]
    assert len(samples) == 3 * 2 * (1 + 4)
    assert {r["phase"] for r in samples} == {"cold", "timed"}
    assert len([r for r in emitted if r["record"] == "results"]) == 6
    hybrid = next(r for r in samples if r["endpoint"] == "hybrid")
    assert set(hybrid["stages"]) == set(bench.ENDPOINTS["hybrid"]["stages"])
    assert hybrid["serialization_ms"] >= 0 and hybrid["request_ms"] >= 0


def test_phase_cl_collector_aborts_when_results_change():
    class Flaky(LatencyClient):
        def get(self, path, params):
            response = super().get(path, params)
            if len(self.calls) == 3:
                response._body["results"] = list(reversed(response._body["results"]))
            return response

    schemas = dict.fromkeys(bench.ENDPOINTS, FakeSchema)
    with pytest.raises(bench.BenchmarkError, match="results changed"):
        bench.collect_phase_cl(Flaky(), schemas, ["laptop"], 1, [].append, 2, 2)


# ---- side-effect-free import, --help and --print-protocol -----------------------------------------


def _run(code_or_args, *, script=False):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("POSTGRES_", "DB_", "EMBEDDING_", "SEARCH_"))
    }
    argv = (
        [sys.executable, "scripts/benchmark_hybrid.py", *code_or_args]
        if script
        else [sys.executable, "-c", code_or_args]
    )
    return subprocess.run(  # noqa: S603 - fixed argv, sys.executable
        argv, cwd=ROOT, capture_output=True, text=True, check=False, env=env, timeout=120
    )


def test_importing_the_harness_loads_no_model_library_and_opens_no_socket():
    code = (
        "import socket\n"
        "def guard(*a, **k): raise SystemExit('socket used during import')\n"
        "socket.socket.connect = guard; socket.create_connection = guard\n"
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('b', 'scripts/benchmark_hybrid.py')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(sorted({'torch', 'sentence_transformers', 'psutil', 'psycopg'} & set(sys.modules)))"
    )
    done = _run(code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_help_is_side_effect_free():
    done = _run(["--help"], script=True)
    assert done.returncode == 0, done.stderr
    for flag in ("--phase", "--verify", "--print-protocol"):
        assert flag in done.stdout
    assert "--child" not in done.stdout and "run 1" not in done.stdout


def test_print_protocol_shows_the_frozen_table_and_hashes():
    done = _run(["--print-protocol"], script=True)
    assert done.returncode == 0, done.stderr
    assert QUERY_SET_SHA256 in done.stdout and EXPECTATION_TABLE_SHA256 in done.stdout
    assert "anc=True (term 'anc')" in done.stdout
    assert "anc=True (term 'noise cancelling')" in done.stdout
    assert done.stdout.count("diagnostic_only") == 2


def test_cli_requires_exactly_one_mode():
    with pytest.raises(SystemExit):
        bench.main([])
    with pytest.raises(SystemExit):
        bench.main(["--phase", "S", "--print-protocol"])
    with pytest.raises(SystemExit):
        bench.main(["--phase", "L"])
