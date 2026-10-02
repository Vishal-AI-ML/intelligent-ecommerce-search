"""Pure Reciprocal Rank Fusion (Milestone 5): formula, exact ties, validation. No DB, no model."""

import random
from decimal import Decimal
from fractions import Fraction

import pytest

from ecommerce_search.search.dense import DenseHit
from ecommerce_search.search.hybrid import RRF_K_STATUS, FusedCandidate, fuse_rrf
from ecommerce_search.search.lexical import LexicalHit

FIELDS = {
    "title": "Test Product",
    "brand": "Test",
    "category": "laptop",
    "subcategory": None,
    "description": None,
    "price": Decimal("1.00"),
    "currency": "INR",
    "rating": None,
    "review_count": None,
    "availability": "in_stock",
}


@pytest.fixture(autouse=True)
def _strict_no_network(no_socket_connect):
    """Fusion is pure: no socket may connect at all."""


def lex(*ids: str) -> list[LexicalHit]:
    return [
        LexicalHit(rank=r, lexical_score=1.0 / r, product_id=p, **FIELDS)
        for r, p in enumerate(ids, start=1)
    ]


def den(*ids: str) -> list[DenseHit]:
    return [
        DenseHit(rank=r, dense_score=1.0 - r / 100, product_id=p, **FIELDS)
        for r, p in enumerate(ids, start=1)
    ]


def oracle(lexical: list[str], dense: list[str], rrf_k: int) -> list[tuple[str, Fraction]]:
    """Independent restatement of ADR-002: exact sum, then product_id ascending."""
    scores: dict[str, Fraction] = {}
    for ranking in (lexical, dense):
        for position, product_id in enumerate(ranking, start=1):
            scores[product_id] = scores.get(product_id, Fraction(0)) + Fraction(1, rrf_k + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def test_formula_on_a_hand_computed_example():
    fused = fuse_rrf(lex("A", "B", "C"), den("C", "D"), rrf_k=60, candidate_k=10)
    assert [(c.product_id, c.rrf_score) for c in fused] == [
        ("C", Fraction(1, 63) + Fraction(1, 61)),
        ("A", Fraction(1, 61)),
        ("B", Fraction(1, 62)),
        ("D", Fraction(1, 62)),  # exact tie with B: product_id ascending
    ]
    c = fused[0]
    assert (c.lexical_rank, c.dense_rank) == (3, 1)
    assert (c.lexical_score, c.dense_score) == (1.0 / 3, 0.99)
    assert fused[1] == FusedCandidate("A", Fraction(1, 61), 1, 1.0, None, None)
    assert fused[3] == FusedCandidate("D", Fraction(1, 62), None, None, 2, 0.98)


@pytest.mark.parametrize("rrf_k", [1, 5, 10, 20, 40, 60, 100, 1000])
@pytest.mark.parametrize("seed", range(5))
def test_matches_an_independent_oracle(rrf_k, seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic test data, not security
    pool = [f"P{n:03d}" for n in range(80)]
    lexical = rng.sample(pool, rng.randint(0, 50))
    dense = rng.sample(pool, rng.randint(0, 50))
    fused = fuse_rrf(lex(*lexical), den(*dense), rrf_k=rrf_k, candidate_k=100)
    assert [(c.product_id, c.rrf_score) for c in fused] == oracle(lexical, dense, rrf_k)


def test_exact_tie_breaks_on_product_id_even_where_floats_disagree():
    # 1/(60+12) + 1/(60+28) == 1/(60+6) + 1/(60+39) == 5/198 exactly, but the float sums differ
    # in the last bit. Float ordering would put B first; the exact rule puts A first.
    assert 1 / 66 + 1 / 99 > 1 / 72 + 1 / 88
    lexical = [f"L{n:02d}" for n in range(1, 40)]
    dense = [f"D{n:02d}" for n in range(1, 40)]
    lexical[12 - 1], dense[28 - 1] = "A", "A"
    lexical[6 - 1], dense[39 - 1] = "B", "B"
    fused = fuse_rrf(lex(*lexical), den(*dense), rrf_k=60, candidate_k=100)
    a, b = fused[0], fused[1]
    assert (a.product_id, b.product_id) == ("A", "B")
    assert a.rrf_score == b.rrf_score == Fraction(5, 198)


def test_tie_order_does_not_depend_on_input_order():
    first = fuse_rrf(lex("Z", "Y"), den("Y", "Z"), rrf_k=60, candidate_k=10)
    assert [c.product_id for c in first] == ["Y", "Z"]  # equal scores
    second = fuse_rrf(lex("Y", "Z"), den("Z", "Y"), rrf_k=60, candidate_k=10)
    assert [(c.product_id, c.rrf_score) for c in first] == [
        (c.product_id, c.rrf_score) for c in second
    ]


def test_one_result_per_product_with_both_ranks_preserved():
    fused = fuse_rrf(lex("A", "B"), den("B", "A"), rrf_k=60, candidate_k=10)
    assert [c.product_id for c in fused] == ["A", "B"]
    assert {(c.product_id, c.lexical_rank, c.dense_rank) for c in fused} == {
        ("A", 1, 2),
        ("B", 2, 1),
    }


def test_lexical_only_and_dense_only_inputs():
    only_lexical = fuse_rrf(lex("A", "B"), [], rrf_k=60, candidate_k=10)
    assert [(c.product_id, c.dense_rank, c.dense_score) for c in only_lexical] == [
        ("A", None, None),
        ("B", None, None),
    ]
    only_dense = fuse_rrf([], den("C", "A"), rrf_k=60, candidate_k=10)
    assert [(c.product_id, c.lexical_rank, c.lexical_score) for c in only_dense] == [
        ("C", None, None),
        ("A", None, None),
    ]
    assert [c.rrf_score for c in only_dense] == [Fraction(1, 61), Fraction(1, 62)]


def test_overlap_outranks_single_source_hits():
    fused = fuse_rrf(lex("A", "B", "C"), den("X", "Y", "C"), rrf_k=60, candidate_k=10)
    assert fused[0].product_id == "C"


def test_empty_inputs():
    assert fuse_rrf([], [], rrf_k=60, candidate_k=10) == []


def test_candidate_truncation_and_prefix_property():
    lexical = [f"P{n:02d}" for n in range(30)]
    dense = list(reversed(lexical)) + [f"Q{n:02d}" for n in range(20)]
    full = fuse_rrf(lex(*lexical), den(*dense), rrf_k=60, candidate_k=100)
    assert len(full) == 50
    for candidate_k in (1, 7, 25, 50):
        assert (
            fuse_rrf(lex(*lexical), den(*dense), rrf_k=60, candidate_k=candidate_k)
            == (full[:candidate_k])
        )
    scores = [c.rrf_score for c in full]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("lexical", "dense"),
    [
        (lex("A", "A"), []),  # duplicate product
        ([], den("A", "B", "A")),
        ([LexicalHit(rank=2, lexical_score=1.0, product_id="A", **FIELDS)], []),  # starts at 2
        (lex("A", "B")[::-1], []),  # ranks out of list order
        ([], [DenseHit(rank=1, dense_score=1.0, product_id=p, **FIELDS) for p in "AB"]),
    ],
)
def test_invalid_source_lists_are_rejected(lexical, dense):
    with pytest.raises(ValueError, match="ranks must be contiguous|ids must be unique"):
        fuse_rrf(lexical, dense, rrf_k=60, candidate_k=10)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "60"])
def test_invalid_parameters_are_rejected(value):
    with pytest.raises(ValueError, match="rrf_k"):
        fuse_rrf(lex("A"), [], rrf_k=value, candidate_k=10)
    with pytest.raises(ValueError, match="candidate_k"):
        fuse_rrf(lex("A"), [], rrf_k=60, candidate_k=value)


def test_rrf_k_status_marks_the_value_as_pending_selection():
    assert RRF_K_STATUS == "candidate_pending_selection"
