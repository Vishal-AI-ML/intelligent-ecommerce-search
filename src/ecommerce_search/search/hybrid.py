"""V1 hybrid retrieval (Milestone 5): lexical + dense candidates fused with Reciprocal Rank Fusion.

`rrf_score(d) = sum over sources s containing d of 1 / (rrf_k + rank_s(d))`, where `rank_s` is
the existing 1-based positional rank of `lexical_search` / `dense_search`. Both sources have
equal weight. Scores are summed as exact fractions, so ordering never depends on float rounding;
ties are broken by `product_id` ascending. The score is a rank-fusion value, not a relevance
probability.

Both source reads run in one PostgreSQL REPEATABLE READ READ ONLY transaction (`read_sources`),
so they see the same snapshot. The transaction is transaction-local (`SET TRANSACTION` as its
first statement): no engine or connection setting is changed. The query must already be encoded
before it is called, so no transaction is open while the model loads or encodes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from ecommerce_search.embeddings.spec import EmbeddingModelSpec
from ecommerce_search.search.dense import DenseHit, DenseResult, dense_search
from ecommerce_search.search.lexical import LexicalHit, LexicalResult, lexical_search

HYBRID_SEARCH_VERSION = "v1_hybrid"
FUSION_METHOD = "rrf"
# The configured `rrf_k` is a candidate until the M5 provisional selection is reviewed.
RRF_K_STATUS: Literal["candidate_pending_selection", "provisional"] = "candidate_pending_selection"

SNAPSHOT_SQL = text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")


@dataclass(frozen=True)
class FusedCandidate:
    product_id: str
    rrf_score: Fraction
    lexical_rank: int | None
    lexical_score: float | None
    dense_rank: int | None
    dense_score: float | None


def _check_source(name: str, hits: Sequence[LexicalHit] | Sequence[DenseHit]) -> None:
    ranks = [hit.rank for hit in hits]
    if ranks != list(range(1, len(hits) + 1)):
        raise ValueError(f"{name} ranks must be contiguous 1..n in list order")
    if len({hit.product_id for hit in hits}) != len(hits):
        raise ValueError(f"{name} product ids must be unique")


def fuse_rrf(
    lexical_hits: Sequence[LexicalHit],
    dense_hits: Sequence[DenseHit],
    rrf_k: int,
    candidate_k: int,
) -> list[FusedCandidate]:
    """Fuse two ranked lists into at most `candidate_k` unique candidates (pure, no I/O)."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    _check_source("lexical", lexical_hits)
    _check_source("dense", dense_hits)
    lexical = {hit.product_id: hit for hit in lexical_hits}
    dense = {hit.product_id: hit for hit in dense_hits}
    fused = []
    for product_id in lexical.keys() | dense.keys():
        lex = lexical.get(product_id)
        den = dense.get(product_id)
        score = Fraction(0)
        if lex is not None:
            score += Fraction(1, rrf_k + lex.rank)
        if den is not None:
            score += Fraction(1, rrf_k + den.rank)
        fused.append(
            FusedCandidate(
                product_id=product_id,
                rrf_score=score,
                lexical_rank=None if lex is None else lex.rank,
                lexical_score=None if lex is None else lex.lexical_score,
                dense_rank=None if den is None else den.rank,
                dense_score=None if den is None else den.dense_score,
            )
        )
    fused.sort(key=lambda candidate: (-candidate.rrf_score, candidate.product_id))
    return fused[:candidate_k]


def read_sources(
    session: Session,
    query: str,
    query_vector: Sequence[float],
    spec: EmbeddingModelSpec,
    lexical_k: int,
    dense_k: int,
) -> tuple[LexicalResult, DenseResult]:
    """Run the lexical and dense reads in one REPEATABLE READ READ ONLY transaction.

    The transaction commits (read-only, nothing to write) on success and rolls back on any
    error; either way the session releases its connection before this returns. Database errors
    propagate."""
    if session.in_transaction():
        raise RuntimeError("read_sources needs a session without an open transaction")
    with session.begin():
        session.execute(SNAPSHOT_SQL)  # must be the first statement of the transaction
        lexical = lexical_search(session, query, lexical_k)
        dense = dense_search(session, query_vector, spec, dense_k)
    return lexical, dense
