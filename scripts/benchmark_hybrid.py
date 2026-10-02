"""Milestone 5 hybrid harness: provisional rrf_k selection (S), comparison (C) and latency (L).

Scratch databases only (`ecommerce_search_bench_<12 hex>`); the development database is refused
by name and never connected to for data. Offline: the pinned MiniLM snapshot must verify against
its manifest, and outbound non-loopback sockets are refused in the parent and every run process.

Frozen protocol: the 17 queries and their deterministic attribute expectations are committed in
this file (`QUERY_PROTOCOL`), validated against the approved expectation rules and pinned by
SHA-256 (`QUERY_SET_SHA256`, `EXPECTATION_TABLE_SHA256`). Nothing is inferred at run time; a
changed table changes the hashes and the harness refuses to run or verify. The table must not be
edited after Phase S results have been seen.

The attribute-consistency score is a provisional deterministic proxy: the share of the top 10
results whose catalog facts (category, brand, anc) equal every explicit expectation of the
query. It is NOT relevance, NOT quality ground truth and NOT a human label. Queries without an
approved explicit expectation are diagnostic only and excluded from every aggregate.

Phase S (2 runs, fresh subprocess + scratch DB each): per run the scratch DB is set up and the
  embeddings generated once, with one provider loaded once; that same provider encodes every
  query (the API's `encode_query`). Each query's lexical and dense source lists are retrieved
  exactly once (production `read_sources`, depth 50, one REPEATABLE READ READ ONLY snapshot) and
  those same immutable lists are fused with production `fuse_rrf` for every rrf_k in the fixed
  grid; the top 10 are kept. rrf_k never reaches retrieval, the configured value is never
  changed, and an independent exact-fraction oracle re-checks every fused id, rank and score.
  Highest mean consistency over checked queries wins; an exact tie goes to the largest tied
  rrf_k. Both runs must be identical, otherwise no value is proposed. The
  outcome is a candidate for a separate review: settings are never changed by this harness.
Phase CL (3 runs, fresh subprocess + scratch DB each, endpoint order rotated per run): in-process
  `GET /search`, `/search/dense`, `/search/hybrid` at top_k=10 with the configured settings. Per
  endpoint and query: 1 cold sample, 20 untimed warm-ups, 200 timed samples. Every stage field,
  `request_ms` and a separately timed `serialization_ms` are recorded; raw samples are never
  dropped. Phase C (counts, zero results, overlaps, hybrid source composition, consistency proxy,
  determinism) is derived from the cold result lists of the same runs. Results are provisional
  and non-authoritative; no latency target is claimed.

Everything in a summary is derived from the raw JSONL records; `--verify` recomputes it and
refuses missing, duplicated, reordered, incomplete or tampered records.
Artifacts (git-ignored): data/processed/hybrid_benchmark/<experiment_id>.{json,md,samples.jsonl}.

Usage (any execution requires a separate approval):
  HF_HUB_OFFLINE=1 uv run --offline python scripts/benchmark_hybrid.py --phase S
  HF_HUB_OFFLINE=1 uv run --offline python scripts/benchmark_hybrid.py --phase CL
  uv run --offline python scripts/benchmark_hybrid.py --verify <artifact .json>
  uv run --offline python scripts/benchmark_hybrid.py --print-protocol
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_dense import (  # noqa: E402 - scripts/ is not a package
    BenchmarkError,
    GitError,
    assert_scratch_name,
    check_model,
    committed_catalog_identity,
    embedding_corpus_digest,
    ids_sha256,
    install_network_guard,
    new_scratch_name,
    power_events,
    scratch_database,
)
from benchmark_dense import environment as server_environment  # noqa: E402
from benchmark_lexical import _git, dumps, read_samples, source_fingerprint, summarize  # noqa: E402
from embedding_model_selection import QUERIES as M4_QUERIES  # noqa: E402

from ecommerce_search.search.query import normalize_query  # noqa: E402

# ---- pinned provenance ---------------------------------------------------------------------------

# Committed M5 core implementation; Phase S must run with identical runtime-affecting paths.
CORE_COMMIT = "f7cb2757e669cb58b281c9ac3d8a37b6da7e1a7a"
# Phases C/L run at the final runtime commit (after the reviewed rrf_k change). It is pinned in a
# separately reviewed harness change; until then Phase CL refuses to run.
RUNTIME_COMMIT: str | None = None
PHASE_COMMITS: dict[str, str | None] = {"S": CORE_COMMIT, "CL": RUNTIME_COMMIT}
GUARDED_PATHS: tuple[str, ...] = (
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
OUTPUT_DIR = ROOT / "data" / "processed" / "hybrid_benchmark"
OUTPUT_PARENT = ROOT / "data" / "processed"
SEED_PRODUCTS = 240
CATALOG_KEYS: tuple[str, ...] = (
    "dataset_id",
    "dataset_version",
    "checksum_sha256",
    "record_count",
    "transform_version",
    "taxonomy_version",
    "rules_version",
)
QUERY_MAX_LENGTH = 200  # the API's default SEARCH_MAX_QUERY_LENGTH

# ---- frozen query protocol (single source of truth; immutable after Phase S results) -------------

M4_QUERY_COUNT = 12
EXPECTED_QUERY_COUNT = 17
# The only approved deterministic expectation rules: an explicit query term maps to one catalog
# fact. A one-word term is explicit only as a whole token of the case-folded query; the one
# approved phrase ("noise cancelling") only as that exact phrase with word boundaries (so
# "noise-cancelling" or "noise cancellation" are not explicit). No synonyms, no other types.
APPROVED_PHRASES: frozenset[str] = frozenset({"noise cancelling"})
APPROVED_TERMS: dict[str, tuple[str, object]] = {
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
FACT_FIELDS: tuple[str, ...] = ("category", "brand", "anc")
CHECKED, DIAGNOSTIC = "checked", "diagnostic_only"


def _cat(term: str) -> dict:
    return {"term": term, "field": "category", "value": term}


def _brand(term: str, value: str) -> dict:
    return {"term": term, "field": "brand", "value": value}


def _anc(term: str) -> dict:
    return {"term": term, "field": "anc", "value": True}


QUERY_PROTOCOL: tuple[dict, ...] = (
    {"query": "laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "hp laptop", "status": CHECKED, "expectations": [_brand("hp", "HP"), _cat("laptop")]},
    {"query": "8gb laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "iphone", "status": DIAGNOSTIC, "expectations": []},
    {"query": "zzqxv", "status": DIAGNOSTIC, "expectations": []},
    {"query": "coding ke liye laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "coding laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "student laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "lightweight laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {"query": "premium phone", "status": CHECKED, "expectations": [_cat("phone")]},
    {"query": "running shoes", "status": CHECKED, "expectations": [_cat("shoes")]},
    {
        "query": "noise cancelling headphones",
        "status": CHECKED,
        "expectations": [_anc("noise cancelling"), _cat("headphones")],
    },
    {"query": "256gb ssd laptop", "status": CHECKED, "expectations": [_cat("laptop")]},
    {
        "query": "apple phone",
        "status": CHECKED,
        "expectations": [_brand("apple", "Apple"), _cat("phone")],
    },
    {
        "query": "nike shoes",
        "status": CHECKED,
        "expectations": [_brand("nike", "Nike"), _cat("shoes")],
    },
    {"query": "wireless headphones", "status": CHECKED, "expectations": [_cat("headphones")]},
    {
        "query": "anc headphones",
        "status": CHECKED,
        "expectations": [_anc("anc"), _cat("headphones")],
    },
)
QUERIES: tuple[str, ...] = tuple(entry["query"] for entry in QUERY_PROTOCOL)
QUERY_SET_SHA256 = "ea1df08db89a8bceb0e5b6437f18e75a30aed997fb70d727d080b441ebbd78d3"
EXPECTATION_TABLE_SHA256 = "ac5647876b33397bc0b0cc2d60e60357af9285daf65f5973e5c675f824c8d643"

# ---- predeclared experiment parameters -----------------------------------------------------------

TOP_K = 10
SOURCE_DEPTH = 50  # search_lexical_k / search_dense_k: the hybrid source depths (I2)
RRF_K_GRID: tuple[int, ...] = (1, 5, 10, 20, 40, 60, 100)
TIE_RULE = "exact tie on mean consistency -> the largest tied rrf_k"
ENDPOINTS: dict[str, dict] = {
    "search": {"path": "/search", "stages": ("lexical_ms",)},
    "dense": {
        "path": "/search/dense",
        "stages": ("model_load_ms", "query_embedding_ms", "vector_ms"),
    },
    "hybrid": {
        "path": "/search/hybrid",
        "stages": ("lexical_ms", "model_load_ms", "query_embedding_ms", "vector_ms", "rrf_ms"),
    },
}
ENDPOINT_SCHEDULE: dict[int, tuple[str, str, str]] = {
    1: ("search", "dense", "hybrid"),
    2: ("dense", "hybrid", "search"),
    3: ("hybrid", "search", "dense"),
}
# model_load_ms is null except on the request that loads the model, so it is reported from the
# cold samples only and is never part of the timed percentiles.
NULLABLE_STAGES = frozenset({"model_load_ms"})
PREDECLARED: dict[str, dict] = {
    "S": {
        "phase": "S",
        "runs": 2,
        "rrf_k_grid": list(RRF_K_GRID),
        "top_k": TOP_K,
        "source_depth": SOURCE_DEPTH,
        "source_retrieval": "once per query per run (encode_query + read_sources); the same "
        "lists are reused for every rrf_k",
        "fusion": "ecommerce_search.search.hybrid.fuse_rrf, called directly per rrf_k",
        "tie_rule": TIE_RULE,
        "score_denominator": TOP_K,
    },
    "CL": {
        "phase": "CL",
        "runs": 3,
        "top_k": TOP_K,
        "cold_per_query": 1,
        "warmup_per_query": 20,
        "samples_per_query": 200,
        "endpoint_schedule": {str(r): list(o) for r, o in ENDPOINT_SCHEDULE.items()},
        "score_denominator": TOP_K,
        "percentile_method": "nearest-rank",
        "primary": "per-run percentiles; no pooled or universal latency target",
    },
}
PROXY_LABEL = (
    "provisional deterministic attribute-consistency proxy: share of the top 10 whose catalog "
    "facts equal every explicit query expectation (missing results count as not satisfying). "
    "Not relevance, not quality ground truth, not human labels"
)
NOT_APPLICABLE: dict[str, str] = {
    "golden_dataset_version": "not applicable: the human-reviewed Golden Dataset is created in "
    "Milestone 9; none exists in Milestone 5",
    "relevance_quality_metrics": "not applicable: no relevance or quality metric is computed "
    "before the Golden Dataset (Milestones 9-10); only a deterministic consistency proxy, "
    "overlaps, determinism and latency are reported",
    "human_labels": "not applicable: no human-reviewed labels exist or are created here",
    "decision_provider": "not applicable: no decision provider exists in Milestone 5",
    "reranker": "not applicable: no cross-encoder reranker exists before Milestone 8",
    "cost": "not applicable: local CPU model and local database only; no paid or metered service",
}
METRIC_DEFINITIONS: dict[str, str] = {
    "request_ms": "in-process TestClient GET, ends before response.json(); not network latency",
    "app_total_ms": "handler entry through result-model construction (API latency_ms.total_ms)",
    "serialization_ms": "the parsed response model's model_dump_json(), timed on its own after "
    "the request_ms interval has ended (never part of request_ms)",
    "lexical_ms": "lexical SQL and row fetch (API latency_ms.lexical_ms)",
    "model_load_ms": "model load on the first request that needs it (cold samples only)",
    "query_embedding_ms": "query token check and encoding (API latency_ms.query_embedding_ms)",
    "vector_ms": "vector SQL and row fetch (API latency_ms.vector_ms)",
    "rrf_ms": "Reciprocal Rank Fusion of the two source lists (API latency_ms.rrf_ms)",
}
RECORD_TYPES: dict[str, frozenset[str]] = {
    "S": frozenset({"run", "facts", "sources", "hybrid"}),
    "CL": frozenset({"run", "facts", "results", "sample"}),
}


class ProvenanceError(BenchmarkError):
    """The tree, pinned commit, protocol hashes or model snapshot are not as required."""


# ---- protocol identity (pure; unit-tested) -------------------------------------------------------


def query_tokens(query: str) -> list[str]:
    return re.findall(r"[0-9a-z]+", query.casefold())


def explicit_terms(query: str) -> set[str]:
    """Approved terms literally present: whole tokens, or an approved phrase verbatim."""
    tokens = set(query_tokens(query))
    folded = query.casefold()
    found = {t for t in APPROVED_TERMS if t not in APPROVED_PHRASES and t in tokens}
    for phrase in APPROVED_PHRASES:
        if re.search(rf"(?<![0-9a-z-]){re.escape(phrase)}(?![0-9a-z-])", folded):
            found.add(phrase)
    return found


def validate_protocol(protocol: Sequence[dict]) -> None:
    """Refuse any table that is not exactly the approved explicit-term expectations."""
    queries = [entry["query"] for entry in protocol]
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise BenchmarkError(f"expected {EXPECTED_QUERY_COUNT} queries, got {len(queries)}")
    if tuple(queries[:M4_QUERY_COUNT]) != tuple(M4_QUERIES):
        raise BenchmarkError("the first 12 queries must be the M4 queries in their committed order")
    if len(set(queries)) != len(queries):
        raise BenchmarkError("the query set contains duplicates")
    folded = [normalize_query(q, QUERY_MAX_LENGTH).casefold() for q in queries]
    if len(set(folded)) != len(folded):
        raise BenchmarkError("two queries normalize to the same text")
    for entry, query in zip(protocol, queries, strict=True):
        if set(entry) != {"query", "status", "expectations"}:
            raise BenchmarkError(f"{query!r}: unexpected keys in the protocol entry")
        if normalize_query(query, QUERY_MAX_LENGTH) != query:
            raise BenchmarkError(f"{query!r}: queries must be stored in normalized form")
        explicit = explicit_terms(query)
        seen = set()
        for expectation in entry["expectations"]:
            if set(expectation) != {"term", "field", "value"}:
                raise BenchmarkError(f"{query!r}: unexpected keys in an expectation")
            term = expectation["term"]
            if term not in APPROVED_TERMS:
                raise BenchmarkError(f"{query!r}: {term!r} is not an approved expectation term")
            field, value = APPROVED_TERMS[term]
            if (expectation["field"], expectation["value"]) != (field, value) or type(
                expectation["value"]
            ) is not type(value):
                raise BenchmarkError(f"{query!r}: expectation for {term!r} differs from the rule")
            if term not in explicit:
                raise BenchmarkError(f"{query!r}: {term!r} is not an explicit term of the query")
            if field in seen:
                raise BenchmarkError(f"{query!r}: two expectations for the field {field!r}")
            seen.add(field)
        required = {APPROVED_TERMS[t] for t in explicit}
        present = {(e["field"], e["value"]) for e in entry["expectations"]}
        if present != required:
            raise BenchmarkError(f"{query!r}: expectations must cover exactly its explicit terms")
        expected_status = CHECKED if entry["expectations"] else DIAGNOSTIC
        if entry["status"] != expected_status:
            raise BenchmarkError(f"{query!r}: status must be {expected_status!r}")


def protocol_identity(protocol: Sequence[dict] = QUERY_PROTOCOL) -> dict:
    validate_protocol(protocol)
    table = [
        {
            "query": e["query"],
            "status": e["status"],
            "expectations": [dict(x) for x in e["expectations"]],
        }
        for e in protocol
    ]
    queries = [e["query"] for e in protocol]
    return {
        "count": len(queries),
        "source": "scripts/benchmark_hybrid.py QUERY_PROTOCOL",
        "queries": queries,
        "query_set_sha256": hashlib.sha256(dumps(queries).encode("utf-8")).hexdigest(),
        "expectation_table": table,
        "expectation_table_sha256": hashlib.sha256(dumps(table).encode("utf-8")).hexdigest(),
        "checked_queries": [e["query"] for e in protocol if e["status"] == CHECKED],
        "diagnostic_only_queries": [e["query"] for e in protocol if e["status"] == DIAGNOSTIC],
        "approved_terms": {t: [f, v] for t, (f, v) in sorted(APPROVED_TERMS.items())},
        "proxy_label": PROXY_LABEL,
    }


def check_protocol_identity(
    protocol: Sequence[dict] = QUERY_PROTOCOL,
    query_sha: str = QUERY_SET_SHA256,
    table_sha: str = EXPECTATION_TABLE_SHA256,
) -> dict:
    identity = protocol_identity(protocol)
    if identity["query_set_sha256"] != query_sha:
        raise ProvenanceError("the query set differs from the pinned QUERY_SET_SHA256")
    if identity["expectation_table_sha256"] != table_sha:
        raise ProvenanceError("the expectation table differs from the pinned SHA-256")
    return identity


# ---- scoring, fusion oracle, overlaps and selection (pure; unit-tested) --------------------------


def satisfies(facts: dict | None, expectations: Sequence[dict]) -> bool:
    if facts is None:
        return False
    return all(
        type(facts.get(e["field"])) is type(e["value"]) and facts.get(e["field"]) == e["value"]
        for e in expectations
    )


def consistency(
    ids: Sequence[str], facts: dict[str, dict], expectations: Sequence[dict]
) -> dict | None:
    """Top-10 share satisfying every expectation; None for a diagnostic-only query."""
    if not expectations:
        return None
    hits = sum(1 for pid in ids[:TOP_K] if satisfies(facts.get(pid), expectations))
    score = Fraction(hits, TOP_K)
    return {"satisfied": hits, "denominator": TOP_K, "score": str(score)}


def mean_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise BenchmarkError("no checked queries to aggregate")
    return sum(values, Fraction(0)) / len(values)


def frac(value: Fraction) -> dict:
    return {"fraction": str(value), "float": round(float(value), 6)}


def rrf_scores(
    lexical_ids: Sequence[str], dense_ids: Sequence[str], rrf_k: int
) -> dict[str, Fraction]:
    """Independent exact RRF scores over positional ranks (sum of 1 / (rrf_k + rank))."""
    scores: dict[str, Fraction] = {}
    for ids in (lexical_ids, dense_ids):
        for rank, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, Fraction(0)) + Fraction(1, rrf_k + rank)
    return scores


def rrf_oracle(lexical_ids: Sequence[str], dense_ids: Sequence[str], rrf_k: int) -> list[str]:
    """Independent exact RRF order; ties by product_id ascending."""
    scores = rrf_scores(lexical_ids, dense_ids, rrf_k)
    return sorted(scores, key=lambda pid: (-scores[pid], pid))


def check_ranked(ids: Sequence[str], ranks: Sequence[int], what: str) -> None:
    if len(set(ids)) != len(ids) or list(ranks) != list(range(1, len(ids) + 1)):
        raise BenchmarkError(f"{what}: ids must be unique with ranks 1..n")


def composition(lexical_ranks: Sequence[int | None], dense_ranks: Sequence[int | None]) -> dict:
    pairs = list(zip(lexical_ranks, dense_ranks, strict=True))
    if any(lex is None and den is None for lex, den in pairs):
        raise BenchmarkError("a hybrid result has neither a lexical nor a dense rank")
    return {
        "both": sum(1 for lex, den in pairs if lex is not None and den is not None),
        "lexical_only": sum(1 for lex, den in pairs if lex is not None and den is None),
        "dense_only": sum(1 for lex, den in pairs if lex is None and den is not None),
    }


def overlap(a: Sequence[str], b: Sequence[str]) -> dict:
    return {
        "unordered": len(set(a) & set(b)),
        "same_position": sum(1 for x, y in zip(a, b, strict=False) if x == y),
        "identical_order": list(a) == list(b),
    }


def has_score_tie(scores: Sequence[float | None]) -> bool:
    present = [s for s in scores if s is not None]
    return len(set(present)) != len(present)


def decide_rrf_k(aggregates: dict[int, Fraction], deterministic: bool) -> dict:
    """Predeclared rule: highest mean consistency; exact tie -> largest tied rrf_k."""
    if sorted(aggregates) != sorted(RRF_K_GRID):
        raise BenchmarkError("aggregates must cover exactly the predeclared rrf_k grid")
    table = {str(k): frac(aggregates[k]) for k in RRF_K_GRID}
    base = {
        "grid": list(RRF_K_GRID),
        "mean_consistency": table,
        "tie_rule": TIE_RULE,
        "label": "candidate for separate review; settings are never changed by this harness",
    }
    if not deterministic:
        return {
            **base,
            "status": "no_selection_nondeterministic",
            "selected_rrf_k": None,
            "tied_rrf_k": [],
        }
    best = max(aggregates.values())
    tied = sorted(k for k in RRF_K_GRID if aggregates[k] == best)
    return {
        **base,
        "status": "candidate_proposed_pending_review",
        "selected_rrf_k": max(tied),
        "best_mean_consistency": frac(best),
        "tied_rrf_k": tied,
    }


def endpoint_order(run: int) -> tuple[str, str, str]:
    if run not in ENDPOINT_SCHEDULE:
        raise BenchmarkError(f"no predeclared endpoint order for run {run}")
    return ENDPOINT_SCHEDULE[run]


def sources_sha256(lexical: dict, dense: dict) -> str:
    """Fingerprint of the exact retrieval inputs (ids, ranks, scores) fusion consumes."""
    keep = ("ids", "ranks", "scores")
    body = {"lexical": {k: lexical[k] for k in keep}, "dense": {k: dense[k] for k in keep}}
    return hashlib.sha256(dumps(body).encode("utf-8")).hexdigest()


def facts_sha256(facts: dict[str, dict]) -> str:
    return hashlib.sha256(dumps(facts).encode("utf-8")).hexdigest()


# ---- provenance and safety (pure with injected git; unit-tested) ---------------------------------


def check_provenance(root: Path, phase: str, git: Callable[..., bytes] = _git) -> dict:
    """Refuse a dirty tree, an unpinned phase or any guarded-path difference from its pin."""
    if phase not in PHASE_COMMITS:
        raise BenchmarkError(f"unknown phase {phase!r}")
    commit = PHASE_COMMITS[phase]
    if commit is None:
        raise ProvenanceError(
            f"phase {phase} has no pinned runtime commit yet; pin it in a reviewed harness change"
        )
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise ProvenanceError("the working tree is not clean; commit or remove changes first")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    try:
        git(root, "merge-base", "--is-ancestor", commit, head)
    except GitError:
        raise ProvenanceError(f"the pinned commit {commit} is not an ancestor of HEAD") from None
    changed = git(root, "diff", "--name-only", commit, head, "--", *GUARDED_PATHS)
    differing = [line for line in changed.decode().splitlines() if line.strip()]
    if differing:
        raise ProvenanceError(
            "runtime-affecting paths differ from the pinned commit: " + ", ".join(differing[:10])
        )
    return {"harness_head": head, "pinned_commit": commit, "guarded": list(GUARDED_PATHS)}


def check_output_dir(root: Path, out_dir: Path, git: Callable[..., bytes] = _git) -> Path:
    """Artifacts go only to an untracked, git-ignored directory under data/processed/."""
    resolved = out_dir.resolve()
    parent = (root / "data" / "processed").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise BenchmarkError("the output directory must be inside data/processed/")
    rel = resolved.relative_to(root.resolve()).as_posix()
    if git(root, "ls-files", "-z", "--", rel).strip(b"\0").strip():
        raise BenchmarkError("the output directory contains tracked files")
    try:
        git(root, "check-ignore", "-q", "--no-index", f"{rel}/probe.json")
    except GitError:
        raise BenchmarkError("the output directory is not git-ignored") from None
    return resolved


@contextlib.contextmanager
def dropped_after(admin, name: str, development_db: str) -> Iterator[str]:
    """Guarantee `DROP DATABASE IF EXISTS` after a run process, even on failure or interrupt."""
    from sqlalchemy import text

    assert_scratch_name(name, development_db)
    try:
        yield name
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


# ---- raw record integrity and derivation (pure; used by --verify) --------------------------------


def check_record_stream(header: dict, records: list[dict]) -> None:
    """Records must be complete per run, in (run, sequence) order, unduplicated and typed."""
    phase, runs = header["protocol"]["phase"], header["protocol"]["runs"]
    expected_run, expected_sequence = 1, 0
    for record in records:
        for key in ("experiment_id", "run", "sequence", "record"):
            if key not in record:
                raise BenchmarkError(f"a record is missing {key!r}")
        if record["experiment_id"] != header["experiment_id"]:
            raise BenchmarkError("a record carries a different experiment_id")
        if record["record"] not in RECORD_TYPES[phase]:
            raise BenchmarkError(f"unexpected record type {record['record']!r}")
        if record["run"] == expected_run + 1 and record["sequence"] == 0 and expected_sequence:
            expected_run, expected_sequence = expected_run + 1, 0
        if (record["run"], record["sequence"]) != (expected_run, expected_sequence):
            raise BenchmarkError(
                f"records are missing, duplicated or reordered near run {record['run']} "
                f"sequence {record['sequence']}"
            )
        expected_sequence += 1
    if expected_run != runs or not records:
        raise BenchmarkError(f"expected records for {runs} runs, found {expected_run}")


def _index(records: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for record in records:
        out.setdefault(tuple(record.get(k) for k in keys), []).append(record)
    return out


def _one(index: dict[tuple, list[dict]], key: tuple, what: str) -> dict:
    found = index.get(key, [])
    if len(found) != 1:
        raise BenchmarkError(f"{what}: expected exactly one record, found {len(found)}")
    return found[0]


def check_header_identity(header: dict) -> dict:
    from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION

    for key in ("catalog", "embedding_text_version", "embedding_corpus_digest", "not_applicable"):
        if key not in header:
            raise BenchmarkError(f"header is missing {key!r} (incomplete artifact)")
    identity = check_protocol_identity()
    if header.get("protocol_identity") != identity:
        raise BenchmarkError("header query/expectation identity differs from the pinned protocol")
    if header["catalog"] != committed_catalog_identity():
        raise BenchmarkError("header catalog identity differs from the committed seed provenance")
    if header["embedding_text_version"] != EMBEDDING_TEXT_VERSION:
        raise BenchmarkError("header embedding_text_version differs from the active builder")
    if header["embedding_corpus_digest"] != embedding_corpus_digest():
        raise BenchmarkError("header embedding corpus digest differs from the active builder")
    if header["not_applicable"] != NOT_APPLICABLE:
        raise BenchmarkError("header not_applicable block is missing fields or reasons")
    return identity


def _run_facts(index, run: int, catalog: dict) -> dict[str, dict]:
    info = _one(index["run"], (run,), f"run {run} run record")
    expected = [{k: catalog[k] for k in CATALOG_KEYS}]
    if info.get("catalog_datasets") != expected:
        raise BenchmarkError(f"run {run}: scratch catalog identity differs from the header")
    if info.get("product_count") != SEED_PRODUCTS or info.get("embedding_rows") != SEED_PRODUCTS:
        raise BenchmarkError(f"run {run}: product or embedding count is not {SEED_PRODUCTS}")
    facts_record = _one(index["facts"], (run,), f"run {run} facts")
    facts = facts_record["facts"]
    if len(facts) != SEED_PRODUCTS or facts_sha256(facts) != facts_record["facts_sha256"]:
        raise BenchmarkError(f"run {run}: catalog facts are incomplete or altered")
    for pid, row in facts.items():
        if set(row) != set(FACT_FIELDS):
            raise BenchmarkError(f"run {run}: facts for {pid} lack {FACT_FIELDS}")
    return facts


def derive_s(header: dict, records: list[dict], identity: dict) -> dict:
    protocol = header["protocol"]
    grid, runs = protocol["rrf_k_grid"], protocol["runs"]
    if grid != list(RRF_K_GRID) or runs != 2 or protocol["top_k"] != TOP_K:
        raise BenchmarkError("Phase S protocol differs from the predeclared grid, runs or top_k")
    expectations = {e["query"]: e["expectations"] for e in identity["expectation_table"]}
    index = {
        "run": _index([r for r in records if r["record"] == "run"], "run"),
        "facts": _index([r for r in records if r["record"] == "facts"], "run"),
        "sources": _index([r for r in records if r["record"] == "sources"], "run", "query"),
        "hybrid": _index([r for r in records if r["record"] == "hybrid"], "run", "query", "rrf_k"),
    }
    out: dict = {"proxy_label": PROXY_LABEL, "runs": {}}
    comparable: dict[int, dict] = {}
    first_aggregates: dict[int, Fraction] = {}
    queries = identity["queries"]
    for run in range(1, runs + 1):
        expected_count = 2 + len(queries) * (1 + len(grid))
        found = sum(1 for r in records if r["run"] == run)
        if found != expected_count:
            raise BenchmarkError(f"run {run}: expected {expected_count} records, found {found}")
        facts = _run_facts(index, run, header["catalog"])
        per_query: dict = {}
        scores: dict[int, list[Fraction]] = {k: [] for k in grid}
        for query in identity["queries"]:
            src = _one(index["sources"], (run, query), f"run {run} {query!r} sources")
            lex, den = src["lexical"], src["dense"]
            check_ranked(lex["ids"], lex["ranks"], f"run {run} {query!r} lexical")
            check_ranked(den["ids"], den["ranks"], f"run {run} {query!r} dense")
            if (
                len(lex["ids"]) > protocol["source_depth"]
                or len(den["ids"]) > protocol["source_depth"]
            ):
                raise BenchmarkError(f"run {run} {query!r}: a source list exceeds the depth")
            if src["sources_sha256"] != sources_sha256(lex, den):
                raise BenchmarkError(f"run {run} {query!r}: source lists differ from their hash")
            if src["model_load_ms"] is not None:
                raise BenchmarkError(
                    f"run {run} {query!r}: the model was (re)loaded during source retrieval"
                )
            entry: dict = {
                "status": CHECKED if expectations[query] else DIAGNOSTIC,
                "lexical": {
                    "ids": lex["ids"],
                    "ranks": lex["ranks"],
                    "count": len(lex["ids"]),
                    "zero_results": not lex["ids"],
                },
                "dense": {
                    "ids": den["ids"],
                    "ranks": den["ranks"],
                    "count": len(den["ids"]),
                    "zero_results": not den["ids"],
                },
                "hybrid": {},
            }
            for k in grid:
                hyb = _one(index["hybrid"], (run, query, k), f"run {run} {query!r} rrf_k={k}")
                if hyb["sources_sha256"] != src["sources_sha256"]:
                    raise BenchmarkError(
                        f"run {run} {query!r} rrf_k={k}: fusion consumed different source lists"
                    )
                fusion = hyb["fusion"]
                if (fusion["method"], fusion["rrf_k"]) != ("rrf", k) or (
                    fusion["lexical_k"],
                    fusion["dense_k"],
                ) != (protocol["source_depth"], protocol["source_depth"]):
                    raise BenchmarkError(f"run {run} {query!r} rrf_k={k}: fusion settings differ")
                fused = rrf_oracle(lex["ids"], den["ids"], k)
                expected_ids = fused[: fusion["candidate_k"]][:TOP_K]
                if hyb["ids"] != expected_ids:
                    raise BenchmarkError(
                        f"run {run} {query!r} rrf_k={k}: hybrid order differs from the RRF oracle "
                        "over the recorded source lists"
                    )
                exact = rrf_scores(lex["ids"], den["ids"], k)
                if hyb["rrf_scores"] != [float(exact[p]) for p in expected_ids] or hyb[
                    "rrf_score_fractions"
                ] != [str(exact[p]) for p in expected_ids]:
                    raise BenchmarkError(f"run {run} {query!r} rrf_k={k}: rrf scores differ")
                lex_rank = {p: i for i, p in enumerate(lex["ids"], start=1)}
                den_rank = {p: i for i, p in enumerate(den["ids"], start=1)}
                if hyb["lexical_ranks"] != [lex_rank.get(p) for p in hyb["ids"]] or hyb[
                    "dense_ranks"
                ] != [den_rank.get(p) for p in hyb["ids"]]:
                    raise BenchmarkError(f"run {run} {query!r} rrf_k={k}: source ranks differ")
                counts = {
                    "lexical_hit_count": len(lex["ids"]),
                    "dense_hit_count": len(den["ids"]),
                    "overlap_count": len(set(lex["ids"]) & set(den["ids"])),
                    "fused_count": len(fused),
                    "candidate_count": min(len(fused), fusion["candidate_k"]),
                    "result_count": len(expected_ids),
                }
                if hyb["counts"] != counts:
                    raise BenchmarkError(f"run {run} {query!r} rrf_k={k}: counts differ")
                score = consistency(hyb["ids"], facts, expectations[query])
                if score is not None:
                    scores[k].append(Fraction(score["score"]))
                entry["hybrid"][str(k)] = {
                    "ids": hyb["ids"],
                    "ranks": list(range(1, len(hyb["ids"]) + 1)),
                    "result_count": len(hyb["ids"]),
                    "zero_results": not hyb["ids"],
                    "composition": composition(hyb["lexical_ranks"], hyb["dense_ranks"]),
                    "consistency": score if score is not None else DIAGNOSTIC,
                }
            per_query[query] = entry
        aggregates = {k: mean_fraction(scores[k]) for k in grid}
        out["runs"][str(run)] = {
            "facts_sha256": facts_sha256(facts),
            "checked_query_count": len(identity["checked_queries"]),
            "per_query": per_query,
            "mean_consistency": {str(k): frac(v) for k, v in aggregates.items()},
        }
        comparable[run] = {
            "facts": facts_sha256(facts),
            "lists": {
                q: [e["lexical"]["ids"], e["dense"]["ids"]]
                + [e["hybrid"][str(k)]["ids"] for k in grid]
                for q, e in per_query.items()
            },
        }
        if run == 1:
            first_aggregates = aggregates
    mismatches = [
        q
        for q in identity["queries"]
        if any(comparable[r]["lists"][q] != comparable[1]["lists"][q] for r in comparable)
    ]
    deterministic = not mismatches and all(
        comparable[r]["facts"] == comparable[1]["facts"] for r in comparable
    )
    out["determinism"] = {
        "runs_identical": deterministic,
        "mismatched_queries": mismatches,
        "rule": "every source and hybrid list of every query and rrf_k identical in both runs",
    }
    out["decision"] = decide_rrf_k(first_aggregates, deterministic)
    return out


def _metric(sample: dict, metric: str):
    return sample[metric] if metric in sample else sample["stages"].get(metric)


def derive_cl(header: dict, records: list[dict], identity: dict) -> dict:
    protocol = header["protocol"]
    runs, samples = protocol["runs"], protocol["samples_per_query"]
    expectations = {e["query"]: e["expectations"] for e in identity["expectation_table"]}
    index = {
        "run": _index([r for r in records if r["record"] == "run"], "run"),
        "facts": _index([r for r in records if r["record"] == "facts"], "run"),
        "results": _index(
            [r for r in records if r["record"] == "results"], "run", "endpoint", "query"
        ),
        "sample": _index(
            [r for r in records if r["record"] == "sample"], "run", "endpoint", "query", "phase"
        ),
    }
    out: dict = {"proxy_label": PROXY_LABEL, "label": "provisional, non-authoritative", "runs": {}}
    lists_by_run: dict[int, dict] = {}
    fusion_seen: list[dict] = []
    per_group = 1 + protocol["cold_per_query"] + samples  # results + cold + timed
    for run in range(1, runs + 1):
        expected_count = 2 + len(ENDPOINTS) * len(identity["queries"]) * per_group
        found = sum(1 for r in records if r["run"] == run)
        if found != expected_count:
            raise BenchmarkError(f"run {run}: expected {expected_count} records, found {found}")
        info = _one(index["run"], (run,), f"run {run} run record")
        order = endpoint_order(run)
        if tuple(info.get("endpoint_order", ())) != order:
            raise BenchmarkError(f"run {run}: endpoint order differs from the declared schedule")
        facts = _run_facts(index, run, header["catalog"])
        latency: dict = {}
        stamps: list[tuple[int, str]] = []
        results: dict[str, dict] = {}
        for endpoint in order:
            stages = ENDPOINTS[endpoint]["stages"]
            metrics = ["request_ms", "app_total_ms", "serialization_ms"] + [
                s for s in stages if s not in NULLABLE_STAGES
            ]
            timed_all: list[dict] = []
            per_query: dict = {}
            model_loads = []
            for query in identity["queries"]:
                what = f"run {run} {endpoint} {query!r}"
                cold = index["sample"].get((run, endpoint, query, "cold"), [])
                timed = index["sample"].get((run, endpoint, query, "timed"), [])
                if len(cold) != protocol["cold_per_query"] or len(timed) != samples:
                    raise BenchmarkError(
                        f"{what}: expected {protocol['cold_per_query']} cold and {samples} timed "
                        f"samples, found {len(cold)} and {len(timed)}"
                    )
                if [s["sample_index"] for s in timed] != list(range(samples)):
                    raise BenchmarkError(f"{what}: timed sample indexes are not 0..n-1")
                result = _one(index["results"], (run, endpoint, query), f"{what} results")
                expected_hash = ids_sha256(result["ids"])
                for sample in (*cold, *timed):
                    if sample["result_ids_sha256"] != expected_hash:
                        raise BenchmarkError(f"{what}: results changed between samples")
                    if set(sample["stages"]) != set(stages):
                        raise BenchmarkError(f"{what}: samples lack stage fields {stages}")
                    for metric in metrics:
                        value = _metric(sample, metric)
                        if not isinstance(value, int | float) or isinstance(value, bool):
                            raise BenchmarkError(f"{what}: {metric} is missing or not numeric")
                    stamps.append((sample["sequence"], sample["timestamp_utc"]))
                if "model_load_ms" in stages and cold[0]["stages"]["model_load_ms"] is not None:
                    model_loads.append({"query": query, "ms": cold[0]["stages"]["model_load_ms"]})

                per_query[query] = {
                    "cold": {m: _metric(cold[0], m) for m in metrics},
                    "timed": {m: summarize([_metric(s, m) for s in timed]) for m in metrics},
                }
                timed_all += [{m: _metric(s, m) for m in metrics} for s in timed]
                results.setdefault(query, {})[endpoint] = result
            latency[endpoint] = {
                "metrics": metrics,
                "overall": {m: summarize([s[m] for s in timed_all]) for m in metrics},
                "per_query": per_query,
                "model_load_ms_cold": model_loads,
            }
        stamps.sort()
        gaps = sorted(
            (
                (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(),
                seq_b,
            )
            for (_, a), (seq_b, b) in zip(stamps, stamps[1:], strict=False)
        )
        comparison: dict = {}
        consistency_by_endpoint: dict[str, list[Fraction]] = {e: [] for e in ENDPOINTS}
        lexical_tie_top10 = {"search": 0, "hybrid": 0}
        for query in identity["queries"]:
            by_endpoint = results[query]
            hybrid = by_endpoint["hybrid"]
            fusion_seen.append(hybrid["fusion"])
            check_ranked(hybrid["ids"], list(range(1, len(hybrid["ids"]) + 1)), "hybrid")
            entry: dict = {
                "status": CHECKED if expectations[query] else DIAGNOSTIC,
                "result_count": {e: len(by_endpoint[e]["ids"]) for e in ENDPOINTS},
                "zero_results": {e: not by_endpoint[e]["ids"] for e in ENDPOINTS},
                "overlap": {
                    "search|dense": overlap(
                        by_endpoint["search"]["ids"], by_endpoint["dense"]["ids"]
                    ),
                    "search|hybrid": overlap(
                        by_endpoint["search"]["ids"], by_endpoint["hybrid"]["ids"]
                    ),
                    "dense|hybrid": overlap(
                        by_endpoint["dense"]["ids"], by_endpoint["hybrid"]["ids"]
                    ),
                },
                "hybrid_composition": composition(hybrid["lexical_ranks"], hybrid["dense_ranks"]),
                "consistency": {},
            }
            for endpoint in ENDPOINTS:
                score = consistency(by_endpoint[endpoint]["ids"], facts, expectations[query])
                entry["consistency"][endpoint] = score if score is not None else DIAGNOSTIC
                if score is not None:
                    consistency_by_endpoint[endpoint].append(Fraction(score["score"]))
            if has_score_tie(by_endpoint["search"]["lexical_scores"]):
                lexical_tie_top10["search"] += 1
            if has_score_tie(hybrid["lexical_scores"]):
                lexical_tie_top10["hybrid"] += 1
            comparison[query] = entry
        lists_by_run[run] = {
            q: {e: results[q][e]["ids"] for e in ENDPOINTS} for q in identity["queries"]
        }
        out["runs"][str(run)] = {
            "endpoint_order": list(order),
            "facts_sha256": facts_sha256(facts),
            "latency": latency,
            "max_sampling_gap_s": round(gaps[-1][0], 3) if gaps else 0.0,
            "largest_gaps": [
                {"gap_s": round(g, 3), "before_sequence": s} for g, s in reversed(gaps[-5:])
            ],
            "comparison": {
                "per_query": comparison,
                "mean_consistency": {
                    e: frac(mean_fraction(v)) for e, v in consistency_by_endpoint.items()
                },
                "top10_with_tied_lexical_scores": {
                    **lexical_tie_top10,
                    "note": "within the returned top-10 only (ties are ordered by product_id)",
                },
            },
        }
    mismatched = [
        q
        for q in identity["queries"]
        if any(lists_by_run[r][q] != lists_by_run[1][q] for r in lists_by_run)
    ]
    out["determinism"] = {
        "within_run": "every cold and timed sample returned the recorded result list",
        "across_runs_identical": not mismatched,
        "mismatched_queries": mismatched,
    }
    statuses = {(f["rrf_k"], f["rrf_k_status"]) for f in fusion_seen}
    if len(statuses) != 1:
        raise BenchmarkError("the hybrid fusion settings changed between requests or runs")
    ((rrf_k, status),) = statuses
    out["hybrid_fusion"] = {"rrf_k": rrf_k, "rrf_k_status": status}
    return out


def derive(header: dict, records: list[dict]) -> dict:
    """Every reported figure, recomputed from raw records only. Raises on incomplete data."""
    identity = check_header_identity(header)
    check_record_stream(header, records)
    phase = header["protocol"]["phase"]
    if phase == "S":
        return derive_s(header, records, identity)
    if phase == "CL":
        return derive_cl(header, records, identity)
    raise BenchmarkError(f"unknown phase {phase!r}")


# ---- artifacts -----------------------------------------------------------------------------------


def raw_body(header: dict, records: list[dict]) -> str:
    ordered = sorted(records, key=lambda r: (r["run"], r["sequence"]))
    return "\n".join([dumps({"record": "header", **header}), *(dumps(r) for r in ordered)]) + "\n"


def write_artifacts(output_dir: Path, header: dict, records: list[dict], extra: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = header["experiment_id"]
    records = sorted(records, key=lambda r: (r["run"], r["sequence"]))
    body = raw_body(header, records)
    raw_path = output_dir / f"{experiment_id}.samples.jsonl"
    raw_path.write_bytes(body.encode("utf-8"))  # exact bytes: LF endings on every platform
    summary = {
        "experiment_id": experiment_id,
        "raw_samples": {
            "file": raw_path.name,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "records": len(records),
        },
        "header": header,
        "derived": json.loads(json.dumps(derive(header, records))),
        "metadata_not_recomputed": extra,
    }
    json_path = output_dir / f"{experiment_id}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / f"{experiment_id}.md").write_text(markdown(summary), encoding="utf-8")
    return json_path


def verify_artifacts(json_path: Path, expected_protocol: dict | None = None) -> list[str]:
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    raw_path = json_path.with_name(summary["raw_samples"]["file"])
    problems = []
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != summary["raw_samples"]["sha256"]:
        problems.append("raw samples file hash differs from the summary")
    header, records = read_samples(raw_path)
    header = {k: v for k, v in header.items() if k != "record"}
    if header != summary["header"]:
        problems.append("raw header differs from the summary header")
    if len(records) != summary["raw_samples"]["records"]:
        problems.append("raw record count differs from the summary")
    phase = header.get("protocol", {}).get("phase")
    expected = expected_protocol or PREDECLARED.get(phase)
    if header.get("protocol") != expected:
        problems.append("the protocol differs from the predeclared protocol")
    try:
        again = json.loads(json.dumps(derive(header, records)))
    except (BenchmarkError, KeyError, TypeError, ValueError) as exc:
        return [*problems, f"raw records are incomplete or inconsistent: {exc}"]
    if again != summary["derived"]:
        problems.append(
            "derived summary (scores, overlaps, latency or decision) does not recompute"
        )
    return problems


def markdown(summary: dict) -> str:
    d, h = summary["derived"], summary["header"]
    ident = h["protocol_identity"]
    lines = [
        f"# Hybrid harness (generated) `{summary['experiment_id']}`, "
        f"phase {h['protocol']['phase']}",
        "",
        f"- Harness HEAD `{h['provenance']['harness_head']}`; pinned commit "
        f"`{h['provenance']['pinned_commit']}`",
        f"- Queries: {ident['count']} (sha256 `{ident['query_set_sha256']}`); expectation table "
        f"sha256 `{ident['expectation_table_sha256']}`",
        f"- {PROXY_LABEL}.",
        "",
    ]
    if h["protocol"]["phase"] == "S":
        decision = d["decision"]
        lines += [
            "## Provisional rrf_k selection (candidate; not applied)",
            "",
            "| rrf_k | mean consistency (proxy) |",
            "|---|---|",
            *(
                f"| {k} | {v['fraction']} ({v['float']}) |"
                for k, v in decision["mean_consistency"].items()
            ),
            "",
            f"- Determinism (2 runs identical): {d['determinism']['runs_identical']}",
            f"- Status: `{decision['status']}`; selected rrf_k: {decision['selected_rrf_k']}; "
            f"tied: {decision['tied_rrf_k']}; tie rule: {decision['tie_rule']}",
        ]
    else:
        lines += [
            "## Latency (per run; ms; provisional)",
            "",
            "| run | endpoint | metric | p50 | p95 | p99 | max |",
            "|---|---|---|---|---|---|---|",
        ]
        for run, data in d["runs"].items():
            for endpoint, lat in data["latency"].items():
                for metric, s in lat["overall"].items():
                    lines.append(
                        f"| {run} | {endpoint} | {metric} | {s['p50']} | {s['p95']} | {s['p99']} "
                        f"| {s['max']} |"
                    )
        lines += ["", "## Comparison (provisional, non-authoritative)", ""]
        for run, data in d["runs"].items():
            means = data["comparison"]["mean_consistency"]
            lines.append(
                f"- run {run}: mean consistency proxy "
                + ", ".join(f"{e} {v['fraction']}" for e, v in means.items())
            )
        lines.append(f"- Determinism across runs: {d['determinism']['across_runs_identical']}")
    return "\n".join(lines) + "\n"


def protocol_table(identity: dict) -> str:
    lines = [
        f"query set sha256:         {identity['query_set_sha256']}",
        f"expectation table sha256: {identity['expectation_table_sha256']}",
        "",
        " # | query                        | status          | expectations",
    ]
    for i, entry in enumerate(identity["expectation_table"], start=1):
        expect = ", ".join(
            f"{e['field']}={e['value']!r} (term {e['term']!r})" for e in entry["expectations"]
        )
        lines.append(f"{i:2d} | {entry['query']:<28} | {entry['status']:<15} | {expect or '-'}")
    return "\n".join(lines)


# ---- run process (database and model work; executed only with approval) --------------------------


def source_lists(hits: Sequence, score: str) -> dict:
    return {
        "ids": [h.product_id for h in hits],
        "ranks": [h.rank for h in hits],
        "scores": [getattr(h, score) for h in hits],
    }


def source_fetcher(
    embedder,
    spec,
    open_session: Callable,
    depth: int,
    read: Callable | None = None,
    encode: Callable | None = None,
) -> Callable[[str], tuple]:
    """Production retrieval inputs for one query: the API's query encoding (`encode_query`) with
    the one already-loaded `embedder`, then `read_sources` (lexical + dense in one REPEATABLE READ
    READ ONLY snapshot) at the hybrid source depth. Returns `(lexical, dense, model_load_ms)`;
    `model_load_ms` is None unless this call had to load the model."""
    if read is None:
        from ecommerce_search.search.hybrid import read_sources as read
    if encode is None:
        from ecommerce_search.api.dense import encode_query as encode

    def fetch(query: str) -> tuple:
        vector, load_ms, _ = encode(embedder, spec, query, ("query", "q"))
        with open_session() as session:
            lexical, dense = read(session, query, vector, spec, depth, depth)
        return lexical, dense, load_ms

    return fetch


def collect_phase_s(
    fetch: Callable[[str], tuple],
    queries: Sequence[str],
    emit: Callable,
    candidate_k: int,
    fuse: Callable | None = None,
) -> None:
    """Retrieve each query's sources exactly once, then fuse those same immutable lists with
    production `fuse_rrf` for every grid value. rrf_k never reaches retrieval."""
    if fuse is None:
        from ecommerce_search.search.hybrid import fuse_rrf as fuse
    for qi, query in enumerate(queries):
        lexical, dense, load_ms = fetch(query)
        lex_hits, den_hits = tuple(lexical.hits), tuple(dense.hits)
        lex = source_lists(lex_hits, "lexical_score")
        den = source_lists(den_hits, "dense_score")
        fingerprint = sources_sha256(lex, den)
        emit(
            {
                "record": "sources",
                "query_index": qi,
                "query": query,
                "lexical": lex,
                "dense": den,
                "sources_sha256": fingerprint,
                "model_load_ms": load_ms,
            }
        )
        lex_ids, den_ids = set(lex["ids"]), set(den["ids"])
        for k in RRF_K_GRID:
            fused = fuse(lex_hits, den_hits, k, candidate_k)
            after = sources_sha256(
                source_lists(lex_hits, "lexical_score"), source_lists(den_hits, "dense_score")
            )
            if after != fingerprint:
                raise BenchmarkError(f"fusion altered the source lists for {query!r}")
            top = fused[:TOP_K]
            emit(
                {
                    "record": "hybrid",
                    "query_index": qi,
                    "query": query,
                    "rrf_k": k,
                    "sources_sha256": fingerprint,
                    "fusion": {
                        "method": "rrf",
                        "implementation": "ecommerce_search.search.hybrid.fuse_rrf",
                        "rrf_k": k,
                        "lexical_k": SOURCE_DEPTH,
                        "dense_k": SOURCE_DEPTH,
                        "candidate_k": candidate_k,
                    },
                    "ids": [c.product_id for c in top],
                    "lexical_ranks": [c.lexical_rank for c in top],
                    "dense_ranks": [c.dense_rank for c in top],
                    "rrf_scores": [float(c.rrf_score) for c in top],
                    "rrf_score_fractions": [str(c.rrf_score) for c in top],
                    "counts": {
                        "lexical_hit_count": len(lex_hits),
                        "dense_hit_count": len(den_hits),
                        "overlap_count": len(lex_ids & den_ids),
                        "fused_count": len(lex_ids | den_ids),
                        "candidate_count": len(fused),
                        "result_count": len(top),
                    },
                }
            )


def _get(client, path: str, query: str, top_k: int) -> dict:
    response = client.get(path, params={"q": query, "top_k": top_k})
    if response.status_code != 200:
        raise BenchmarkError(f"{path} returned {response.status_code} for {query!r}")
    return response.json()


def results_record(endpoint: str, query: str, body: dict) -> dict:
    rows = body["results"]
    record = {
        "record": "results",
        "endpoint": endpoint,
        "query": query,
        "ids": [r["product_id"] for r in rows],
        "lexical_scores": [r.get("lexical_score") for r in rows],
    }
    if endpoint != "search":
        record["dense_scores"] = [r.get("dense_score") for r in rows]
    if endpoint == "hybrid":
        record.update(
            {
                "lexical_ranks": [r["lexical_rank"] for r in rows],
                "dense_ranks": [r["dense_rank"] for r in rows],
                "rrf_scores": [r["rrf_score"] for r in rows],
                "fusion": body["fusion"],
            }
        )
    return record


def collect_phase_cl(client, schemas: dict, queries, run: int, emit, warmup, samples) -> None:
    for endpoint in endpoint_order(run):
        path, stages = ENDPOINTS[endpoint]["path"], ENDPOINTS[endpoint]["stages"]
        for qi, query in enumerate(queries):
            expected = None
            for phase, count in (("cold", 1), ("warmup", warmup), ("timed", samples)):
                for i in range(count):
                    wall = datetime.now(UTC).isoformat(timespec="microseconds")
                    started = time.perf_counter()
                    response = client.get(path, params={"q": query, "top_k": TOP_K})
                    request_ms = (time.perf_counter() - started) * 1000
                    if response.status_code != 200:
                        raise BenchmarkError(f"{path} {response.status_code} for {query!r}")
                    body = response.json()
                    ids_hash = ids_sha256([r["product_id"] for r in body["results"]])
                    if expected is None:
                        expected = ids_hash
                        emit({**results_record(endpoint, query, body), "query_index": qi})
                    elif ids_hash != expected:
                        raise BenchmarkError(f"{path}: results changed for {query!r}")
                    if phase == "warmup":
                        continue
                    parsed = schemas[endpoint].model_validate(body)
                    started = time.perf_counter()
                    parsed.model_dump_json()
                    serialization_ms = (time.perf_counter() - started) * 1000
                    lat = body["latency_ms"]
                    emit(
                        {
                            "record": "sample",
                            "endpoint": endpoint,
                            "query_index": qi,
                            "query": query,
                            "phase": phase,
                            "sample_index": i,
                            "timestamp_utc": wall,
                            "status_code": response.status_code,
                            "result_count": body["result_count"],
                            "result_ids_sha256": ids_hash,
                            "request_ms": request_ms,
                            "app_total_ms": lat["total_ms"],
                            "serialization_ms": serialization_ms,
                            "stages": {s: lat[s] for s in stages},
                        }
                    )


def child_main(args: argparse.Namespace) -> int:  # pragma: no cover - needs a database and model
    install_network_guard()
    os.environ["HF_HUB_OFFLINE"] = "1"
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from ecommerce_search.api.app import create_app
    from ecommerce_search.api.schemas import (
        DenseSearchResponse,
        HybridSearchResponse,
        SearchResponse,
    )
    from ecommerce_search.config import get_settings
    from ecommerce_search.embeddings.sentence_transformers_provider import (
        SentenceTransformerEmbedder,
    )
    from ecommerce_search.ingestion.service import ingest_file
    from ecommerce_search.search.dense_indexing import embed, embedding_status

    phase = args.child_phase
    if phase not in PREDECLARED or not isinstance(args.run, int):
        raise BenchmarkError("child needs a known phase and a run number")
    if phase == "CL" and (args.warmup, args.samples) != (
        PREDECLARED["CL"]["warmup_per_query"],
        PREDECLARED["CL"]["samples_per_query"],
    ):
        raise BenchmarkError("child sampling differs from the predeclared protocol")
    settings = get_settings()
    name = assert_scratch_name(args.database, settings.postgres_db)
    queries = check_protocol_identity()["queries"]
    records: list[dict] = []
    sequence = 0

    def emit(record: dict) -> None:
        nonlocal sequence
        records.append(
            {"experiment_id": args.experiment_id, "run": args.run, "sequence": sequence, **record}
        )
        sequence += 1

    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    engine = None
    try:
        with scratch_database(admin, name, settings.postgres_db):
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database"] = name
            command.upgrade(config, "head")
            engine = create_engine(settings.database_url(database=name))
            seed = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
            outcome = ingest_file(engine, seed, seed.with_name("catalog_seed_v1.provenance.json"))
            if outcome.result is None or outcome.result.inserted != SEED_PRODUCTS:
                raise BenchmarkError("the committed seed did not ingest as 240 products")
            spec = settings.embedding_spec()
            provider = SentenceTransformerEmbedder(
                spec, settings.resolved_models_dir(), settings.embedding_batch_size
            )
            provider.load()
            first = embed(engine, provider)
            with Session(engine) as session:
                status = embedding_status(session, spec, provider)
            if first.inserted != SEED_PRODUCTS or not status.current:
                raise BenchmarkError("embeddings are not CURRENT for all 240 products")
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("ANALYZE"))
                datasets = [
                    dict(row)
                    for row in conn.execute(
                        text(
                            "SELECT dataset_id, dataset_version, checksum_sha256, record_count, "
                            "transform_version, taxonomy_version, rules_version "
                            "FROM catalog_datasets ORDER BY id"
                        )
                    ).mappings()
                ]
                counts = {
                    "product_count": conn.execute(
                        text("SELECT count(*) FROM products")
                    ).scalar_one(),
                    "embedding_rows": conn.execute(
                        text("SELECT count(*) FROM product_embeddings")
                    ).scalar_one(),
                }
                facts = {
                    row.product_id: {"category": row.category, "brand": row.brand, "anc": row.anc}
                    for row in conn.execute(
                        text(
                            "SELECT p.product_id, p.category, p.brand, h.anc FROM products p "
                            "LEFT JOIN headphone_specs h ON h.product_id = p.product_id "
                            "ORDER BY p.product_id"
                        )
                    )
                }
            emit(
                {
                    "record": "run",
                    "phase": phase,
                    "endpoint_order": list(endpoint_order(args.run)) if phase == "CL" else [],
                    "catalog_datasets": datasets,
                    **counts,
                }
            )
            emit({"record": "facts", "facts": facts, "facts_sha256": facts_sha256(facts)})
            run_settings = settings.model_copy(update={"postgres_db": name})
            if phase == "S":
                # The provider that generated the embeddings (loaded once above) encodes every
                # query; each query is retrieved once and fused for every grid value.
                fetch = source_fetcher(provider, spec, lambda: Session(engine), SOURCE_DEPTH)
                collect_phase_s(fetch, queries, emit, settings.search_candidate_k)
            else:
                schemas = {
                    "search": SearchResponse,
                    "dense": DenseSearchResponse,
                    "hybrid": HybridSearchResponse,
                }
                with TestClient(create_app(run_settings)) as client:
                    collect_phase_cl(
                        client, schemas, queries, args.run, emit, args.warmup, args.samples
                    )
            engine.dispose()
            engine = None
    finally:
        if engine is not None:
            engine.dispose()
        admin.dispose()
    Path(args.out).write_text("\n".join(dumps(r) for r in records) + "\n", encoding="utf-8")
    return 0


# ---- parent: orchestration -----------------------------------------------------------------------


def parent_main(phase: str) -> int:  # pragma: no cover - runs the benchmark
    from sqlalchemy import create_engine

    from ecommerce_search.config import get_settings
    from ecommerce_search.embeddings.fetch import verify_manifest
    from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION

    install_network_guard()
    provenance = check_provenance(ROOT, phase)
    identity = check_protocol_identity()
    out_dir = check_output_dir(ROOT, OUTPUT_DIR)
    settings = get_settings()
    model = check_model(settings.resolved_models_dir(), settings.embedding_spec(), verify_manifest)
    protocol = PREDECLARED[phase]
    started = datetime.now(UTC)
    experiment_id = f"hybrid-m5-{phase.lower()}-{started.strftime('%Y%m%dT%H%M%SZ')}-"
    experiment_id += uuid.uuid4().hex[:8]
    header = {
        "experiment_id": experiment_id,
        "generated_at_utc": started.isoformat(timespec="seconds"),
        "provenance": provenance,
        "source": source_fingerprint(ROOT),
        "model": model,
        "catalog": committed_catalog_identity(),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_corpus_digest": embedding_corpus_digest(),
        "not_applicable": NOT_APPLICABLE,
        "metric_definitions": METRIC_DEFINITIONS,
        "protocol_identity": identity,
        "protocol": protocol,
        "settings": {
            "search_rrf_k": settings.search_rrf_k,
            "search_candidate_k": settings.search_candidate_k,
            "search_lexical_k": settings.search_lexical_k,
            "search_dense_k": settings.search_dense_k,
        },
        "environment": server_environment(settings),
    }
    records: list[dict] = []
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with tempfile.TemporaryDirectory(prefix="hybrid-bench-") as tmp:
            for run in range(1, protocol["runs"] + 1):
                name = assert_scratch_name(new_scratch_name(), settings.postgres_db)
                out = Path(tmp) / f"run{run}.jsonl"
                print(f"phase {phase} run {run}/{protocol['runs']} ...", flush=True)
                argv = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--child-phase",
                    phase,
                    "--run",
                    str(run),
                    "--database",
                    name,
                    "--experiment-id",
                    experiment_id,
                    "--out",
                    str(out),
                ]
                if phase == "CL":
                    argv += [
                        "--warmup",
                        str(protocol["warmup_per_query"]),
                        "--samples",
                        str(protocol["samples_per_query"]),
                    ]
                with dropped_after(admin, name, settings.postgres_db):
                    subprocess.run(  # noqa: S603 - fixed argv, sys.executable
                        argv, cwd=ROOT, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}
                    )
                records += [
                    json.loads(line)
                    for line in out.read_text(encoding="utf-8").splitlines()
                    if line
                ]
    finally:
        admin.dispose()
    ended = datetime.now(UTC)
    extra = {
        "power_events": power_events(started, ended),
        "window_utc": [started.isoformat(timespec="seconds"), ended.isoformat(timespec="seconds")],
    }
    json_path = write_artifacts(out_dir, header, records, extra)
    problems = verify_artifacts(json_path)
    if problems:
        raise BenchmarkError("artifact self-verification failed: " + "; ".join(problems))
    print(f"wrote {json_path.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=("S", "CL"), help="run a phase (requires approval)")
    mode.add_argument("--verify", type=Path, help="recompute a summary from its raw records")
    mode.add_argument(
        "--print-protocol", action="store_true", help="print the frozen query/expectation table"
    )
    mode.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-phase", dest="child_phase", help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-id", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_protocol:
        print(protocol_table(check_protocol_identity()))
        return 0
    if args.verify:
        problems = verify_artifacts(args.verify)
        print("\n".join(problems) or "OK: summary and decision recompute exactly from raw records")
        return 1 if problems else 0
    if args.child:
        return child_main(args)
    return parent_main(args.phase)


if __name__ == "__main__":
    sys.exit(main())
