"""Milestone 4 dense benchmark: generation, production exact retrieval, experimental HNSW.

Scratch databases only (`ecommerce_search_bench_<12 hex>`); the development database is refused
by name and never connected to for data. Offline: the pinned MiniLM snapshot must verify against
its manifest, and outbound non-loopback sockets are refused in every run process.

Provenance guard: the working tree must be clean and every runtime-affecting path (src/,
migrations/, pyproject.toml, uv.lock, docker-compose.yml, alembic.ini) must be identical between
the Phase B production commit and the benchmark HEAD. Only the harness and its tests may differ.

Protocol (predeclared; 3 runs, each in a fresh subprocess with a fresh scratch database):
  A  embedding generation: model import and load, first `embed()` (wall time, products/s, encode
     time measured by a pass-through wrapper, residual), memory, second-run idempotency
     (unchanged rows, unchanged `embedded_at`), CURRENT status with zero vector mismatches.
  B1 production, end to end: in-process GET /search/dense, top_k=10, using the application's own
     provider (the first request's model load is recorded separately).
  B2 production SQL: `dense_search()` / RETRIEVAL_SQL at k=10 and k=50 with pre-encoded query
     vectors, plus EXPLAIN and a check that no vector index exists.
  C1 EXPERIMENTAL (not production SQL): an ANN-compatible query (ORDER BY distance LIMIT k in a
     CTE, then the content-hash join), no vector index.
  C2 EXPERIMENTAL (not production SQL): the identical query text with a temporary HNSW index,
     timed with session-local `enable_seqscan = off`; natural (unforced) plans are recorded.
  C-phase order per run is fixed: run 1 C1>C2, run 2 C2>C1, run 3 C1>C2. Each phase gets its own
  untimed warm-ups. Per query and k: 1 cold sample, `--warmup` untimed, `--samples` timed.

Everything in the summary (latency tables, determinism, recall, plan facts and the decision
rule inputs) is derived from the raw JSONL records; `--verify` recomputes it and compares.
Artifacts (git-ignored): data/processed/dense_benchmark/<experiment_id>.{json,md,samples.jsonl}.

Usage (execution requires a separate approval):
  HF_HUB_OFFLINE=1 uv run --offline python scripts/benchmark_dense.py
  uv run --offline python scripts/benchmark_dense.py --verify <artifact .json>
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_lexical import ProvenanceError as GitError  # noqa: E402
from benchmark_lexical import (  # noqa: E402 - scripts/ is not a package
    _git,
    dumps,
    read_samples,
    source_fingerprint,
    summarize,
)
from embedding_model_selection import QUERIES as SELECTION_QUERIES  # noqa: E402
from embedding_model_selection import _peak_rss_mb  # noqa: E402

from ecommerce_search.search.query import normalize_query  # noqa: E402

PRODUCTION_COMMIT = "0ab972e1cc45dd489b262a39962a4c6debea5c43"
GUARDED_PATHS: tuple[str, ...] = (
    "src/",
    "migrations/",
    "pyproject.toml",
    "uv.lock",
    "docker-compose.yml",
    "alembic.ini",
)
QUERIES: tuple[str, ...] = tuple(SELECTION_QUERIES)  # single source of truth
EXPECTED_QUERY_COUNT = 12
QUERY_MAX_LENGTH = 200  # the API's default SEARCH_MAX_QUERY_LENGTH
K_VALUES: tuple[int, ...] = (10, 50)
API_TOP_K = 10
RUNS = 3
PHASE_SCHEDULE: dict[int, tuple[str, str]] = {1: ("C1", "C2"), 2: ("C2", "C1"), 3: ("C1", "C2")}
SEED_PRODUCTS = 240
BENCH_DB_PREFIX = "ecommerce_search_bench_"
BENCH_DB_RE = re.compile(r"^ecommerce_search_bench_[0-9a-f]{12}$")
INDEX_NAME = "ix_bench_hnsw_product_embeddings"
OUTPUT_DIR = ROOT / "data" / "processed" / "dense_benchmark"
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE_FILE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"
TIE_EPSILON = 1e-9

SUITES: dict[str, dict] = {
    "B1": {
        "name": "production end-to-end GET /search/dense (top_k=10)",
        "label": "production",
        "metrics": (
            "request_ms",
            "app_total_ms",
            "query_embedding_ms",
            "vector_ms",
            "serialization_ms",
        ),
    },
    "B2": {
        "name": "production dense_search() / RETRIEVAL_SQL (exact scan)",
        "label": "production",
        "metrics": ("vector_ms",),
    },
    "C1": {
        "name": "EXPERIMENTAL ANN-compatible SQL, no vector index (not production SQL)",
        "label": "experimental",
        "metrics": ("ann_sql_ms",),
    },
    "C2": {
        "name": "EXPERIMENTAL ANN-compatible SQL, temporary HNSW, forced (not production SQL)",
        "label": "experimental",
        "metrics": ("ann_sql_ms",),
    },
}
DERIVED_METRIC_NOTE = (
    "request_overhead_ms = request_ms - app_total_ms (derived): routing, request validation, "
    "dependency resolution and framework response serialization inside the in-process "
    "TestClient call. It is not a direct serialization measurement and not network latency."
)
SERIALIZATION_NOTE = (
    "serialization_ms (measured directly, M3 convention): the response payload is parsed with "
    "DenseSearchResponse.model_validate(...) and DenseSearchResponse.model_dump_json() is timed "
    "on its own, after the request_ms interval has ended, so it is never part of request_ms."
)
METRIC_DEFINITIONS: dict[str, str] = {
    "request_ms": "in-process TestClient GET, ends before response.json(); not network latency",
    "app_total_ms": "handler entry through result-model construction (API latency_ms.total_ms)",
    "query_embedding_ms": "query token check and encoding (API latency_ms.query_embedding_ms)",
    "vector_ms": "vector SQL and row fetch (API latency_ms.vector_ms / dense_search())",
    "serialization_ms": SERIALIZATION_NOTE,
    "request_overhead_ms": DERIVED_METRIC_NOTE,
    "ann_sql_ms": "EXPERIMENTAL ANN-compatible SQL execute and fetch (not production SQL)",
}

# Experiment metadata that does not apply in Milestone 4 (explicit, never null).
NOT_APPLICABLE: dict[str, str] = {
    "golden_dataset_version": "not applicable: the human-reviewed Golden Dataset is created in "
    "Milestone 9; none exists in Milestone 4",
    "relevance_quality_metrics": "not applicable: no relevance or quality metric is computed "
    "before the Golden Dataset (Milestones 9-10); this benchmark measures latency, "
    "determinism and exact-vs-HNSW agreement only",
    "decision_provider": "not applicable: no decision provider exists in Milestone 4 "
    "(deterministic provider in Milestone 6, optional Jev in Milestones 11-12)",
    "policy_prompt_version": "not applicable: no model policy or prompt is used in Milestone 4 "
    "(bounded decision providers arrive in Milestones 11-15)",
    "reranker": "not applicable: no cross-encoder reranker exists before Milestone 8",
    "cost": "not applicable: local CPU model and local database only; no paid API or metered "
    "service is used, so there is no per-request cost to report",
}

# Predeclared decision rule (approved before execution; never tuned after seeing results).
MIN_RELATIVE_SAVING = 0.25
MIN_ABSOLUTE_SAVING_MS = 1.0
MIN_END_TO_END_SHARE = 0.05
REQUIRED_TIE_AWARE_RECALL = 1.0

# Experimental ANN-compatible SQL: identical text for C1 and C2. Filters are the production
# contract; content-stale rows are removed by the hash join, never returned as hits.
ANN_SQL = """
WITH ann AS (
    SELECT e.product_id, e.source_content_sha256,
           e.embedding <=> CAST(:qvec AS vector) AS distance
    FROM product_embeddings e
    WHERE e.model_id = :model_id
      AND e.model_revision = :model_revision
      AND e.embedding_config_sha256 = :config_sha
      AND e.embedding_text_version = :text_version
      AND e.dimension = :dimension
      AND e.normalized = :normalized
    ORDER BY e.embedding <=> CAST(:qvec AS vector)
    LIMIT :limit
)
SELECT ann.product_id, ann.distance
FROM ann JOIN products p ON p.product_id = ann.product_id
WHERE ann.source_content_sha256 = p.content_sha256
ORDER BY ann.distance ASC, ann.product_id ASC
"""


class BenchmarkError(RuntimeError):
    """The benchmark cannot produce trustworthy results; nothing partial is reported as a pass."""


class ProvenanceError(BenchmarkError):
    """The tree, the production commit or the model snapshot is not as required."""


# ---- identity, provenance and safety (pure; unit-tested) --------------------------------------


def query_set_identity(queries: Sequence[str] = QUERIES) -> dict:
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise BenchmarkError(f"expected {EXPECTED_QUERY_COUNT} queries, got {len(queries)}")
    if len(set(queries)) != len(queries):
        raise BenchmarkError("the query set contains duplicates")
    normalized = [normalize_query(q, QUERY_MAX_LENGTH) for q in queries]
    if len(set(normalized)) != len(normalized):
        raise BenchmarkError("two queries normalize to the same text")
    return {
        "count": len(queries),
        "source": "scripts/embedding_model_selection.py QUERIES",
        "sha256": hashlib.sha256(dumps(list(queries)).encode("utf-8")).hexdigest(),
        "normalized_sha256": hashlib.sha256(dumps(normalized).encode("utf-8")).hexdigest(),
        "normalized": normalized,
    }


def embedding_corpus_digest() -> str:
    """The pinned embedding-corpus digest of the committed seed (same canonical form as
    tests/unit/search/test_embedding_corpus_digest.py, which pins it per text version)."""
    from ecommerce_search.embeddings import text as et
    from ecommerce_search.ingestion.loader import load_catalog

    catalog = load_catalog(SEED)
    semantics = {
        "connectivity_phrases": {k.value: v for k, v in et.CONNECTIVITY_PHRASES.items()},
        "texts": [
            {"product_id": line.record.product_id, "text": et.build_embedding_text(line.record)}
            for line in sorted(catalog.lines, key=lambda line: line.record.product_id)
        ],
    }
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


CATALOG_FIELDS: tuple[str, ...] = (
    "dataset_id",
    "dataset_version",
    "checksum_sha256",
    "record_count",
    "transform_version",
    "taxonomy_version",
    "rules_version",
)


def committed_catalog_identity() -> dict:
    """Dataset identity from the committed seed provenance file (and the committed rules
    version, which ingestion records but the provenance file does not carry)."""
    from ecommerce_search.catalog.quality.parameters import RULES_VERSION

    provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    if hashlib.sha256(SEED.read_bytes()).hexdigest() != provenance["checksum_sha256"]:
        raise BenchmarkError("the committed seed does not match its provenance checksum")
    identity = {name: provenance[name] for name in CATALOG_FIELDS if name != "rules_version"}
    identity["rules_version"] = RULES_VERSION
    return {
        **identity,
        "sources": {
            "rules_version": "ecommerce_search.catalog.quality.parameters.RULES_VERSION",
            "other_fields": "data/seed/catalog_seed_v1.provenance.json",
        },
    }


def check_header_identity(header: dict) -> None:
    """Refuse artifacts whose identity is missing or differs from the committed sources."""
    from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION

    for key in ("catalog", "embedding_text_version", "embedding_corpus_digest", "not_applicable"):
        if key not in header:
            raise BenchmarkError(f"header is missing {key!r} (incomplete or older artifact)")
    if header["catalog"] != committed_catalog_identity():
        raise BenchmarkError("header catalog identity differs from the committed seed provenance")
    if header["embedding_text_version"] != EMBEDDING_TEXT_VERSION:
        raise BenchmarkError("header embedding_text_version differs from the active builder")
    if header["embedding_corpus_digest"] != embedding_corpus_digest():
        raise BenchmarkError("header embedding corpus digest differs from the active builder")
    if header["not_applicable"] != NOT_APPLICABLE:
        raise BenchmarkError("header not_applicable block is missing fields or reasons")


def vectors_sha256(vectors: Sequence[Sequence[float]]) -> str:
    canonical = "\n".join(",".join(repr(float(v)) for v in vector) for vector in vectors)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def check_provenance(
    root: Path,
    production_commit: str = PRODUCTION_COMMIT,
    git: Callable[..., bytes] = _git,
) -> dict:
    """Refuse a dirty tree or any runtime-affecting difference from the production commit."""
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise ProvenanceError("the working tree is not clean; commit or remove changes first")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    try:
        git(root, "merge-base", "--is-ancestor", production_commit, head)
    except (GitError, ProvenanceError):
        raise ProvenanceError(
            f"the production commit {production_commit} is not an ancestor of HEAD"
        ) from None
    changed = git(root, "diff", "--name-only", production_commit, head, "--", *GUARDED_PATHS)
    differing = [line for line in changed.decode().splitlines() if line.strip()]
    if differing:
        raise ProvenanceError(
            "runtime-affecting paths differ from the production commit: "
            + ", ".join(differing[:10])
        )
    return {"harness_head": head, "production_commit": production_commit, "guarded": GUARDED_PATHS}


def check_model(models_dir: Path | None, spec, verify: Callable[..., list[str]]) -> dict:
    if models_dir is None:
        raise ProvenanceError("no models directory is known (not a source checkout)")
    problems = verify(models_dir, spec.model_id, spec.revision)
    if problems:
        raise ProvenanceError(f"the pinned model snapshot failed verification: {problems[:3]}")
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dimension": spec.dimension,
        "max_seq_length": spec.max_seq_length,
        "normalize": spec.normalize,
        "config_sha256": spec.config_sha256(),
        "manifest_problems": [],
    }


def new_scratch_name() -> str:
    return f"{BENCH_DB_PREFIX}{uuid.uuid4().hex[:12]}"


def assert_scratch_name(name: str, development_db: str) -> str:
    """Only uniquely named benchmark databases are ever created or dropped."""
    if name == development_db:
        raise BenchmarkError("refusing to use the development database")
    if not BENCH_DB_RE.fullmatch(name):
        raise BenchmarkError("scratch database names must match ecommerce_search_bench_<12 hex>")
    return name


def phase_order(run: int) -> tuple[str, str]:
    if run not in PHASE_SCHEDULE:
        raise BenchmarkError(f"no predeclared C-phase order for run {run}")
    return PHASE_SCHEDULE[run]


# ---- ranking and agreement (pure; unit-tested) ------------------------------------------------


def validate_ranking(ids: Sequence[str], distances: Sequence[float]) -> None:
    """Distances ascending; equal distances in ascending product_id; no duplicates."""
    if len(ids) != len(distances) or len(set(ids)) != len(ids):
        raise BenchmarkError("result list has duplicates or mismatched lengths")
    for i in range(1, len(ids)):
        if distances[i] < distances[i - 1]:
            raise BenchmarkError("results are not ordered by ascending distance")
        if distances[i] == distances[i - 1] and ids[i] < ids[i - 1]:
            raise BenchmarkError("tied results are not ordered by ascending product_id")


def strict_recall(exact_ids: Sequence[str], other_ids: Sequence[str], k: int) -> float:
    return len(set(exact_ids[:k]) & set(other_ids[:k])) / k


def tie_aware_recall(
    exact_distances: Sequence[float], other_distances: Sequence[float], k: int
) -> float:
    """Share of k slots filled by results within the exact k-th distance (ties count).

    A missing result (a list shorter than k) never counts, so short lists lower the value."""
    if not exact_distances:
        return 0.0
    cutoff = exact_distances[min(k, len(exact_distances)) - 1] + TIE_EPSILON
    return sum(1 for d in other_distances[:k] if d <= cutoff) / k


def max_distance_difference(exact: dict[str, float], other: dict[str, float]) -> float:
    common = set(exact) & set(other)
    return max((abs(exact[p] - other[p]) for p in common), default=0.0)


# ---- decision rule (pure; unit-tested) ----------------------------------------------------------


def decide(
    c1_p50: dict[int, float],
    c2_p50: dict[int, float],
    b1_median_request_ms: float,
    tie_aware_recalls: Sequence[float],
    natural_plan_uses_hnsw: Sequence[bool],
) -> dict:
    """Apply the predeclared rule; every criterion is reported individually."""
    runs = sorted(c1_p50)
    spread = max(
        max(c1_p50.values()) - min(c1_p50.values()), max(c2_p50.values()) - min(c2_p50.values())
    )
    savings = {r: c1_p50[r] - c2_p50[r] for r in runs}
    per_run = {
        r: {
            "c1_p50_ms": c1_p50[r],
            "c2_p50_ms": c2_p50[r],
            "saving_ms": savings[r],
            "beats_spread": savings[r] > spread,
            "relative_ok": c1_p50[r] > 0 and savings[r] >= MIN_RELATIVE_SAVING * c1_p50[r],
            "absolute_ok": savings[r] >= MIN_ABSOLUTE_SAVING_MS,
        }
        for r in runs
    }
    latency = all(
        v["beats_spread"] and v["relative_ok"] and v["absolute_ok"] for v in per_run.values()
    )
    min_saving = min(savings.values())
    share_needed = MIN_END_TO_END_SHARE * b1_median_request_ms
    criteria = {
        "1_latency_every_run": {
            "passed": latency,
            "spread_ms": spread,
            "per_run": per_run,
            "rule": "saving > larger run-to-run p50 spread, >= 25% of C1 p50, >= 1 ms; every run",
        },
        "2_end_to_end_share": {
            "passed": min_saving >= share_needed,
            "min_saving_ms": min_saving,
            "b1_median_request_ms": b1_median_request_ms,
            "required_ms": share_needed,
            "rule": "smallest per-run saving >= 5% of the B1 median request_ms",
        },
        "3_tie_aware_recall": {
            "passed": bool(tie_aware_recalls)
            and all(r >= REQUIRED_TIE_AWARE_RECALL for r in tie_aware_recalls),
            "min": min(tie_aware_recalls) if tie_aware_recalls else None,
            "rule": "tie-aware recall@k = 1.0 for every query, k in {10, 50} and run",
        },
        "4_natural_plan_uses_hnsw": {
            "passed": bool(natural_plan_uses_hnsw) and all(natural_plan_uses_hnsw),
            "uses": sum(bool(u) for u in natural_plan_uses_hnsw),
            "of": len(natural_plan_uses_hnsw),
            "rule": "the planner chooses the HNSW index without enable_seqscan = off",
        },
    }
    flag = all(c["passed"] for c in criteria.values())
    return {
        "criteria": criteria,
        "flag_hnsw_for_separate_review": flag,
        "outcome": (
            "flag HNSW for a separate review (no automatic change)"
            if flag
            else "retain exact scan at 240 rows (approved default)"
        ),
    }


# ---- derivation from raw records (pure; unit-tested; used by --verify) ------------------------


def _group(records, **match):
    return [r for r in records if all(r.get(k) == v for k, v in match.items())]


def derive(header: dict, records: list[dict]) -> dict:
    """Every reported figure, recomputed from raw records only. Raises on incomplete data."""
    check_header_identity(header)
    expected_catalog = {name: header["catalog"][name] for name in CATALOG_FIELDS}
    queries = header["queries"]["normalized"]
    samples, k_values = header["protocol"]["samples_per_query"], header["protocol"]["k_values"]
    runs = header["protocol"]["runs"]
    out: dict = {"runs": {}, "suites": {name: dict(meta) for name, meta in SUITES.items()}}
    for meta in out["suites"].values():
        meta["metrics"] = list(meta["metrics"])
    recalls_tie, natural_uses, c1_p50, c2_p50, b1_request = [], [], {}, {}, []
    for run in range(1, runs + 1):
        run_info = _group(records, record="run", run=run)
        generation = _group(records, record="generation", run=run)
        if len(run_info) != 1 or len(generation) != 1:
            raise BenchmarkError(f"run {run}: missing run or generation record")
        if tuple(run_info[0]["phase_order"]) != phase_order(run):
            raise BenchmarkError(f"run {run}: phase order differs from the declared schedule")
        if run_info[0].get("catalog_datasets") != [expected_catalog]:
            raise BenchmarkError(
                f"run {run}: scratch catalog_datasets identity differs from the committed "
                "seed provenance (dataset id, version, checksum, record count or versions)"
            )
        if run_info[0].get("product_count") != SEED_PRODUCTS:
            raise BenchmarkError(f"run {run}: product count is not {SEED_PRODUCTS}")
        if generation[0].get("embedding_rows") != SEED_PRODUCTS:
            raise BenchmarkError(f"run {run}: embedding row count is not {SEED_PRODUCTS}")
        run_out: dict = {
            "phase_order": run_info[0]["phase_order"],
            "catalog_datasets": run_info[0]["catalog_datasets"],
            "product_count": run_info[0]["product_count"],
            "generation": generation[0],
            "index": _group(records, record="index", run=run),
            "vector_index_check": _group(records, record="index_check", run=run),
            "suites": {},
            "recall": {},
            "plans": {},
        }
        for suite in SUITES:
            ks = [API_TOP_K] if suite == "B1" else k_values
            for k in ks:
                key = f"{suite}@{k}"
                timed_all = []
                per_query = {}
                for q in queries:
                    group = _group(records, record="sample", run=run, suite=suite, k=k, query=q)
                    cold = [s for s in group if s["phase"] == "cold"]
                    timed = [s for s in group if s["phase"] == "timed"]
                    if len(cold) != 1 or len(timed) != samples:
                        raise BenchmarkError(
                            f"run {run} {key} {q!r}: expected 1 cold and {samples} timed samples, "
                            f"found {len(cold)} and {len(timed)}"
                        )
                    result = _group(records, record="results", run=run, suite=suite, k=k, query=q)
                    if len(result) != 1:
                        raise BenchmarkError(f"run {run} {key} {q!r}: missing result list")
                    expected_hash = ids_sha256(result[0]["ids"])
                    if any(s["result_ids_sha256"] != expected_hash for s in group):
                        raise BenchmarkError(
                            f"run {run} {key} {q!r}: results changed between samples"
                        )
                    validate_ranking(result[0]["ids"], result[0]["distances"])
                    missing = [m for s in group for m in SUITES[suite]["metrics"] if m not in s]
                    if missing:
                        raise BenchmarkError(
                            f"run {run} {key} {q!r}: samples lack metrics {sorted(set(missing))} "
                            "(incomplete or older artifact)"
                        )
                    per_query[q] = {
                        "timed": {
                            m: summarize([s[m] for s in timed]) for m in SUITES[suite]["metrics"]
                        },
                        "cold": {m: cold[0][m] for m in SUITES[suite]["metrics"]},
                    }
                    timed_all += timed
                metrics = {
                    m: summarize([s[m] for s in timed_all]) for m in SUITES[suite]["metrics"]
                }
                if suite == "B1":
                    metrics["request_overhead_ms"] = summarize(
                        [s["request_ms"] - s["app_total_ms"] for s in timed_all]
                    )
                    b1_request += [s["request_ms"] for s in timed_all]
                run_out["suites"][key] = {
                    "label": SUITES[suite]["label"],
                    "overall": metrics,
                    "per_query": per_query,
                }
        for k in k_values:
            for q in queries:
                exact = _group(records, record="results", run=run, suite="B2", k=k, query=q)[0]
                for other in ("C1", "C2"):
                    res = _group(records, record="results", run=run, suite=other, k=k, query=q)[0]
                    entry = {
                        "strict_recall": strict_recall(exact["ids"], res["ids"], k),
                        "tie_aware_recall": tie_aware_recall(
                            exact["distances"], res["distances"], k
                        ),
                        "identical_order": exact["ids"] == res["ids"],
                        "max_distance_difference": max_distance_difference(
                            dict(zip(exact["ids"], exact["distances"], strict=True)),
                            dict(zip(res["ids"], res["distances"], strict=True)),
                        ),
                    }
                    run_out["recall"][f"{other}@{k}|{q}"] = entry
                    if other == "C2":
                        recalls_tie.append(entry["tie_aware_recall"])
        for plan in _group(records, record="plan", run=run):
            run_out["plans"][f"{plan['mode']}@{plan['k']}|{plan['query']}"] = {
                "nodes": plan["nodes"],
                "uses_hnsw": plan["uses_hnsw"],
            }
            if plan["mode"] == "C2_natural":
                natural_uses.append(plan["uses_hnsw"])
            if plan["mode"] == "C2_forced" and not plan["uses_hnsw"]:
                raise BenchmarkError(f"run {run}: forced C2 plan does not use the HNSW index")
            if plan["mode"] == "B2_production" and plan["uses_hnsw"]:
                raise BenchmarkError(f"run {run}: production plan unexpectedly uses an index")
        expected_plans = len(queries) * len(k_values)
        for mode in ("B2_production", "C1_no_index", "C2_natural", "C2_forced"):
            found = len(_group(records, record="plan", run=run, mode=mode))
            if found != expected_plans:
                raise BenchmarkError(
                    f"run {run}: expected {expected_plans} {mode} plans, found {found}"
                )
        c1_p50[run] = run_out["suites"]["C1@10"]["overall"]["ann_sql_ms"]["p50"]
        c2_p50[run] = run_out["suites"]["C2@10"]["overall"]["ann_sql_ms"]["p50"]
        stamps = sorted(
            (s["sequence"], s["timestamp_utc"]) for s in _group(records, record="sample", run=run)
        )
        gaps = [
            (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
            for (_, a), (_, b) in zip(stamps, stamps[1:], strict=False)
        ]
        run_out["max_sampling_gap_s"] = round(max(gaps, default=0.0), 3)
        out["runs"][str(run)] = run_out
    out["decision"] = decide(
        c1_p50, c2_p50, statistics.median(b1_request), recalls_tie, natural_uses
    )
    out["derived_metric_note"] = DERIVED_METRIC_NOTE
    return out


# ---- cleanup helpers (unit-tested with fakes) ---------------------------------------------------


@contextlib.contextmanager
def scratch_database(admin, name: str, development_db: str) -> Iterator[str]:
    """CREATE a uniquely named benchmark database and always DROP it, even on failure."""
    from sqlalchemy import text

    assert_scratch_name(name, development_db)
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@contextlib.contextmanager
def temporary_hnsw_index(engine, m: int, ef_construction: int) -> Iterator[dict]:
    """Build the experimental HNSW index (scratch database only) and always drop it."""
    from sqlalchemy import text

    if (
        not (isinstance(m, int) and isinstance(ef_construction, int))
        or m < 2
        or ef_construction < 4
    ):
        raise BenchmarkError("invalid HNSW parameters")
    started = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE INDEX {INDEX_NAME} ON product_embeddings USING hnsw "
                f"(embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction})"
            )
        )
    build_ms = (time.perf_counter() - started) * 1000
    try:
        with engine.begin() as conn:
            conn.execute(text("ANALYZE product_embeddings"))
            size = conn.execute(text(f"SELECT pg_relation_size('{INDEX_NAME}')")).scalar_one()
        yield {"build_ms": round(build_ms, 3), "size_bytes": int(size)}
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))


def install_network_guard() -> None:
    """Refuse outbound non-loopback name resolution and connections in this process."""
    loopback = {"127.0.0.1", "::1", "localhost"}
    real_getaddrinfo, real_create = socket.getaddrinfo, socket.create_connection

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in loopback and host is not None:
            raise OSError("network access is disabled in the dense benchmark")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_create(address, *args, **kwargs):
        if address[0] not in loopback:
            raise OSError("network access is disabled in the dense benchmark")
        return real_create(address, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = guarded_create  # type: ignore[assignment]


# ---- run process (database and model work; executed only with approval) ----------------------


def plan_nodes(plan_json) -> tuple[list[str], bool]:
    plan = plan_json if isinstance(plan_json, list) else json.loads(plan_json)
    nodes: list[str] = []
    uses = False

    def walk(node: dict) -> None:
        nonlocal uses
        label = node["Node Type"]
        if "Index Name" in node:
            label += f" [{node['Index Name']}]"
            uses = uses or node["Index Name"] == INDEX_NAME
        if "Relation Name" in node:
            label += f" on {node['Relation Name']}"
        nodes.append(label)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan[0]["Plan"])
    return nodes, uses


class EncodeTimer:
    """Pass-through Embedder wrapper: times embed_documents only; behaviour is unchanged."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.encode_ms = 0.0

    @property
    def spec(self):
        return self.inner.spec

    def load(self):
        return self.inner.load()

    def count_tokens(self, texts, kind):
        return self.inner.count_tokens(texts, kind)

    def embed_documents(self, texts):
        started = time.perf_counter()
        try:
            return self.inner.embed_documents(texts)
        finally:
            self.encode_ms += (time.perf_counter() - started) * 1000

    def embed_query(self, text):
        return self.inner.embed_query(text)


def child_main(args: argparse.Namespace) -> int:  # pragma: no cover - needs a database and model
    install_network_guard()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import psutil
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from ecommerce_search.api.app import create_app
    from ecommerce_search.api.schemas import DenseSearchResponse
    from ecommerce_search.config import get_settings
    from ecommerce_search.ingestion.service import ingest_file
    from ecommerce_search.search.dense import RETRIEVAL_SQL, dense_search, retrieval_params
    from ecommerce_search.search.dense_indexing import embed, embedding_status

    settings = get_settings()
    name = assert_scratch_name(args.database, settings.postgres_db)
    spec = settings.embedding_spec()
    models_dir = settings.resolved_models_dir()
    order = phase_order(args.run)
    queries = list(query_set_identity()["normalized"])
    process = psutil.Process()
    records: list[dict] = []
    sequence = 0

    def emit(record: dict) -> None:
        nonlocal sequence
        records.append(
            {"experiment_id": args.experiment_id, "run": args.run, "sequence": sequence, **record}
        )
        sequence += 1

    def sample(suite, k, query, phase, index, ids, metrics):
        emit(
            {
                "record": "sample",
                "suite": suite,
                "label": SUITES[suite]["label"],
                "k": k,
                "query": query,
                "phase": phase,
                "sample_index": index,
                "phase_order": list(order),
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="microseconds"),
                "result_count": len(ids),
                "result_ids_sha256": ids_sha256(ids),
                **metrics,
            }
        )

    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    engine = None
    try:
        with scratch_database(admin, name, settings.postgres_db):
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database"] = name
            command.upgrade(config, "head")
            engine = create_engine(settings.database_url(database=name))
            outcome = ingest_file(engine, SEED, PROVENANCE_FILE)
            if outcome.result is None or outcome.result.inserted != SEED_PRODUCTS:
                raise BenchmarkError("the committed seed did not ingest as 240 products")
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("ANALYZE"))
                pgvector = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar_one()
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
                product_count = conn.execute(text("SELECT count(*) FROM products")).scalar_one()
            emit(
                {
                    "record": "run",
                    "phase_order": list(order),
                    "pgvector": pgvector,
                    "catalog_datasets": datasets,
                    "product_count": product_count,
                }
            )

            # A: generation
            rss_start = process.memory_info().rss / 2**20
            started = time.perf_counter()
            import sentence_transformers
            import torch

            import_ms = (time.perf_counter() - started) * 1000
            from ecommerce_search.embeddings.sentence_transformers_provider import (
                SentenceTransformerEmbedder,
            )

            provider = SentenceTransformerEmbedder(spec, models_dir, settings.embedding_batch_size)
            load_ms = provider.load()
            rss_loaded = process.memory_info().rss / 2**20
            timer = EncodeTimer(provider)
            started = time.perf_counter()
            first = embed(engine, timer)
            first_ms = (time.perf_counter() - started) * 1000
            with engine.connect() as conn:
                stamps = dict(
                    conn.execute(
                        text("SELECT product_id, embedded_at FROM product_embeddings")
                    ).all()
                )
            started = time.perf_counter()
            second = embed(engine, provider)
            second_ms = (time.perf_counter() - started) * 1000
            with engine.connect() as conn:
                stamps_after = dict(
                    conn.execute(
                        text("SELECT product_id, embedded_at FROM product_embeddings")
                    ).all()
                )
            with Session(engine) as session:
                status = embedding_status(session, spec, provider)
            if (first.inserted, first.changed_during_run) != (SEED_PRODUCTS, 0):
                raise BenchmarkError("first embed did not insert 240 rows cleanly")
            if (second.inserted, second.updated, second.unchanged) != (0, 0, SEED_PRODUCTS):
                raise BenchmarkError("second embed was not idempotent")
            if stamps != stamps_after or not status.current or status.counts["vector_mismatch"]:
                raise BenchmarkError(
                    "embeddings are not CURRENT and unchanged after the second run"
                )
            emit(
                {
                    "record": "generation",
                    "products": SEED_PRODUCTS,
                    "import_ms": round(import_ms, 3),
                    "load_ms": load_ms,
                    "first_embed_ms": round(first_ms, 3),
                    "products_per_second": round(SEED_PRODUCTS / (first_ms / 1000), 3),
                    "encode_ms": round(timer.encode_ms, 3),
                    "residual_ms": round(first_ms - timer.encode_ms, 3),
                    "residual_note": "first_embed_ms - encode_ms: planning, text building, "
                    "token-limit "
                    "check and the database write transaction",
                    "first": {
                        "inserted": first.inserted,
                        "updated": first.updated,
                        "unchanged": first.unchanged,
                        "changed_during_run": first.changed_during_run,
                    },
                    "second": {
                        "inserted": second.inserted,
                        "updated": second.updated,
                        "unchanged": second.unchanged,
                        "changed_during_run": second.changed_during_run,
                        "ms": round(second_ms, 3),
                    },
                    "embedded_at_unchanged": True,
                    "status_current": True,
                    "vector_mismatch": 0,
                    "rss_mb": {
                        "start": round(rss_start, 1),
                        "after_load": round(rss_loaded, 1),
                        "after_embed": round(process.memory_info().rss / 2**20, 1),
                        "peak": round(_peak_rss_mb(process), 1),
                    },
                    "versions": {
                        "torch": torch.__version__,
                        "sentence_transformers": sentence_transformers.__version__,
                        "transformers": __import__("transformers").__version__,
                    },
                    "torch_num_threads": torch.get_num_threads(),
                    "embedding_rows": len(stamps),
                }
            )

            # B1: production end to end (the application's own provider: cold load recorded)
            run_settings = settings.model_copy(update={"postgres_db": name})
            with TestClient(create_app(run_settings)) as client:
                for qi, query in enumerate(queries):
                    for phase, count in (
                        ("cold", 1),
                        ("warmup", args.warmup),
                        ("timed", args.samples),
                    ):
                        for i in range(count):
                            started = time.perf_counter()
                            response = client.get(
                                "/search/dense", params={"q": query, "top_k": API_TOP_K}
                            )
                            request_ms = (time.perf_counter() - started) * 1000
                            if response.status_code != 200:
                                raise BenchmarkError(
                                    f"B1 status {response.status_code} for {query!r}"
                                )
                            body = response.json()
                            ids = [r["product_id"] for r in body["results"]]
                            if phase == "warmup":
                                continue
                            # M3 convention: parse, then time model_dump_json() on its own,
                            # outside (after) the request_ms interval.
                            parsed = DenseSearchResponse.model_validate(body)
                            started = time.perf_counter()
                            parsed.model_dump_json()
                            serialization_ms = (time.perf_counter() - started) * 1000
                            lat = body["latency_ms"]
                            sample(
                                "B1",
                                API_TOP_K,
                                query,
                                phase,
                                i,
                                ids,
                                {
                                    "request_ms": request_ms,
                                    "app_total_ms": lat["total_ms"],
                                    "query_embedding_ms": lat["query_embedding_ms"],
                                    "vector_ms": lat["vector_ms"],
                                    "serialization_ms": serialization_ms,
                                    "model_load_ms": lat["model_load_ms"]
                                    if qi == 0 and phase == "cold"
                                    else None,
                                },
                            )
                            if phase == "cold":
                                emit(
                                    {
                                        "record": "results",
                                        "suite": "B1",
                                        "k": API_TOP_K,
                                        "query": query,
                                        "ids": ids,
                                        "distances": [
                                            1.0 - r["dense_score"] for r in body["results"]
                                        ],
                                    }
                                )

            vectors = [provider.embed_query(q) for q in queries]
            emit(
                {
                    "record": "vectors",
                    "query_vectors_sha256": vectors_sha256(vectors),
                    "per_query": {
                        q: vectors_sha256([v]) for q, v in zip(queries, vectors, strict=True)
                    },
                }
            )

            # B2: production SQL, exact scan, before any experimental index exists
            with engine.connect() as conn:
                indexes = sorted(
                    conn.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename = 'product_embeddings'"
                        )
                    ).scalars()
                )
                ann = conn.execute(
                    text(
                        "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                        "JOIN pg_am a ON a.oid = c.relam WHERE a.amname IN ('hnsw', 'ivfflat')"
                    )
                ).scalar_one()
            if indexes != ["pk_product_embeddings"] or ann:
                raise BenchmarkError("a vector index exists before the production measurements")
            emit({"record": "index_check", "indexes": indexes, "ann_indexes": ann})
            with Session(engine) as session:
                for k in K_VALUES:
                    for query, vector in zip(queries, vectors, strict=True):
                        for phase, count in (
                            ("cold", 1),
                            ("warmup", args.warmup),
                            ("timed", args.samples),
                        ):
                            for i in range(count):
                                result = dense_search(session, vector, spec, k)
                                ids = [h.product_id for h in result.hits]
                                if phase == "warmup":
                                    continue
                                sample(
                                    "B2", k, query, phase, i, ids, {"vector_ms": result.vector_ms}
                                )
                                if phase == "cold":
                                    emit(
                                        {
                                            "record": "results",
                                            "suite": "B2",
                                            "k": k,
                                            "query": query,
                                            "ids": ids,
                                            "distances": [1.0 - h.dense_score for h in result.hits],
                                        }
                                    )
                        raw = session.execute(
                            text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + str(RETRIEVAL_SQL)),
                            retrieval_params(vector, spec, k),
                        ).scalar_one()
                        nodes, uses = plan_nodes(raw)
                        emit(
                            {
                                "record": "plan",
                                "mode": "B2_production",
                                "k": k,
                                "query": query,
                                "nodes": nodes,
                                "uses_hnsw": uses,
                            }
                        )

            # C1 / C2: experimental, in the predeclared order
            from ecommerce_search.search.dense import retrieval_params as params_for

            def ann_phase(suite: str, forced: bool) -> None:
                with engine.connect() as conn:
                    conn.exec_driver_sql(f"SET hnsw.ef_search = {int(args.hnsw_ef_search)}")
                    if forced:
                        conn.exec_driver_sql("SET enable_seqscan = off")
                    for k in K_VALUES:
                        for query, vector in zip(queries, vectors, strict=True):
                            params = params_for(vector, spec, k)
                            for phase, count in (
                                ("cold", 1),
                                ("warmup", args.warmup),
                                ("timed", args.samples),
                            ):
                                for i in range(count):
                                    started = time.perf_counter()
                                    rows = conn.execute(text(ANN_SQL), params).all()
                                    ms = (time.perf_counter() - started) * 1000
                                    ids = [r.product_id for r in rows]
                                    if len(ids) != k:
                                        raise BenchmarkError(
                                            f"{suite} returned {len(ids)} rows for k={k}"
                                        )
                                    if phase == "warmup":
                                        continue
                                    sample(suite, k, query, phase, i, ids, {"ann_sql_ms": ms})
                                    if phase == "cold":
                                        emit(
                                            {
                                                "record": "results",
                                                "suite": suite,
                                                "k": k,
                                                "query": query,
                                                "ids": ids,
                                                "distances": [float(r.distance) for r in rows],
                                            }
                                        )
                            raw = conn.execute(
                                text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + ANN_SQL), params
                            ).scalar_one()
                            nodes, uses = plan_nodes(raw)
                            mode = "C2_forced" if forced else "C1_no_index"
                            emit(
                                {
                                    "record": "plan",
                                    "mode": mode,
                                    "k": k,
                                    "query": query,
                                    "nodes": nodes,
                                    "uses_hnsw": uses,
                                }
                            )
                    conn.rollback()

            def natural_plans() -> None:
                with engine.connect() as conn:
                    conn.exec_driver_sql(f"SET hnsw.ef_search = {int(args.hnsw_ef_search)}")
                    for k in K_VALUES:
                        for query, vector in zip(queries, vectors, strict=True):
                            raw = conn.execute(
                                text("EXPLAIN (FORMAT JSON) " + ANN_SQL),
                                params_for(vector, spec, k),
                            ).scalar_one()
                            nodes, uses = plan_nodes(raw)
                            emit(
                                {
                                    "record": "plan",
                                    "mode": "C2_natural",
                                    "k": k,
                                    "query": query,
                                    "nodes": nodes,
                                    "uses_hnsw": uses,
                                }
                            )
                    conn.rollback()

            for suite in order:
                if suite == "C1":
                    ann_phase("C1", forced=False)
                else:
                    with temporary_hnsw_index(
                        engine, args.hnsw_m, args.hnsw_ef_construction
                    ) as built:
                        emit(
                            {
                                "record": "index",
                                "m": args.hnsw_m,
                                "ef_construction": args.hnsw_ef_construction,
                                "ef_search": args.hnsw_ef_search,
                                **built,
                            }
                        )
                        natural_plans()
                        ann_phase("C2", forced=True)
            if engine is not None:
                engine.dispose()
                engine = None
    finally:
        if engine is not None:
            engine.dispose()
        admin.dispose()
    Path(args.out).write_text("\n".join(dumps(r) for r in records) + "\n", encoding="utf-8")
    return 0


# ---- parent: orchestration, artifacts, verification ----------------------------------------------


def environment(settings) -> dict:  # pragma: no cover - reads the server
    from sqlalchemy import create_engine, text

    admin = create_engine(settings.database_url(database="postgres"))
    try:
        with admin.connect() as conn:
            server = {"version": conn.execute(text("SELECT version()")).scalar_one()}
            for name in (
                "shared_buffers",
                "work_mem",
                "effective_cache_size",
                "max_parallel_workers_per_gather",
                "jit",
                "fsync",
                "synchronous_commit",
            ):
                server[name] = conn.execute(text(f"SHOW {name}")).scalar_one()
            server["pgvector_default_version"] = conn.execute(
                text("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
            ).scalar_one()
    finally:
        admin.dispose()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "postgres": server,
        "seed_sha256": hashlib.sha256(SEED.read_bytes()).hexdigest(),
    }


def power_events(start: datetime, end: datetime) -> dict:  # pragma: no cover - Windows only
    """Kernel-Power sleep/standby events in the window (local, read-only; best effort)."""
    if platform.system() != "Windows":
        return {"available": False, "reason": "not Windows"}
    fmt = "%Y-%m-%d %H:%M:%S"
    script = (
        "Get-WinEvent -FilterHashtable @{LogName='System'; "
        "ProviderName='Microsoft-Windows-Kernel-Power'; "
        f"StartTime=[datetime]'{start.astimezone().strftime(fmt)}'; "
        f"EndTime=[datetime]'{end.astimezone().strftime(fmt)}'"
        "} -ErrorAction SilentlyContinue | Where-Object { $_.Id -in @(42,506,507,107,187) } | "
        "ForEach-Object { '{0:o} {1}' -f $_.TimeCreated, $_.Id }"
    )
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    return {
        "available": True,
        "events": [line for line in done.stdout.splitlines() if line.strip()],
    }


def write_artifacts(output_dir: Path, header: dict, records: list[dict], extra: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = header["experiment_id"]
    ordered = sorted(records, key=lambda r: (r["run"], r["sequence"]))
    body = "\n".join([dumps({"record": "header", **header}), *(dumps(r) for r in ordered)]) + "\n"
    raw_path = output_dir / f"{experiment_id}.samples.jsonl"
    raw_path.write_bytes(body.encode("utf-8"))
    summary = {
        "experiment_id": experiment_id,
        "raw_samples": {
            "file": raw_path.name,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        "header": header,
        "derived": derive(header, ordered),
        "metadata_not_recomputed": extra,
    }
    json_path = output_dir / f"{experiment_id}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / f"{experiment_id}.md").write_text(markdown(summary), encoding="utf-8")
    return json_path


def verify_artifacts(json_path: Path) -> list[str]:
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    raw_path = json_path.with_name(summary["raw_samples"]["file"])
    problems = []
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != summary["raw_samples"]["sha256"]:
        problems.append("raw samples file hash differs from the summary")
    header, records = read_samples(raw_path)
    header = {k: v for k, v in header.items() if k != "record"}
    if header != summary["header"]:
        problems.append("raw header differs from the summary header")
    if any(r.get("experiment_id") != summary["experiment_id"] for r in records):
        problems.append("a record carries a different experiment_id")
    try:
        again = json.loads(json.dumps(derive(header, records)))
    except BenchmarkError as exc:
        return [*problems, f"raw records are incomplete or inconsistent: {exc}"]
    if again != summary["derived"]:
        problems.append("derived summary (latency, recall, plans or decision) does not recompute")
    return problems


def markdown(summary: dict) -> str:
    d, h = summary["derived"], summary["header"]
    lines = [
        f"# Dense benchmark (generated) `{summary['experiment_id']}`",
        "",
        f"- Harness HEAD `{h['provenance']['harness_head']}`; production commit "
        f"`{h['provenance']['production_commit']}`; "
        f"tree clean: {not h['source']['git_working_tree_dirty']}",
        f"- Model `{h['model']['model_id']}@{h['model']['revision']}`, "
        f"dimension {h['model']['dimension']}",
        f"- Queries: {h['queries']['count']} (sha256 `{h['queries']['sha256']}`)",
        f"- Dataset `{h['catalog']['dataset_id']}` version {h['catalog']['dataset_version']}, "
        f"checksum `{h['catalog']['checksum_sha256']}`, "
        f"record count {h['catalog']['record_count']}, "
        f"transform/taxonomy/rules versions {h['catalog']['transform_version']}/"
        f"{h['catalog']['taxonomy_version']}/{h['catalog']['rules_version']}",
        f"- Embedding text version {h['embedding_text_version']}, corpus digest "
        f"`{h['embedding_corpus_digest']}`",
        "",
        "## Metric definitions",
        "",
        *(f"- `{name}`: {text}" for name, text in METRIC_DEFINITIONS.items()),
        "",
        "## Not applicable in Milestone 4",
        "",
        *(f"- `{name}`: {reason}" for name, reason in h["not_applicable"].items()),
        "",
        "## Production reference (B1 end to end, B2 production SQL)",
        "",
        "| run | suite | metric | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for run, data in d["runs"].items():
        for key, suite in data["suites"].items():
            if suite["label"] != "production":
                continue
            for metric, s in suite["overall"].items():
                lines.append(
                    f"| {run} | {key} | {metric} | {s['p50']} | {s['p95']} | {s['p99']} "
                    f"| {s['max']} |"
                )
    lines += [
        "",
        "## EXPERIMENTAL ANN-compatible SQL (not production SQL; never production latency)",
        "",
        "| run | phase order | suite | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for run, data in d["runs"].items():
        for key, suite in data["suites"].items():
            if suite["label"] != "experimental":
                continue
            s = suite["overall"]["ann_sql_ms"]
            lines.append(
                f"| {run} | {'>'.join(data['phase_order'])} | {key} | {s['p50']} | {s['p95']} "
                f"| {s['p99']} | {s['max']} |"
            )
    lines += ["", "## Decision (predeclared rule)", "", f"**{d['decision']['outcome']}**", ""]
    for name, criterion in d["decision"]["criteria"].items():
        lines.append(f"- {name}: passed={criterion['passed']} ({criterion['rule']})")
    return "\n".join(lines) + "\n"


def parent_main(args: argparse.Namespace) -> int:  # pragma: no cover - runs the benchmark
    from sqlalchemy import create_engine, text

    from ecommerce_search.config import get_settings
    from ecommerce_search.embeddings.fetch import verify_manifest
    from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION

    provenance = check_provenance(ROOT)
    settings = get_settings()
    model = check_model(settings.resolved_models_dir(), settings.embedding_spec(), verify_manifest)
    queries = query_set_identity()
    started = datetime.now(UTC)
    experiment_id = f"dense-m4-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    header = {
        "experiment_id": experiment_id,
        "generated_at_utc": started.isoformat(timespec="seconds"),
        "provenance": {**provenance, "guarded": list(provenance["guarded"])},
        "source": source_fingerprint(ROOT),
        "model": model,
        "catalog": committed_catalog_identity(),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_corpus_digest": embedding_corpus_digest(),
        "not_applicable": NOT_APPLICABLE,
        "metric_definitions": METRIC_DEFINITIONS,
        "queries": queries,
        "environment": environment(settings),
        "protocol": {
            "runs": RUNS,
            "warmup_per_query": args.warmup,
            "samples_per_query": args.samples,
            "k_values": list(K_VALUES),
            "api_top_k": API_TOP_K,
            "phase_schedule": {str(r): list(o) for r, o in PHASE_SCHEDULE.items()},
            "hnsw": {
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef_search": args.hnsw_ef_search,
                "index": INDEX_NAME,
                "forced_setting": "session-local enable_seqscan = off (C2 timing only)",
            },
            "percentile_method": "nearest-rank",
        },
    }
    records: list[dict] = []
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with tempfile.TemporaryDirectory(prefix="dense-bench-") as tmp:
            for run in range(1, RUNS + 1):
                name = assert_scratch_name(new_scratch_name(), settings.postgres_db)
                out = Path(tmp) / f"run{run}.jsonl"
                print(f"run {run}/{RUNS} ({'>'.join(phase_order(run))}) ...", flush=True)
                try:
                    subprocess.run(  # noqa: S603 - fixed argv, sys.executable
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--child",
                            "--run",
                            str(run),
                            "--database",
                            name,
                            "--experiment-id",
                            experiment_id,
                            "--out",
                            str(out),
                            "--warmup",
                            str(args.warmup),
                            "--samples",
                            str(args.samples),
                            "--hnsw-m",
                            str(args.hnsw_m),
                            "--hnsw-ef-construction",
                            str(args.hnsw_ef_construction),
                            "--hnsw-ef-search",
                            str(args.hnsw_ef_search),
                        ],
                        cwd=ROOT,
                        check=True,
                        env={**os.environ, "HF_HUB_OFFLINE": "1"},
                    )
                finally:
                    with admin.connect() as conn:  # belt and braces: the child drops it too
                        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
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
    json_path = write_artifacts(OUTPUT_DIR, header, records, extra)
    problems = verify_artifacts(json_path)
    if problems:
        raise BenchmarkError("artifact self-verification failed: " + "; ".join(problems))
    print(f"wrote {json_path.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--verify", type=Path, help="recompute a summary from its raw records")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-id", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.samples < 1 or args.hnsw_ef_search < max(K_VALUES):
        raise SystemExit("invalid sampling or hnsw.ef_search (must be >= the largest k)")
    if args.verify:
        problems = verify_artifacts(args.verify)
        print("\n".join(problems) or "OK: summary and decision recompute exactly from raw records")
        return 1 if problems else 0
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
