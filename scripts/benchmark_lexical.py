"""V0 lexical-search latency benchmark (Milestone 3). Scratch databases only.

Protocol (fixed by the milestone approval):
  * every run creates a fresh throwaway database, migrates it to head, ingests the committed
    240-product seed, runs ANALYZE, measures, and drops the database. The development database
    is never connected to for data;
  * 12 queries; per query: 1 cold first execution (recorded separately), 20 untimed warm-ups,
    then 200 timed executions, from a single sequential in-process client;
  * 3 independent runs by default (2,400 timed samples per run, 7,200 across three runs).

Per request this records: `lexical_ms` (SQL and row fetch, from the response), `app_total_ms`
(handler processing through result-model construction, from the response), `serialization_ms`
(Pydantic `model_dump_json` of the parsed response, measured separately) and `request_ms` (the
in-process FastAPI TestClient `get` call: routing, validation, handler and framework response
serialization; the interval ends BEFORE `response.json()`, so client JSON decoding is not
included). `request_ms` is NOT network latency: there is no socket and no real server.

Artifacts (only under data/processed/search_benchmark/, git-ignored):
  * `<experiment_id>.json`: summary, environment, source fingerprint, EXPLAIN plans;
  * `<experiment_id>.samples.jsonl`: EVERY measured sample (cold and timed), one JSON object per
    line after a header line. Nothing is dropped or winsorized. A sample is flagged `outlier`
    when `request_ms` or `lexical_ms` exceeds OUTLIER_FACTOR times the median of its own
    (run, query) timed samples. The flag is diagnostic only: flagged samples stay in every
    percentile;
  * `<experiment_id>.md`: a human-readable rendering.
Every summary figure is recomputable exactly from the raw samples (`--verify`).

Percentiles use the nearest-rank method.

Usage:
  uv run python scripts/benchmark_lexical.py [--runs 3] [--warmup 20] [--samples 200]
  uv run python scripts/benchmark_lexical.py --verify data/processed/search_benchmark/<id>.json
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from ecommerce_search.api.app import create_app
from ecommerce_search.api.schemas import SearchResponse
from ecommerce_search.config import get_settings
from ecommerce_search.ingestion.service import ingest_file
from ecommerce_search.search.documents import DOCUMENT_VERSION, FTS_CONFIG
from ecommerce_search.search.lexical import RETRIEVAL_SQL, retrieval_params
from ecommerce_search.search.query import RANK_NORMALIZATION, RANK_WEIGHTS, SEARCH_VERSION

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "search_benchmark"

QUERIES = [
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
]
METRICS = ("lexical_ms", "app_total_ms", "serialization_ms", "request_ms")
OUTLIER_FACTOR = 10  # diagnostic rule: metric > 10 x the median of its (run, query) samples
OUTLIER_METRICS = ("request_ms", "lexical_ms")

# ---- source provenance -----------------------------------------------------------------------
# Only repository source/config/migration files that can change what the benchmark executes.
RELEVANT_PREFIXES = ("src/", "migrations/", "scripts/", "data/seed/")
RELEVANT_FILES = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
        "docker-compose.yml",
        "Dockerfile",
        ".env.example",
        ".python-version",
        ".gitattributes",
    }
)
# Never fingerprinted, even if somehow tracked: secrets, local tooling, outputs, caches.
EXCLUDED = re.compile(
    r"(^|/)(\.env(\..*)?|\.claude|data/processed|data/raw|__pycache__|\.venv|\.pytest_cache|"
    r"\.ruff_cache|\.git)(/|$)"
)


class ProvenanceError(RuntimeError):
    """Source provenance could not be computed completely."""


class BenchmarkError(RuntimeError):
    """A request failed; the benchmark aborts instead of recording it as a latency sample."""


def is_relevant(path: str) -> bool:
    if path == ".env.example":
        return True
    if EXCLUDED.search(path):
        return False
    return path in RELEVANT_FILES or path.startswith(RELEVANT_PREFIXES)


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv
            ["git", *args],  # noqa: S607
            cwd=root,
            input=stdin,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ProvenanceError(
            f"git is unavailable ({type(exc).__name__}); cannot compute source provenance"
        ) from None
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace")[:200]
        raise ProvenanceError(f"`git {' '.join(args[:2])}` failed in {root}: {detail}")
    return done.stdout


def _nul_list(data: bytes) -> list[str]:
    return [item for item in data.decode("utf-8").split("\0") if item]


def source_fingerprint(root: Path) -> dict:
    """Deterministic fingerprint of the relevant repository source (tracked and untracked).

    Entries are `path + Git-normalized blob hash` (`git hash-object` applies the repository's
    attribute and line-ending filters, so equivalent CRLF/LF content hashes the same), sorted
    by path. Ignored files (.env, caches, data/processed, ...) never contribute."""
    base_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    tracked = set(_nul_list(_git(root, "ls-files", "-z", "--cached")))
    untracked = set(_nul_list(_git(root, "ls-files", "-z", "--others", "--exclude-standard")))
    paths = sorted(p for p in tracked | untracked if is_relevant(p))
    for path in paths:
        if "\n" in path:
            raise ProvenanceError("a relevant path contains a newline; cannot fingerprint")
    present = [p for p in paths if (root / p).is_file()]
    hashes: dict[str, str] = {}
    if present:
        out = _git(root, "hash-object", "--stdin-paths", stdin="\n".join(present).encode())
        lines = out.decode().split()
        if len(lines) != len(present):
            raise ProvenanceError("git hash-object returned an unexpected number of hashes")
        hashes = dict(zip(present, lines, strict=True))
    entries = {p: hashes.get(p, "deleted") for p in paths}
    tree = hashlib.sha256()
    for path in sorted(entries):
        tree.update(f"{path}\0{entries[path]}\n".encode())
    pathspec = [*sorted(RELEVANT_FILES), *RELEVANT_PREFIXES]
    diff = _git(root, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-color", "--", *pathspec)
    untracked_files = {p: entries[p] for p in sorted(untracked) if p in entries}
    return {
        "base_git_commit": base_commit,
        "git_working_tree_dirty": bool(diff) or bool(untracked_files),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked_files,
        "source_files": entries,
        "source_file_count": len(entries),
        "source_tree_sha256": tree.hexdigest(),
        "fingerprint_scope": {
            "prefixes": list(RELEVANT_PREFIXES),
            "files": sorted(RELEVANT_FILES),
            "excluded": "secrets, .claude, data/processed, caches, anything git ignores",
        },
    }


def new_experiment_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"lexical-v0-{stamp}-{uuid.uuid4().hex[:8]}"


# ---- measurement ------------------------------------------------------------------------------


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": round(percentile(ordered, 50), 3),
        "p95": round(percentile(ordered, 95), 3),
        "p99": round(percentile(ordered, 99), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def measure_request(client, query: str) -> dict:
    """One request. Any non-200 response aborts the benchmark (never recorded as a latency)."""
    wall = datetime.now(UTC).isoformat(timespec="microseconds")
    started = time.perf_counter()
    response = client.get("/search", params={"q": query, "top_k": 10})
    request_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise BenchmarkError(f"unexpected status {response.status_code} for {query!r}")
    payload = response.json()
    model = SearchResponse.model_validate(payload)
    started = time.perf_counter()
    model.model_dump_json()
    serialization_ms = (time.perf_counter() - started) * 1000
    return {
        "timestamp_utc": wall,
        "status_code": response.status_code,
        "result_count": payload["result_count"],
        "lexical_ms": payload["latency_ms"]["lexical_ms"],
        "app_total_ms": payload["latency_ms"]["total_ms"],
        "serialization_ms": serialization_ms,
        "request_ms": request_ms,
    }


def flag_outliers(timed: list[dict]) -> None:
    """Diagnostic flag on one (run, query) group of timed samples. Never excludes anything."""
    medians = {m: statistics.median(s[m] for s in timed) for m in OUTLIER_METRICS}
    for sample in timed:
        hit = [m for m in OUTLIER_METRICS if sample[m] > OUTLIER_FACTOR * medians[m]]
        sample["outlier"] = bool(hit)
        sample["outlier_metrics"] = hit


def collect_run(client, *, experiment_id: str, run: int, queries, warmup: int, samples: int):
    """All recorded samples of one run: per query one cold sample, then `samples` timed ones.

    Warm-ups are executed (and must return 200) but are not recorded."""
    records: list[dict] = []
    sequence = 0

    def record(measured: dict, query_index: int, query: str, phase: str, index: int) -> dict:
        nonlocal sequence
        entry = {
            "record": "sample",
            "experiment_id": experiment_id,
            "run": run,
            "query_index": query_index,
            "query": query,
            "phase": phase,
            "sample_index": index,
            "sequence": sequence,
            **measured,
            "outlier": False,
            "outlier_metrics": [],
        }
        sequence += 1
        records.append(entry)
        return entry

    for query_index, query in enumerate(queries):
        record(measure_request(client, query), query_index, query, "cold", 0)
        for _ in range(warmup):
            measure_request(client, query)
        timed = [
            record(measure_request(client, query), query_index, query, "timed", i)
            for i in range(samples)
        ]
        if len({t["result_count"] for t in timed}) != 1:
            raise BenchmarkError(f"result_count changed during the run for {query!r}")
        flag_outliers(timed)
    return records


def summarize_records(records: list[dict]) -> dict:
    """Per-run summary computed only from raw records (cold samples are reported separately)."""
    timed = [r for r in records if r["phase"] == "timed"]
    per_query = {}
    for query in dict.fromkeys(r["query"] for r in timed):
        mine = [r for r in timed if r["query"] == query]
        cold = next(r for r in records if r["phase"] == "cold" and r["query"] == query)
        per_query[query] = {
            "result_count": mine[0]["result_count"],
            "cold_first_run_ms": {m: round(cold[m], 3) for m in METRICS},
            "timed": {m: summarize([r[m] for r in mine]) for m in METRICS},
        }
    return {
        "timed_samples": len(timed),
        "cold_samples": len(records) - len(timed),
        "overall": {m: summarize([r[m] for r in timed]) for m in METRICS},
        "per_query": per_query,
        "flagged_outliers": [
            {
                "query": r["query"],
                "sample_index": r["sample_index"],
                "sequence": r["sequence"],
                "timestamp_utc": r["timestamp_utc"],
                "metrics": r["outlier_metrics"],
                **{m: r[m] for m in METRICS},
            }
            for r in timed
            if r["outlier"]
        ],
    }


def pooled_summary(records: list[dict]) -> dict:
    timed = [r for r in records if r["phase"] == "timed"]
    return {
        "label": "pooled timed samples from all runs; per-run percentiles are the primary result",
        "n_per_metric": len(timed),
        "overall": {m: summarize([r[m] for r in timed]) for m in METRICS},
    }


def recompute(records: list[dict]) -> dict:
    """Summaries recomputed from raw samples only, keeping run boundaries."""
    runs = sorted({r["run"] for r in records})
    return {
        "runs": {run: summarize_records([r for r in records if r["run"] == run]) for run in runs},
        "pooled": pooled_summary(records),
    }


# ---- artifacts --------------------------------------------------------------------------------


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_samples(path: Path, header: dict, records: list[dict]) -> str:
    ordered = sorted(records, key=lambda r: (r["run"], r["sequence"]))
    body = "\n".join([dumps({"record": "header", **header}), *(dumps(r) for r in ordered)]) + "\n"
    path.write_bytes(body.encode("utf-8"))  # exact bytes: LF endings on every platform
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_samples(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("record") != "header":
        raise ValueError("raw samples file has no header line")
    return lines[0], lines[1:]


def verify_artifacts(summary_path: Path) -> list[str]:
    """Problems found when recomputing a summary artifact from its raw samples (empty = OK)."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_path = summary_path.with_name(summary["raw_samples"]["file"])
    header, records = read_samples(raw_path)
    problems = []
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != summary["raw_samples"]["sha256"]:
        problems.append("raw samples file hash differs from the summary")
    if header["experiment_id"] != summary["experiment_id"]:
        problems.append("experiment_id differs between header and summary")
    if header["source_tree_sha256"] != summary["source"]["source_tree_sha256"]:
        problems.append("source_tree_sha256 differs between header and summary")
    if any(r["experiment_id"] != summary["experiment_id"] for r in records):
        problems.append("a sample carries a different experiment_id")
    again = recompute(records)
    for run in summary["runs"]:
        mine = {k: run[k] for k in ("timed_samples", "cold_samples", "overall", "per_query")}
        mine["flagged_outliers"] = run["flagged_outliers"]
        if again["runs"][run["run"]] != mine:
            problems.append(f"run {run['run']} summary does not match its raw samples")
    if again["pooled"] != summary["pooled_across_runs"]:
        problems.append("pooled summary does not match the raw samples")
    return problems


def assemble_report(
    *, experiment_id, source, environment, protocol, run_results, records, raw_name, raw_sha
) -> dict:
    runs = []
    for result in run_results:
        mine = [r for r in records if r["run"] == result["run"]]
        runs.append(
            {
                "run": result["run"],
                "catalog": result["catalog"],
                **summarize_records(mine),
                "explain": result["explain"],
            }
        )
    return {
        "experiment_id": experiment_id,
        "generated_at_utc": environment["generated_at_utc"],
        "source": source,
        "environment": environment,
        "protocol": protocol,
        "raw_samples": {
            "file": raw_name,
            "sha256": raw_sha,
            "timed_samples": sum(1 for r in records if r["phase"] == "timed"),
            "cold_samples": sum(1 for r in records if r["phase"] == "cold"),
            "outlier_rule": (
                f"diagnostic only: {'/'.join(OUTLIER_METRICS)} > {OUTLIER_FACTOR} x the median of "
                "the same (run, query) timed samples; flagged samples stay in every percentile"
            ),
        },
        "runs": runs,
        "pooled_across_runs": pooled_summary(records),
    }


def markdown(report: dict) -> str:
    src, proto = report["source"], report["protocol"]
    lines = [
        f"# V0 lexical benchmark (generated) `{report['experiment_id']}`",
        "",
        f"Generated {report['generated_at_utc']}; base commit `{src['base_git_commit']}`, "
        f"working tree dirty: {src['git_working_tree_dirty']}; "
        f"`source_tree_sha256` {src['source_tree_sha256']}.",
        "",
        f"Protocol: {proto['runs']} runs x {len(proto['queries'])} queries x "
        f"{proto['samples_per_query']} timed samples ({proto['timed_samples_per_run']} per run, "
        f"{proto['timed_samples_total']} total), {proto['warmup_per_query']} warm-ups per query, "
        "single sequential in-process client.",
        "",
    ]
    for run in report["runs"]:
        lines += [f"## Run {run['run']} overall (n={run['timed_samples']} per metric, ms)", ""]
        lines += [
            "| metric | p50 | p95 | p99 | min | max | mean |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric in METRICS:
            s = run["overall"][metric]
            cells = [s[k] for k in ("p50", "p95", "p99", "min", "max", "mean")]
            lines.append(f"| {metric} | " + " | ".join(str(c) for c in cells) + " |")
        lines += ["", f"### Run {run['run']} per query (n=200 each)", ""]
        lines += [
            "| query | results | cold request_ms | lexical p50 | lexical p95 | lexical p99 | "
            "request p50 | request p95 | request p99 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for query, data in run["per_query"].items():
            lx, rq = data["timed"]["lexical_ms"], data["timed"]["request_ms"]
            cells = [
                data["result_count"],
                data["cold_first_run_ms"]["request_ms"],
                *(lx[k] for k in ("p50", "p95", "p99")),
                *(rq[k] for k in ("p50", "p95", "p99")),
            ]
            lines.append(f"| `{query}` | " + " | ".join(str(c) for c in cells) + " |")
        lines += [
            "",
            f"Flagged diagnostic outliers in run {run['run']}: {len(run['flagged_outliers'])}",
        ]
        for o in run["flagged_outliers"]:
            lines.append(
                f"* `{o['query']}` sample {o['sample_index']} at {o['timestamp_utc']}: "
                f"request_ms {o['request_ms']:.3f}, lexical_ms {o['lexical_ms']}, "
                f"app_total_ms {o['app_total_ms']}, flagged on {', '.join(o['metrics'])}"
            )
        lines += [""]
    pooled = report["pooled_across_runs"]
    lines += [f"## Pooled across runs (n={pooled['n_per_metric']} per metric; ms)", ""]
    lines += ["| metric | p50 | p95 | p99 |", "|---|---|---|---|"]
    for metric in METRICS:
        s = pooled["overall"][metric]
        lines.append(f"| {metric} | {s['p50']} | {s['p95']} | {s['p99']} |")
    lines += ["", "## EXPLAIN (run 1, ANALYZE, top_k=10)", ""]
    for query, entry in report["runs"][0]["explain"].items():
        nodes = " > ".join(entry["nodes"])
        lines.append(f"* `{query}`: {nodes} (execution {entry['execution_ms']} ms)")
    return "\n".join(lines) + "\n"


# ---- database runs ----------------------------------------------------------------------------


def explain(engine, query: str) -> dict:
    sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + str(RETRIEVAL_SQL)
    with engine.connect() as conn:
        raw = conn.execute(text(sql), retrieval_params(" ".join(query.split()), 10)).scalar_one()
    plan = raw if isinstance(raw, list) else json.loads(raw)
    nodes: list[str] = []

    def walk(node: dict) -> None:
        label = node["Node Type"]
        if "Index Name" in node:
            label += f" [{node['Index Name']}]"
        if "Relation Name" in node:
            label += f" on {node['Relation Name']}"
        nodes.append(label)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan[0]["Plan"])
    return {
        "nodes": nodes,
        "planning_ms": plan[0].get("Planning Time"),
        "execution_ms": plan[0].get("Execution Time"),
        "plan": plan,
    }


def run_once(settings, experiment_id: str, run_number: int, warmup: int, samples: int) -> dict:
    name = f"ecommerce_search_bench_{uuid.uuid4().hex[:12]}"
    admin = create_engine(settings.database_url(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = None
    try:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database"] = name
        command.upgrade(config, "head")
        engine = create_engine(settings.database_url(database=name))
        outcome = ingest_file(engine, SEED, PROVENANCE)
        if outcome.result is None or outcome.result.inserted != 240:
            raise BenchmarkError("the committed seed did not ingest as 240 products")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("ANALYZE"))  # planner statistics for the freshly loaded tables
        with engine.connect() as conn:
            counts = {
                "products": conn.execute(text("SELECT count(*) FROM products")).scalar_one(),
                "documents": conn.execute(
                    text("SELECT count(*) FROM product_search_documents")
                ).scalar_one(),
            }
        run_settings = settings.model_copy(update={"postgres_db": name})
        with TestClient(create_app(run_settings)) as client:
            records = collect_run(
                client,
                experiment_id=experiment_id,
                run=run_number,
                queries=QUERIES,
                warmup=warmup,
                samples=samples,
            )
        plans = {query: explain(engine, query) for query in QUERIES}
        return {"run": run_number, "catalog": counts, "records": records, "explain": plans}
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def environment(settings) -> dict:
    admin = create_engine(settings.database_url(database="postgres"))
    with admin.connect() as conn:
        server = {
            "version": conn.execute(text("SELECT version()")).scalar_one(),
            **{
                name: conn.execute(text(f"SHOW {name}")).scalar_one()
                for name in (
                    "shared_buffers",
                    "work_mem",
                    "effective_cache_size",
                    "max_parallel_workers_per_gather",
                    "jit",
                    "default_text_search_config",
                    "fsync",
                    "synchronous_commit",
                )
            },
        }
    admin.dispose()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "postgres": server,
        "postgres_location": "Docker container (compose service `db`) on the same host",
        "dataset": {
            "file": str(SEED.relative_to(ROOT)),
            "checksum_sha256": hashlib.sha256(SEED.read_bytes()).hexdigest(),
            "provenance": json.loads(PROVENANCE.read_text(encoding="utf-8")),
        },
        "search": {
            "search_version": SEARCH_VERSION,
            "document_version": DOCUMENT_VERSION,
            "fts_config": FTS_CONFIG,
            "query_mode": "plainto_tsquery (strict AND)",
            "rank_function": "ts_rank",
            "rank_weights": RANK_WEIGHTS,
            "rank_normalization": RANK_NORMALIZATION,
            "top_k": 10,
            "settings": {
                "search_lexical_k": settings.search_lexical_k,
                "search_default_top_k": settings.search_default_top_k,
                "search_max_query_length": settings.search_max_query_length,
                "db_pool_size": settings.db_pool_size,
                "db_max_overflow": settings.db_max_overflow,
                "db_pool_timeout_seconds": settings.db_pool_timeout_seconds,
                "db_statement_timeout_ms": settings.db_statement_timeout_ms,
            },
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--verify", type=Path, help="recompute a summary artifact from raw samples")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify_artifacts(args.verify)
        print("\n".join(problems) or "OK: summary recomputes exactly from the raw samples")
        return 1 if problems else 0

    source = source_fingerprint(ROOT)  # fails clearly if git is unavailable
    experiment_id = new_experiment_id()
    settings = get_settings()
    env = environment(settings)
    run_results = []
    for number in range(1, args.runs + 1):
        print(f"run {number}/{args.runs} ...", flush=True)
        run_results.append(run_once(settings, experiment_id, number, args.warmup, args.samples))
    records = [r for result in run_results for r in result["records"]]
    timed_per_run = len(QUERIES) * args.samples
    protocol = {
        "runs": args.runs,
        "queries": QUERIES,
        "warmup_per_query": args.warmup,
        "samples_per_query": args.samples,
        "timed_samples_per_run": timed_per_run,
        "timed_samples_total": timed_per_run * args.runs,
        "percentile_method": "nearest-rank",
        "client": "FastAPI TestClient, in-process, single sequential client",
        "notes": [
            "request_ms is in-process, ends before response.json(), and is not network latency",
            "serialization_ms is model_dump_json of the parsed response, measured separately",
            "ANALYZE runs after ingestion; every run uses a fresh scratch database",
            "cold samples are recorded separately from timed samples",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"{experiment_id}.samples.jsonl"
    header = {
        "experiment_id": experiment_id,
        "base_git_commit": source["base_git_commit"],
        "git_working_tree_dirty": source["git_working_tree_dirty"],
        "source_tree_sha256": source["source_tree_sha256"],
        "generated_at_utc": env["generated_at_utc"],
        "protocol": protocol,
    }
    raw_sha = write_samples(raw_path, header, records)
    report = assemble_report(
        experiment_id=experiment_id,
        source=source,
        environment=env,
        protocol=protocol,
        run_results=run_results,
        records=records,
        raw_name=raw_path.name,
        raw_sha=raw_sha,
    )
    json_path, md_path = OUTPUT_DIR / f"{experiment_id}.json", OUTPUT_DIR / f"{experiment_id}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    problems = verify_artifacts(json_path)
    if problems:
        raise BenchmarkError("artifact self-verification failed: " + "; ".join(problems))
    print(f"wrote {json_path.relative_to(ROOT)}, {raw_path.name} and {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
