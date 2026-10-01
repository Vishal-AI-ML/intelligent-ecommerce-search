"""Embedding-model selection experiment (Milestone 4). No database, no network.

Compares the shortlisted local sentence-transformers models on the committed 240-product seed,
using the production provider (`SentenceTransformerEmbedder`) and the production embedding-text
builder. Snapshots must already have been fetched (`python -m ecommerce_search.search
model-fetch`) into the git-ignored `models/` directory. Every model run happens in a fresh
subprocess with outbound sockets disabled, so memory figures are isolated and any accidental
network access fails loudly.

Measured per model and run:
  import_ms (torch + sentence-transformers), load_ms, RSS before/after load and peak, snapshot
  bytes, corpus token counts (max, p95, truncated), corpus encode time at a fixed batch size
  (repeated) and throughput, re-encode agreement between batch sizes, one cold first query,
  then for each query `--warmup` untimed and `--samples` timed query encodings.

Predeclared gates (all must pass): permissive declared licence (still to be verified by a human
against the official model page), immutable revision and intact file manifest, offline load
without remote code, zero truncated corpus documents, valid vectors (enforced by the provider)
and re-encode cosine >= 0.9999.

Predeclared ranking among passing models: lowest pooled warm query-encode p50; a difference
smaller than the run-to-run spread is a tie and falls through to peak RSS, then snapshot bytes.

Retrieval behaviour is reported qualitatively only (top-10 per probe, cross-model overlap@10).
There is no Golden Dataset, so no relevance metric is computed and no quality claim is made.

Artifacts (git-ignored): data/processed/embedding_selection/<id>.{json,md,samples.jsonl}.

Usage:
  uv run python scripts/embedding_model_selection.py \
      --candidate sentence-transformers/all-MiniLM-L6-v2@<sha> \
      --candidate BAAI/bge-small-en-v1.5@<sha> --candidate intfloat/e5-small-v2@<sha>
  uv run python scripts/embedding_model_selection.py --verify <artifact .json>
  (artifacts are written under data/processed/embedding_selection/)
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_lexical import (  # noqa: E402 - scripts/ is not a package
    dumps,
    read_samples,
    source_fingerprint,
    summarize,
)

from ecommerce_search.embeddings.fetch import declared_license, verify_manifest  # noqa: E402
from ecommerce_search.embeddings.spec import (  # noqa: E402
    EmbeddingModelSpec,
    check_model_id,
    check_revision,
    snapshot_dir,
)
from ecommerce_search.embeddings.text import EMBEDDING_TEXT_VERSION  # noqa: E402

SEED = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed" / "embedding_selection"

# Model-card facts used by the experiment. The prefixes come from the official model cards and
# must be re-checked there; dimension and max length are READ FROM THE SNAPSHOT, never assumed.
CANDIDATE_PREFIXES: dict[str, dict[str, str]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {"query_prefix": "", "document_prefix": ""},
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
    "intfloat/e5-small-v2": {"query_prefix": "query: ", "document_prefix": "passage: "},
}
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit"})

QUERIES = [
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
]
CORPUS_REPEATS = 3
REENCODE_MIN_COSINE = 0.9999
TOP_K = 10


# ---- pure helpers (unit-tested) -----------------------------------------------------------------


def parse_candidate(value: str) -> tuple[str, str]:
    model_id, sep, revision = value.partition("@")
    if not sep:
        raise argparse.ArgumentTypeError("candidate must be MODEL_ID@REVISION")
    return check_model_id(model_id), check_revision(revision)


def snapshot_spec(models_dir: Path, model_id: str, revision: str) -> EmbeddingModelSpec:
    """Spec built from the snapshot's own configuration files (dimension, max length)."""
    snap = snapshot_dir(models_dir, model_id, revision)
    st_config = json.loads((snap / "sentence_bert_config.json").read_text(encoding="utf-8"))
    pooling = json.loads((snap / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    return EmbeddingModelSpec(
        model_id=model_id,
        revision=revision,
        dimension=int(pooling["word_embedding_dimension"]),
        max_seq_length=int(st_config["max_seq_length"]),
        normalize=True,  # always normalized at encode time (cosine == dot product)
        **CANDIDATE_PREFIXES[model_id],
    )


def cosine(a: list[float], b: list[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    return dot / (na * nb)


def top_k(query: list[float], docs: list[list[float]], ids: list[str], k: int) -> list[dict]:
    """Exact cosine ranking; ties broken by product_id ascending (as the service will)."""
    scored = sorted(((-cosine(query, d), pid) for d, pid in zip(docs, ids, strict=True)))
    return [{"product_id": pid, "score": round(-neg, 6)} for neg, pid in scored[:k]]


def overlap_at_k(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return round(len(set(a) & set(b)) / max(len(a), len(b)), 3)


def decide(models: dict[str, dict]) -> dict:
    """Apply the predeclared ranking rule to models that passed every gate.

    Criteria in order: per-run warm query-encode p50, per-run peak RSS, snapshot bytes. For the
    per-run criteria, the best model wins only if its pooled value beats the runner-up by more
    than the larger run-to-run spread of the two; otherwise it is a tie and the next criterion
    decides among the tied models."""
    passing = sorted(m for m, info in models.items() if info["gates"]["all_passed"])
    trace: list[str] = []
    if not passing:
        return {"winner": None, "passing": [], "trace": ["no model passed every gate"]}
    pool = passing
    for criterion in ("query_encode_p50_ms", "peak_rss_mb"):
        if len(pool) == 1:
            break
        values = {m: models[m]["ranking"][criterion] for m in pool}
        ordered = sorted(pool, key=lambda m: (values[m]["pooled"], m))
        best, runner_up = ordered[0], ordered[1]
        spread = max(values[best]["spread"], values[runner_up]["spread"])
        gap = values[runner_up]["pooled"] - values[best]["pooled"]
        if gap > spread:
            trace.append(
                f"{criterion}: {best} wins ({values[best]['pooled']} vs "
                f"{values[runner_up]['pooled']}, gap {round(gap, 3)} > spread {round(spread, 3)})"
            )
            pool = [best]
        else:
            tied = [m for m in ordered if values[m]["pooled"] - values[best]["pooled"] <= spread]
            trace.append(
                f"{criterion}: tie among {tied} (gap {round(gap, 3)} <= spread {round(spread, 3)})"
            )
            pool = tied
    if len(pool) > 1:
        pool = [min(pool, key=lambda m: (models[m]["snapshot_bytes"], m))]
        trace.append(f"snapshot_bytes: {pool[0]} is smallest")
    return {"winner": pool[0], "passing": passing, "trace": trace}


def per_run_criterion(values_by_run: dict[int, float]) -> dict:
    values = list(values_by_run.values())
    return {
        "by_run": {str(r): round(v, 3) for r, v in sorted(values_by_run.items())},
        "pooled": round(sorted(values)[len(values) // 2], 3),  # median of per-run values
        "spread": round(max(values) - min(values), 3),
    }


# ---- child: one model, one run ------------------------------------------------------------------


def _block_network() -> None:
    import socket

    def refuse(*args, **kwargs):
        raise OSError("network access is disabled in the model-selection experiment")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def _corpus() -> tuple[list[str], list[str]]:
    from ecommerce_search.embeddings.text import build_embedding_text
    from ecommerce_search.ingestion.loader import load_catalog

    lines = sorted(load_catalog(SEED).lines, key=lambda line: line.record.product_id)
    return [ln.record.product_id for ln in lines], [build_embedding_text(ln.record) for ln in lines]


def _peak_rss_mb(process) -> float:
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource  # POSIX

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return peak / 2**20


def child_main(args: argparse.Namespace) -> int:
    _block_network()
    import psutil

    process = psutil.Process()
    model_id, revision = args.candidate
    spec = snapshot_spec(args.models_dir, model_id, revision)
    ids, texts = _corpus()
    rss_start = process.memory_info().rss / 2**20

    started = time.perf_counter()
    import sentence_transformers
    import torch

    import_ms = (time.perf_counter() - started) * 1000
    from ecommerce_search.embeddings.sentence_transformers_provider import (
        SentenceTransformerEmbedder,
    )

    embedder = SentenceTransformerEmbedder(spec, args.models_dir, batch_size=args.batch_size)
    load_ms = embedder.load()
    rss_loaded = process.memory_info().rss / 2**20

    tokens = embedder.count_tokens(texts, "document")
    corpus_ms: list[float] = []
    vectors: list[list[float]] = []
    for _ in range(CORPUS_REPEATS):
        t0 = time.perf_counter()
        vectors = embedder.embed_documents(texts)
        corpus_ms.append((time.perf_counter() - t0) * 1000)
    single = SentenceTransformerEmbedder(spec, args.models_dir, batch_size=1)
    single._model = embedder._model  # same loaded weights, different batch composition
    reencoded = single.embed_documents(texts)
    min_reencode = min(cosine(a, b) for a, b in zip(vectors, reencoded, strict=True))

    samples: list[dict] = []
    sequence = 0

    def record(kind: str, q_index: int, s_index: int, ms: float) -> None:
        nonlocal sequence
        samples.append(
            {
                "record": "sample",
                "model_id": model_id,
                "run": args.run,
                "kind": kind,
                "query": QUERIES[q_index],
                "query_index": q_index,
                "sample_index": s_index,
                "sequence": sequence,
                "utc": datetime.now(UTC).isoformat(timespec="microseconds"),
                "query_encode_ms": round(ms, 4),
            }
        )
        sequence += 1

    t0 = time.perf_counter()
    first = embedder.embed_query(QUERIES[0])
    record("cold", 0, 0, (time.perf_counter() - t0) * 1000)
    probes = {}
    query_vectors = {QUERIES[0]: first}
    for q_index, query in enumerate(QUERIES):
        for _ in range(args.warmup):
            embedder.embed_query(query)
        for s_index in range(args.samples):
            t0 = time.perf_counter()
            query_vectors[query] = embedder.embed_query(query)
            record("timed", q_index, s_index, (time.perf_counter() - t0) * 1000)
    for query in QUERIES:
        probes[query] = top_k(query_vectors[query], vectors, ids, TOP_K)

    modules = json.loads(
        (snapshot_dir(args.models_dir, model_id, revision) / "modules.json").read_text("utf-8")
    )
    result = {
        "model_id": model_id,
        "revision": revision,
        "run": args.run,
        "spec": spec.config(),
        "config_sha256": spec.config_sha256(),
        "snapshot_has_normalize_module": any(
            m.get("type", "").endswith("Normalize") for m in modules
        ),
        "versions": {
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "torch_num_threads": torch.get_num_threads(),
        "import_ms": round(import_ms, 3),
        "load_ms": load_ms,
        "rss_mb": {
            "start": round(rss_start, 1),
            "after_load": round(rss_loaded, 1),
            "peak": round(_peak_rss_mb(process), 1),
        },
        "tokens": {
            "max": max(tokens),
            "p95": sorted(tokens)[math.ceil(0.95 * len(tokens)) - 1],
            "truncated": sum(1 for t in tokens if t > spec.max_seq_length),
            "max_seq_length": spec.max_seq_length,
        },
        "corpus": {
            "documents": len(texts),
            "batch_size": args.batch_size,
            "encode_ms": [round(v, 3) for v in corpus_ms],
            "docs_per_second": [round(len(texts) / (v / 1000), 2) for v in corpus_ms],
        },
        "reencode_min_cosine": round(min_reencode, 8),
        "probes": probes if args.run == 1 else None,
    }
    Path(args.out).write_text(dumps({"result": result, "samples": samples}), encoding="utf-8")
    return 0


# ---- parent ------------------------------------------------------------------------------------


def environment() -> dict:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "seed": {
            "file": str(SEED.relative_to(ROOT).as_posix()),
            "checksum_sha256": hashlib.sha256(SEED.read_bytes()).hexdigest(),
        },
    }


def gates(model_id: str, revision: str, runs: list[dict], models_dir: Path) -> dict:
    snap = snapshot_dir(models_dir, model_id, revision)
    license_value = declared_license(snap / "README.md")
    manifest_problems = verify_manifest(models_dir, model_id, revision)
    checks = {
        "declared_license": license_value,
        "license_declared_permissive": (license_value or "").lower() in PERMISSIVE_LICENSES,
        "license_human_verified": False,  # set only by a person, in ADR-006
        "immutable_revision_and_manifest_intact": not manifest_problems,
        "manifest_problems": manifest_problems,
        "offline_load_without_remote_code": all(r["load_ms"] is not None for r in runs),
        "zero_truncation": all(r["tokens"]["truncated"] == 0 for r in runs),
        "valid_vectors": True,  # the provider raises on invalid output, which aborts the run
        "reencode_agreement": all(r["reencode_min_cosine"] >= REENCODE_MIN_COSINE for r in runs),
    }
    checks["all_passed"] = all(
        checks[k]
        for k in (
            "license_declared_permissive",
            "immutable_revision_and_manifest_intact",
            "offline_load_without_remote_code",
            "zero_truncation",
            "valid_vectors",
            "reencode_agreement",
        )
    )
    return checks


def summarize_models(results: list[dict], samples: list[dict], models_dir: Path) -> dict:
    out: dict[str, dict] = {}
    for model_id in sorted({r["model_id"] for r in results}):
        runs = sorted((r for r in results if r["model_id"] == model_id), key=lambda r: r["run"])
        timed = [s for s in samples if s["model_id"] == model_id and s["kind"] == "timed"]
        p50_by_run = {
            r["run"]: summarize([s["query_encode_ms"] for s in timed if s["run"] == r["run"]])[
                "p50"
            ]
            for r in runs
        }
        snap = snapshot_dir(models_dir, model_id, runs[0]["revision"])
        out[model_id] = {
            "revision": runs[0]["revision"],
            "spec": runs[0]["spec"],
            "config_sha256": runs[0]["config_sha256"],
            "versions": runs[0]["versions"],
            "torch_num_threads": runs[0]["torch_num_threads"],
            "snapshot_has_normalize_module": runs[0]["snapshot_has_normalize_module"],
            "snapshot_bytes": sum(p.stat().st_size for p in snap.rglob("*") if p.is_file()),
            "per_run": [
                {
                    "run": r["run"],
                    "import_ms": r["import_ms"],
                    "load_ms": r["load_ms"],
                    "rss_mb": r["rss_mb"],
                    "corpus": r["corpus"],
                    "tokens": r["tokens"],
                    "reencode_min_cosine": r["reencode_min_cosine"],
                    "query_encode_ms": summarize(
                        [s["query_encode_ms"] for s in timed if s["run"] == r["run"]]
                    ),
                    "cold_first_query_ms": next(
                        s["query_encode_ms"]
                        for s in samples
                        if s["model_id"] == model_id
                        and s["run"] == r["run"]
                        and s["kind"] == "cold"
                    ),
                }
                for r in runs
            ],
            "query_encode_ms_pooled": summarize([s["query_encode_ms"] for s in timed]),
            "query_encode_ms_by_query": {
                q: summarize([s["query_encode_ms"] for s in timed if s["query"] == q])
                for q in QUERIES
            },
            "ranking": {
                "query_encode_p50_ms": per_run_criterion(p50_by_run),
                "peak_rss_mb": per_run_criterion({r["run"]: r["rss_mb"]["peak"] for r in runs}),
            },
            "gates": gates(model_id, runs[0]["revision"], runs, models_dir),
            "probes": next(r["probes"] for r in runs if r["probes"] is not None),
        }
    return out


def probe_overlap(models: dict[str, dict]) -> dict:
    names = sorted(models)
    table: dict[str, dict] = {}
    for query in QUERIES:
        table[query] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ids_a = [h["product_id"] for h in models[a]["probes"][query]]
                ids_b = [h["product_id"] for h in models[b]["probes"][query]]
                table[query][f"{a} | {b}"] = overlap_at_k(ids_a, ids_b)
    return table


def markdown(report: dict, titles: dict[str, str]) -> str:
    lines = [
        f"# Embedding model selection: {report['experiment_id']}",
        "",
        "> Qualitative probes only. No Golden Dataset exists (Milestone 9); nothing here is a "
        "relevance or quality metric. Synthetic, templated 240-product catalog.",
        "",
        f"- Source: base commit `{report['source']['base_git_commit']}`, dirty "
        f"`{report['source']['git_working_tree_dirty']}`, source tree sha256 "
        f"`{report['source']['source_tree_sha256']}`",
        f"- Protocol: {report['protocol']}",
        "",
        "## Gates",
        "",
        "| Model | Revision | Licence (declared) | Manifest | Offline load | Truncated | "
        "Re-encode min cos | All passed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        g = m["gates"]
        lines.append(
            f"| {name} | `{m['revision']}` | {g['declared_license']} | "
            f"{g['immutable_revision_and_manifest_intact']} | "
            f"{g['offline_load_without_remote_code']} | "
            f"{max(r['tokens']['truncated'] for r in m['per_run'])} | "
            f"{min(r['reencode_min_cosine'] for r in m['per_run'])} | {g['all_passed']} |"
        )
    lines += [
        "",
        "## Measurements (per run)",
        "",
        "| Model | Run | import ms | load ms | corpus encode ms (3x) | docs/s "
        "| RSS start/load/peak MB | query p50 / p95 / p99 / max ms | cold first query ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        for r in m["per_run"]:
            q = r["query_encode_ms"]
            lines.append(
                f"| {name} | {r['run']} | {r['import_ms']} | {r['load_ms']} | "
                f"{r['corpus']['encode_ms']} | {r['corpus']['docs_per_second']} | "
                f"{r['rss_mb']['start']}/{r['rss_mb']['after_load']}/{r['rss_mb']['peak']} | "
                f"{q['p50']} / {q['p95']} / {q['p99']} / {q['max']} | {r['cold_first_query_ms']} |"
            )
    lines += ["", "## Snapshot", ""]
    for name, m in report["models"].items():
        spec = m["spec"]
        lines.append(
            f"- {name}: dimension {spec['dimension']}, max_seq_length {spec['max_seq_length']}, "
            f"query prefix {spec['query_prefix']!r}, document prefix {spec['document_prefix']!r}, "
            f"snapshot bytes {m['snapshot_bytes']}, normalize module in snapshot "
            f"{m['snapshot_has_normalize_module']}, versions {m['versions']}, "
            f"torch threads {m['torch_num_threads']}"
        )
    decision = report["decision"]
    lines += ["", "## Predeclared rule outcome", "", f"- Passing: {decision['passing']}"]
    lines += [f"- {step}" for step in decision["trace"]]
    lines += [f"- **Rule result: {decision['winner']}** (requires human sign-off)", ""]
    lines += ["## Qualitative probes (top 5 of 10, exact cosine)", ""]
    for query in QUERIES:
        lines += [f"### `{query}`", ""]
        for name, m in report["models"].items():
            hits = ", ".join(
                f"{h['product_id']} {titles.get(h['product_id'], '')[:40]!r} ({h['score']})"
                for h in m["probes"][query][:5]
            )
            lines.append(f"- {name}: {hits}")
        lines.append(f"- overlap@10: {report['probe_overlap_at_10'][query]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> int:
    for model_id, revision in args.candidate:
        if model_id not in CANDIDATE_PREFIXES:
            print(f"error: {model_id} is not on the approved shortlist", file=sys.stderr)
            return 2
        if not snapshot_dir(args.models_dir, model_id, revision).is_dir():
            print(f"error: snapshot for {model_id}@{revision} is missing; run model-fetch first")
            return 2
    experiment_id = (
        f"embed-select-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    source = source_fingerprint(ROOT)
    results: list[dict] = []
    samples: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="embed-select-") as tmp:
        for run in range(1, args.runs + 1):
            for model_id, revision in args.candidate:
                out = Path(tmp) / f"{run}-{model_id.replace('/', '--')}.json"
                print(f"run {run}: {model_id}", flush=True)
                subprocess.run(  # noqa: S603 - fixed argv, sys.executable
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--candidate",
                        f"{model_id}@{revision}",
                        "--run",
                        str(run),
                        "--warmup",
                        str(args.warmup),
                        "--samples",
                        str(args.samples),
                        "--batch-size",
                        str(args.batch_size),
                        "--models-dir",
                        str(args.models_dir),
                        "--out",
                        str(out),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                data = json.loads(out.read_text(encoding="utf-8"))
                results.append(data["result"])
                samples.extend(data["samples"])
    models = summarize_models(results, samples, args.models_dir)
    report = {
        "experiment_id": experiment_id,
        "kind": "embedding_model_selection",
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "protocol": (
            f"{args.runs} runs x {len(args.candidate)} models, fresh subprocess each, sockets "
            f"blocked; corpus 240 docs batch {args.batch_size} x{CORPUS_REPEATS}; "
            f"{len(QUERIES)} queries, 1 cold + {args.warmup} warm-up + {args.samples} timed "
            "query encodings each; nearest-rank percentiles"
        ),
        "queries": QUERIES,
        "environment": environment(),
        "source": source,
        "models": models,
        "probe_overlap_at_10": probe_overlap(models),
        "decision": decide(models),
        "samples_file": f"{experiment_id}.samples.jsonl",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = {"experiment_id": experiment_id, "count": len(samples)}
    ordered = sorted(samples, key=lambda s: (s["model_id"], s["run"], s["sequence"]))
    body = "\n".join([dumps({"record": "header", **header}), *(dumps(s) for s in ordered)]) + "\n"
    (OUTPUT_DIR / report["samples_file"]).write_bytes(body.encode("utf-8"))
    report["samples_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (OUTPUT_DIR / f"{experiment_id}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    titles = _titles()
    (OUTPUT_DIR / f"{experiment_id}.md").write_text(markdown(report, titles), encoding="utf-8")
    print(f"experiment: {experiment_id}")
    print(f"rule result: {report['decision']['winner']} (requires human sign-off)")
    return 0


def _titles() -> dict[str, str]:
    from ecommerce_search.ingestion.loader import load_catalog

    return {ln.record.product_id: ln.record.title for ln in load_catalog(SEED).lines}


def verify(path: Path) -> int:
    """Recompute every query-encode summary from the raw samples and compare."""
    report = json.loads(path.read_text(encoding="utf-8"))
    samples_path = path.with_name(report["samples_file"])
    body = samples_path.read_bytes()
    problems = []
    if hashlib.sha256(body).hexdigest() != report["samples_sha256"]:
        problems.append("samples file hash differs")
    _, samples = read_samples(samples_path)
    for model_id, m in report["models"].items():
        timed = [s for s in samples if s["model_id"] == model_id and s["kind"] == "timed"]
        if summarize([s["query_encode_ms"] for s in timed]) != m["query_encode_ms_pooled"]:
            problems.append(f"{model_id}: pooled summary differs")
        for r in m["per_run"]:
            values = [s["query_encode_ms"] for s in timed if s["run"] == r["run"]]
            if summarize(values) != r["query_encode_ms"]:
                problems.append(f"{model_id} run {r['run']}: summary differs")
    for problem in problems:
        print(f"MISMATCH: {problem}")
    print("OK" if not problems else "FAILED")
    return 0 if not problems else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.verify)
    if args.child:
        args.candidate = args.candidate[0]
        return child_main(args)
    if not args.candidate:
        parser.error("at least one --candidate MODEL_ID@REVISION is required")
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
