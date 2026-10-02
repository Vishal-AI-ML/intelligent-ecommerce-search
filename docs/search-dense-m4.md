# Milestone 4: dense semantic retrieval (local embeddings + pgvector)

- Status: Milestone 4 implemented. `search_version = "dense_only"` (not a V-numbered version;
  hybrid V1 is Milestone 5), `EMBEDDING_TEXT_VERSION = "1"`.
- Related: ADR-004 (strategy), ADR-006 (model selection and measured index decision),
  `docs/architecture.md` §5, `docs/data-quality.md` §3.1 and §7, `docs/search-v0-baseline.md`.
- **No relevance or quality claim is made.** There is no Golden Dataset yet (Milestone 9). Every
  number below is latency, determinism or exact-versus-HNSW agreement on a **synthetic,
  templated 240-product catalog** on one laptop.

## 1. Design

* **Model (ADR-006).** `sentence-transformers/all-MiniLM-L6-v2` at immutable revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`: 384 dimensions, normalized embeddings, no
  query/document prefix, maximum sequence length 256, CPU only. The specification is pinned in
  code (`embeddings/spec.py` `ALL_MINILM_L6_V2`, configuration hash
  `6d5cfc238e11c60053e8227b5206d295ef4f06b3fcfbd8d359c2bbb96efb743b`); settings may only select a
  reviewed registry entry.
* **Weights.** Fetched explicitly, once, with `python -m ecommerce_search.search model-fetch` into
  the git-ignored repository `models/` directory (Hugging Face hub layout plus a SHA-256 file
  manifest). It is the only command that uses the network. Every other path loads the local
  snapshot offline (`HF_HUB_OFFLINE=1`, `trust_remote_code=False`); weights are never committed
  and never copied into the Docker image.
* **Embedding text.** Built by `embeddings/text.py` from a closed whitelist of validated catalog
  fields (title, brand, category/subcategory, typed specifications, description). Excluded: raw
  seller lines, seller id, price, currency, rating, review count, availability, provenance,
  synthetic markers, review outcomes and labels, shoe size. The corpus digest for text version 1
  is pinned in `tests/unit/search/test_embedding_corpus_digest.py`:
  `7faa9d147f03da730bab18d6ca93894700b14508ed5e968e569cfd1c014618e4`.
* **Storage.** `product_embeddings` (migration `0004`): one row per product with `embedding`
  `vector(384)` and `model_id`, `model_revision`, `dimension`, `normalized`,
  `embedding_text_version`, `embedding_config_sha256`, `source_content_sha256`,
  `embedding_text_sha256`, `embedded_at`. Named CHECK constraints reject wrong dimensions, zero
  vectors and non-unit vectors flagged as normalized; pgvector 0.8.6 rejects NaN and infinity on
  input (all verified against a real server). The only index is the primary key: **no HNSW or
  IVFFlat index** (section 5).
* **Freshness (content-addressed).** A row is current only when its model id, revision,
  configuration hash, text version, dimension and normalization match the active
  configuration **and** `source_content_sha256` equals the product's current `content_sha256`.
  Product content changes, model/configuration changes and embedding-text changes therefore make
  a row stale. A dataset-version-only change with identical effective product content does not
  (see `docs/data-quality.md` §7).

## 2. Generation workflow

```bash
uv run python -m ecommerce_search.search model-fetch \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41      # network, once
uv run alembic upgrade head                                  # revision 0004 (schema only)
uv run python -m ecommerce_search.catalog ingest --database <db>
uv run python -m ecommerce_search.search embed --database <db>
uv run python -m ecommerce_search.search embed-status --database <db> --require-current
```

* `embed` runs in three phases: a read-only plan, encoding with no transaction (every text is
  checked against the 256-token limit first; nothing is ever truncated), then one write
  transaction under the embedding advisory lock. Any failure writes nothing; current rows keep
  their `embedded_at`; `--all` re-embeds everything.
* Lock order is dataset → lexical search → embedding. Ingestion never loads the model and never
  touches embeddings; a product changed by a concurrent ingestion stays detectably stale until
  the next `embed`.
* `embed-status` is read-only and reports missing, stale (model, text version, configuration,
  content), dimension, normalization and invalid-vector counts; `--verify-vectors` re-encodes and
  compares. Dataset identity is reported by `catalog check --database` (its `dataset` section)
  and recorded in every benchmark artifact, not by `embed-status`.
* `catalog check --database` adds the check `dense_embedding_consistency`: missing or stale rows
  are warnings, malformed, tampered or invalid rows are errors, and a `dense_index` report
  section records the configuration and counts (not applicable on file reports).

## 3. API

`GET /search/dense?q=...&top_k=...` and `POST /search/dense` (`{"query", "top_k"}`) share one
path; `GET`/`POST /search` (lexical V0) and `/health` are unchanged.

* Exact cosine distance in pgvector (`<=>`), ordered by `dense_score` (cosine similarity,
  `1 - distance`) descending, then `product_id` ascending. The score is uncalibrated and **not a
  probability**. There is **no similarity threshold**: every query returns up to `top_k` nearest
  products.
* Only current embeddings take part, so a changed product never ranks with its old vector.
  Results can be incomplete while embeddings are missing or stale; search never claims
  completeness.
* Validation as `/search` (blank, control, surrogate and invisible-format characters, over-long
  text: `422`, input never echoed); `top_k` is 1..`SEARCH_DENSE_K` (default 50). A query with no
  letter or digit returns `200` with no results without loading the model.
* Response metadata: `search_version`, `embedding_model_id`, `embedding_model_revision`,
  `embedding_dimension`, `embedding_text_version`, `distance_metric: "cosine"`, `top_k`,
  `result_count`, `applied_filters` (always empty) and `latency_ms` with `model_load_ms` (null
  unless loaded in this request), `query_embedding_ms`, `vector_ms` and `total_ms`.
* Missing model snapshot, model load or encode failure, missing table or database error: a fixed
  `503 {"detail": "search unavailable"}`. The server log records only a reason code (for example
  `snapshot_missing`) and an operator hint, never paths, SQL, hosts, credentials or stack traces.
* The model loads lazily on the first dense request (the first request is slow; torch import
  plus load took about 11.3 s in the Docker check in section 7). Inference is serialized by a lock.

## 4. Benchmark (valid experiment `dense-m4-20261002T060655Z-e3c7b7dc`)

**Identity.** Harness commit `5c388cd2d1a523537c8a33d37c79490a911957b5` (clean tree; source tree
SHA-256 `4a42e4c732752fa1779cf721fc91df6fe8e7697c46fdc03d21b7cb481c9908da`); production code
identical to commit `0ab972e1cc45dd489b262a39962a4c6debea5c43` (guarded paths: `src/`,
`migrations/`, `pyproject.toml`, `uv.lock`, `docker-compose.yml`, `alembic.ini`). Dataset
`synthetic-seed` version 1, checksum
`2c1c5fa43eec5f7346df40c17e57c07f6594ab1833c5d9d4c862e6d2e8bbbbc8`, record count 240,
transform/taxonomy/rules versions 1/1/1, identical in all three runs; 240 products and 240
embeddings per run. Queries: the 12 committed queries of `scripts/embedding_model_selection.py`
(set SHA-256 `43b78da81068b57cbb5b6b5cfabace65cf1e6ba0cd71cd8fa0ad000941c167a2`); query-vector
SHA-256 `1e41e6aae792949fcd26cb8b7a96d58bcad3dda9ebcb3d3f33037ddb4fd6af89`, identical in every run.

**Environment.** Windows 11 (10.0.26300), Intel i5-1135G7 (8 logical CPUs), Python 3.12.3,
PostgreSQL 16.15 with pgvector 0.8.6 in Docker on the same host (`shared_buffers` 128MB, `jit`
on), torch 2.14.1+cpu (4 threads), sentence-transformers 6.1.0, transformers 5.18.0. AC power,
lid open, a process-scoped keep-awake request; zero Kernel-Power events in the run interval
(2026-10-02 06:06:55Z to 06:13:15Z); maximum gap between consecutive samples 0.534, 0.797 and
0.748 s per run.

**Protocol (predeclared).** Three runs, each in a fresh subprocess with a fresh scratch database
(`ecommerce_search_bench_*`, dropped afterwards). Per query and k: 1 cold sample (reported
separately), 20 untimed warm-ups, 200 timed samples; nearest-rank percentiles; 50,652 raw
samples (50,400 timed and 252 cold). Per-run figures are primary.

**Artifacts** (git-ignored, not committed; SHA-256): `.json`
`19e7ed1543303d173b003c029e08ff0ffbbfef8689c22383e840c8261ade3628`, `.md`
`9610799b2706ef0dc1280a328b04e3ad8b2a90267407d3280c0d2aac3ab87ed2`, `.samples.jsonl`
`805414a699aee1f4d07f3d8b751115bb9c2a412f46c9594cd68555a009b8f70c`. `--verify` recomputed every
summary and the decision inputs exactly, and a separate script that does not use the harness's
derivation reproduced all counts, percentiles, recall values, plan facts and criteria with zero
mismatches.

An earlier attempt, `dense-m4-20261002T053443Z-44107e19`, is **invalid** (the machine entered
Modern Standby during run 2); its artifacts were deleted and none of its numbers are used.

### 4.1 Embedding generation (240 products)

| Run | Import (ms) | Load (ms) | First embed (ms) | Products/s | Encode (ms) | Residual (ms) | Second embed (ms) | RSS start / after load / after embed / peak (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8175.05 | 205.318 | 3935.306 | 60.986 | 3566.012 | 369.294 | 47.467 | 100.5 / 542.1 / 620.4 / 669.3 |
| 2 | 7667.503 | 200.873 | 3883.933 | 61.793 | 3437.74 | 446.193 | 40.861 | 100.4 / 541.9 / 640.1 / 669.8 |
| 3 | 10628.51 | 255.288 | 5009.001 | 47.914 | 4633.528 | 375.473 | 64.301 | 100.4 / 542.6 / 597.7 / 671.6 |

Import is torch plus sentence-transformers; encode is measured by a pass-through wrapper around
the provider; residual is first embed minus encode (planning, text building, the token-limit
check and the write transaction). Every first embed inserted 240 rows (0 updated, 0 changed during
the run); every second embed was idempotent (240 unchanged, `embedded_at` unchanged); final
status CURRENT with 0 vector mismatches on re-encoding.

### 4.2 B1: production end to end (`GET /search/dense`, top_k=10, ms)

| Metric | Run 1 p50 / p95 / p99 | Run 2 | Run 3 |
|---|---|---|---|
| `query_embedding_ms` | 9.186 / 11.962 / 14.156 | 9.255 / 11.815 / 13.387 | 10.952 / 17.017 / 20.906 |
| `vector_ms` | 5.814 / 7.437 / 8.488 | 5.762 / 7.219 / 7.958 | 6.295 / 8.228 / 9.843 |
| `app_total_ms` | 15.117 / 19.322 / 22.253 | 15.118 / 18.906 / 21.407 | 17.474 / 24.897 / 29.396 |
| `serialization_ms` | 0.049 / 0.075 / 0.095 | 0.048 / 0.075 / 0.103 | 0.054 / 0.09 / 0.127 |
| `request_overhead_ms` | 4.051 / 5.471 / 6.134 | 4.128 / 5.332 / 6.05 | 4.736 / 6.184 / 7.005 |
| `request_ms` | 19.213 / 24.596 / 27.673 | 19.25 / 24.076 / 27.147 | 22.311 / 30.496 / 35.008 |

Definitions: `request_ms` is the in-process FastAPI TestClient call (ends before
`response.json()`; not network latency). `app_total_ms` is the API's `total_ms` (handler entry
through result-model construction). `serialization_ms` is measured directly, as in M3: the
payload is parsed with `DenseSearchResponse.model_validate(...)` and `model_dump_json()` is timed
on its own, outside `request_ms`. `request_overhead_ms = request_ms - app_total_ms` is derived:
routing, validation, dependency resolution and framework response serialization; it is **not**
a serialization measurement. Cold first requests (including the model load in each run's
application) took 210.1, 196.4 and 302.5 ms. The median `request_ms` over all B1 samples is
19.976 ms.

### 4.3 B2: production exact retrieval (`dense_search()`, `vector_ms`, ms)

| k | Run 1 p50 / p95 / p99 | Run 2 | Run 3 |
|---|---|---|---|
| 10 | 1.975 / 2.877 / 3.95 | 2.144 / 3.026 / 3.888 | 2.204 / 3.201 / 3.895 |
| 50 | 2.664 / 4.081 / 4.793 | 2.838 / 4.227 / 5.091 | 2.835 / 3.704 / 4.533 |

Every sample of a query returned an identical ordered list, scores never increased, and ties were
in `product_id` order. Before measurement the only index on `product_embeddings` was
`pk_product_embeddings` (0 HNSW/IVFFlat indexes). Every production plan (24 per run) was
`Limit > Sort > Hash Join > Seq Scan on product_embeddings`: an exact scan.

### 4.4 Experimental exact-versus-HNSW comparison (not production SQL, `ann_sql_ms`, ms)

C1 and C2 use one ANN-compatible SQL text (ORDER BY distance LIMIT k in a CTE, then the
content-hash join) because the production SQL cannot use an HNSW index; C1 has no index, C2 a
temporary HNSW index (m 16, ef_construction 64, `hnsw.ef_search` 100) timed with session-local
`enable_seqscan = off`. These numbers are **never production latency**; B2 is the production
reference. The C-phase order alternates to control order effects.

| Run (C-phase order) | C1@10 p50 / p95 / p99 | C2@10 | C1@50 | C2@50 |
|---|---|---|---|---|
| 1 (C1>C2) | 1.33 / 1.834 / 2.369 | 1.586 / 2.23 / 2.977 | 1.587 / 2.111 / 2.621 | 1.785 / 2.336 / 2.783 |
| 2 (C2>C1) | 1.596 / 2.194 / 2.573 | 1.534 / 2.18 / 2.728 | 1.733 / 2.497 / 2.995 | 1.71 / 2.357 / 2.966 |
| 3 (C1>C2) | 1.486 / 2.034 / 2.416 | 1.701 / 2.31 / 2.822 | 1.562 / 2.068 / 2.43 | 1.859 / 2.854 / 3.527 |

* Index build 231.368, 596.235 and 167.805 ms; size 499,712 bytes in every run.
* Natural plans (no forcing): the planner chose the HNSW index in **0 of 72** cases (sequential
  scan). Forced plans: **72 of 72** used `Index Scan` on the HNSW index.
* Agreement with exact (B2): strict and tie-aware recall 1.0 at k=10 and k=50 for every query
  and run; ordered lists identical in all 36 comparisons per suite and k; maximum distance
  difference 0.0.

### 4.5 Predeclared decision rule (fixed before execution)

| Criterion | Result | Evidence |
|---|---|---|
| 1. Every run: C2 p50 at k=10 beats C1 by more than the larger run-to-run spread, by at least 25 % and by at least 1 ms | fail | savings -0.256, +0.062 and -0.215 ms (C1 1.33 / 1.596 / 1.486 vs C2 1.586 / 1.534 / 1.701); spread 0.266 ms |
| 2. Smallest saving at least 5 % of the B1 median `request_ms` | fail | -0.256 ms vs 0.999 ms required (5 % of 19.976 ms) |
| 3. Tie-aware recall 1.0 for every query, k and run | pass | minimum 1.0 |
| 4. The planner chooses HNSW without forcing | fail | 0 of 72 |

**Outcome: retain the exact scan at 240 rows; no persistent ANN index.** HNSW gave identical
results but no latency benefit, and the planner would not use it. This applies only to the
current 240-product synthetic catalog; ANN indexing must be re-evaluated at materially larger
catalog sizes.

## 5. Index use

PostgreSQL scans `product_embeddings` sequentially for every production query (section 4.3).
The planner also ignored a temporary HNSW index unless forced (section 4.4). No vector index is
created by migrations; the decision is measured, not assumed, and is recorded in ADR-006.

## 6. Limitations

* 240 synthetic, templated products; one laptop; an in-process client; one sequential client;
  no concurrency. Nothing here predicts behaviour at larger scale.
* No relevance metric exists before the Golden Dataset. The ADR-006 probes are qualitative only;
  semantic intents such as "student" or "lightweight" are weakly supported by the catalog text.
* English-only model: Hinglish queries are not understood (query understanding is Milestone 6).
* No similarity threshold: nonsense queries still return the nearest products.
* The first dense request in a process pays the torch import and model load.
* Hybrid retrieval (RRF, V1) is Milestone 5; filters, reranking and Golden Dataset evaluation are
  later milestones.

## 7. Docker verification (2026-10-02, scratch database only)

* **Image.** Built from the committed tree only (`git archive HEAD | docker build -`) at commit
  `5c388cd2d1a523537c8a33d37c79490a911957b5` with the committed Dockerfile and `uv sync --frozen`
  (57 locked packages, `torch==2.14.1+cpu`); `uv.lock` unchanged. Base images pulled explicitly:
  `python:3.12-slim-bookworm` digest
  `sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3` and
  `ghcr.io/astral-sh/uv:0.11.12` digest
  `sha256:3a59a3cdd5f7c217faa36e32dbc7fddbb0412889c2a0a5229f6d790e5a019dd7` (tags can move; the
  digests identify what was used). Build exit 0 in 270 s; image
  `sha256:c381aceffd79f1ea9f6f0fe9004c949925302256896eaa2519e18ba73a48c0d0`, 2,052,548,738 bytes
  (linux/amd64, Python 3.12.15).
* **No weights in the image.** A network-disabled container without a mount found no
  `/models`, no `.safetensors`/`.onnx`/`*.bin`, no `models--*` snapshot or manifest and no
  Hugging Face cache.
* **Data.** A scratch database `ecommerce_search_docker_*` was migrated to 0004, the seed
  ingested (240 products, 240 lexical documents CURRENT), all 240 embeddings generated with the
  local MiniLM snapshot (status CURRENT, 0 vector mismatches) and audited (`catalog check`: 0
  errors, check 10 pass, `dense_embedding_consistency` pass). It was dropped afterwards; the
  development database was never targeted.
* **Valid read-only mount** (`models/` at `/models:ro`; write attempt failed with
  "Read-only file system"): `/health` 200, `/search` 200, `GET` and `POST /search/dense` 200 with
  typed responses, unique product ids, descending scores, model and revision metadata; repeated
  identical requests returned the same order. The first dense request loaded the model
  (`model_load_ms` 11308.051); later requests reported `model_load_ms` null. No model file was
  written (`docker diff` showed only the mount point and a torch cache directory under `/tmp`).
* **Empty read-only mount and no mount:** `/health` 200 and `/search` 200; `GET` and
  `POST /search/dense` returned `503 {"detail": "search unavailable"}`. The log recorded only
  `dense search unavailable: snapshot_missing` with the `model-fetch` hint, and no path, SQL,
  password or traceback.
