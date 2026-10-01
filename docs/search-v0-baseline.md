# Search V0 baseline: PostgreSQL lexical search

- Status: Milestone 3 implemented. `search_version = "v0_lexical"`, `document_version = "1"`.
- Related: `docs/architecture.md` §3.1 and §5, `docs/data-quality.md` §3.1 (check 10), `README.md`.
- **No relevance or quality claim is made.** There is no Golden Dataset yet (Milestone 9). The
  checks below are consistency checks and latency measurements on a **synthetic, templated
  240-product catalog**, which is easier than real listings.

## 1. Design

* **Storage.** Dedicated 1:1 table `product_search_documents` (migration `0003`): `product_id`
  (PK, FK to `products`), `document_version`, `source_content_sha256`, weighted `search_vector`
  (`tsvector`), `built_at`, and GIN index `ix_product_search_documents_search_vector`. No search
  column exists on `products`, no section text is duplicated, no vector (pgvector) column exists.
  The table is derived and rebuildable.
* **FTS configuration.** `simple`, fixed by the constant `FTS_CONFIG` and covered by
  `DOCUMENT_VERSION`. (The server's own `default_text_search_config` is `english`; V0 never
  relies on it: the configuration is always passed explicitly.)
* **Sections and weights (initial baseline, not tuned).**

  | Weight | Section | Content |
  |---|---|---|
  | A | name | title, brand |
  | B | taxonomy | explicit singular/plural forms of known category and subcategory values; hyphen-stripped model-identifier variants |
  | C | attributes | canonical technical attributes of the product's own category (`8gb ram`, `256gb`, `1024gb 1tb`, `ssd`, `nvme`, processor, gpu, OS, camera, `5000mah`, color, material, gender, `wireless`, `bluetooth`, `2.4ghz wireless`, `anc`) |
  | D | description | description text |

  Missing attributes emit nothing; negative booleans emit nothing (`anc=false` never emits
  `anc`). Excluded: raw seller lines, dataset/provenance, `is_synthetic`/`source_type`, review
  outcomes, seller id, price, availability, rating, review count, shoe size, screen size, weight,
  battery hours.
* **Plural forms** are an explicit versioned mapping (`laptop/laptops`, `phone/phones`,
  `shoe/shoes`, `headphone/headphones`, `smartphone/smartphones`, `ultrabook/ultrabooks`), not an
  inflector. No synonyms.
* **Query.** `plainto_tsquery('simple', q)`: strict AND, no OR fallback, never raises on user
  text. Whitespace collapsed; blank, control-character (incl. NUL) or over-long text is `422`;
  punctuation-only is `200` with no results.
* **Ranking.** `ts_rank` with query-time weights `{D: 0.1, C: 0.2, B: 0.4, A: 1.0}` and
  normalization flag 0, then `product_id` ascending. The score is query-relative and not a
  probability. Changing section assignments requires a `DOCUMENT_VERSION` bump and a reindex.
* **Synchronization.** Ingestion builds documents in the same transaction as the catalog rows;
  current documents (and `built_at`) are untouched; `search reindex --database NAME` heals
  missing and stale documents; `search status` and `catalog check --database` (check 10) report
  completeness. **A missing or stale index may produce incomplete results until `reindex`
  succeeds**; search never claims completeness. Running against a database not yet migrated to
  `0003` is refused with a sanitized message to run `uv run alembic upgrade head`.
* **Query text policy.** Rejected with `422` (fixed text, the input is never echoed): surrogate
  code points, control characters (NUL included) and invisible format characters (zero-width
  space, bidirectional overrides, byte-order mark, soft hyphen, ...). Accepted: ordinary
  whitespace (collapsed) and letters of every script, including the ZWNJ/ZWJ joiners used in
  Persian and Indic text, private-use and unassigned code points. A GET cannot carry a lone
  surrogate (the HTTP layer decodes invalid UTF-8 with replacement characters); a POST can, and
  is rejected.
* **Document semantics are pinned.** A golden SHA-256 over the four section texts of all 240
  seed documents, the section-to-weight assignment, `FTS_CONFIG` and the explicit plural
  mapping is recorded per `DOCUMENT_VERSION` in
  `tests/unit/search/test_document_corpus_digest.py`: version `1` →
  `86bf819eb93d57298432d9638e32da3f63ddee6df18bae6b88bad62fceda1b2d`. A semantic change must
  bump `DOCUMENT_VERSION`, pin a reviewed new digest, reindex and rerun evaluation.
* **Audit metadata.** Database quality reports carry a `search_index` section (document version,
  FTS configuration, search version, product and document counts, and missing / stale-version /
  hash-mismatch / tampered-vector counts); file reports state it is not applicable.

## 2. PostgreSQL tokenization observed (PostgreSQL 16.15, `simple`)

Pinned by `tests/integration/test_search_characterization.py` (output of `to_tsvector`):

| Input | Lexemes |
|---|---|
| `8GB` | `8gb` |
| `8 GB` | `8`, `gb` (so `8 gb` is a different query from `8gb`) |
| `256GB` | `256gb` |
| `14.0-inch` | `14.0`, `inch` |
| `i5` | `i5` |
| `LP101` | `lp101` |
| `In-ear` | `in-ear`, `in`, `ear` |
| `wireless_2_4ghz` | `wireless`, `2`, `4ghz` (hence the builder writes `2.4ghz wireless`) |
| `2.4GHz` | `2.4`, `ghz` |
| `shoe` / `shoes`; `headphone` / `headphones` | not conflated (no stemming) |
| `WH-1000XM5` | `wh-1000xm5`, `wh`, `1000xm5` (hence the hyphen-stripped identifier variant) |
| `NVMe`, `1TB`, `5000mAh` | `nvme`, `1tb`, `5000mah` |

Query side: `nike -shoes` becomes `'nike' & 'shoes'` (the hyphen is not an exclusion), `"nike`
becomes `'nike'`, `hp-laptop` becomes `'hp-laptop' & 'hp' & 'laptop'`, `!!!` becomes an empty
query (PostgreSQL emits a NOTICE; the service returns no rows without running the match). No
observation contradicted the plan assumptions; no switch to `english` was made.

Example stored vectors (excerpt, `SYN-HDP-0003`): `'anc':22C`, `'bluetooth':21C`,
`'headphone':15B`, `'headphones':7A,16B,28`, `'jbl':1A,14A`, `'hd103':13A`, `'wireless':6A,20C,27`.

## 3. Smoke-query behavior on the committed seed (provisional, not relevance labels)

Full result sets come from the internal retrieval function with a test-only unbounded limit; the
public API never returns more than `top_k` (maximum 50). Each set was compared with an
independent word-matching oracle that does not call the production builder or any PostgreSQL
full-text function.

| Query | tsquery | Matches in catalog | Notes |
|---|---|---|---|
| `laptop` | `'laptop'` | 80 | Exceeds the public limit of 50; **all 80 have the same score** (ties ordered by `product_id`). The endpoint returns only the first 50. |
| `hp laptop` | `'hp' & 'laptop'` | 12 | 4 distinct scores |
| `8gb laptop` | `'8gb' & 'laptop'` | 34 | |
| `256gb ssd laptop` | `'256gb' & 'ssd' & 'laptop'` | 15 | |
| `apple phone` | `'apple' & 'phone'` | 10 | Needs the category term; Apple phone titles do not contain "phone" |
| `iphone` | `'iphone'` | **0** | The term is not in the catalog (invented series names). A lexical limitation, not a success. |
| `nike shoes` | `'nike' & 'shoes'` | 10 | |
| `wireless headphones` | `'wireless' & 'headphones'` | 40 | |
| `anc headphones` | `'anc' & 'headphones'` | 20 | Exactly the products with `anc = true` |
| `zzqxv` | `'zzqxv'` | 0 | |
| `HP   Laptop`, `hp laptop!!`, `LAPTOP` | same as the plain forms | 12, 12, 80 | Case, whitespace and punctuation variants return the same lists |

Invariants verified by tests: every result contains every query term (oracle); the internal
result set equals the oracle set; for queries whose full set fits within `top_k` the public
result equals the full set; for the broad `laptop` query the public result is the first 50 of the
ranked list; scores are non-negative and non-increasing, ties are in ascending `product_id`, no
duplicates, `top_k` respected, repeated runs identical; seller-style variants (`8 GB`, `1TB`,
lower-case brand) are retrievable through canonical tokens; `anc=false` products never match
`anc`.

**V0 limitations (recorded, to be addressed later):**
* Strict AND: a term the catalog does not contain empties the result. `best laptop`,
  `coding ke liye laptop` and `sasta laptop` return nothing. `laptop for coding` returns only
  laptops whose *description* happens to contain both `for` and `coding`, as plain words, not as
  an interpreted intent. Filler words and Hinglish handling belong to Milestone 6.
* No stemming: only the explicit taxonomy plural forms above are matched.
* `8 gb` (with a space) does not equal `8gb`; no unit or number parsing exists.
* Quotes, hyphens and operators in a query are plain text.
* Scores of templated documents tie heavily; ordering inside a tie is by `product_id` only.

## 4. Index use (EXPLAIN, run 1 of the corrected benchmark, `top_k = 10`, ANALYZE after load)

PostgreSQL chose **sequential scans on 240 rows for every query**; the GIN index was **not used**
in these plans (plans were a hash join with sequential scans on both tables, or a nested loop
with a sequential scan on `product_search_documents` and a primary-key index scan on `products`).
Execution times reported by EXPLAIN ANALYZE were between 0.077 ms and 0.357 ms in the corrected
benchmark (the pre-fix benchmark showed the same plan shapes, 0.066 to 0.281 ms).
A test-only capability check shows the GIN index is usable: with only `enable_seqscan = off`, the
planner still chose a full primary-key index scan on the search table, and with
`enable_seqscan = off` and `enable_indexscan = off` it used a bitmap scan on
`ix_product_search_documents_search_vector`. That says nothing about the plan chosen normally;
production code never changes planner settings. No claim about index benefit at larger scale is
made: 240 rows cannot show it.

## 5. Latency (measured)

No latency target is set or implied. Two sets of measurements exist: the **corrected benchmark**
(5.1, current) and the **pre-fix, pre-commit measurements** (5.2, kept as historical evidence).

### 5.1 Corrected benchmark

* **Experiment** `lexical-v0-20261001T095432Z-548f08fa`, generated 2026-10-01T09:54:32Z.
* **Source fingerprint.** Base commit `e58a00ec7e11701815c277ccd9d3e6d75572e1a7`, working tree
  dirty (the M3 changes are uncommitted, so no commit hash identifies this code).
  `git_diff_sha256` `d01021a2c783f4be3e9ffbb3839ea68784608be7ba2a21b8cbbce6d95016f6a1`;
  `source_tree_sha256` `87b11e6a60b2bb164c1753cfec4144da17ffda41541cdcd9d928d0c63f003a41`, computed
  from sorted `path + Git-normalized blob hash` entries of the 59 relevant source, config,
  migration and seed files (12 of them untracked; `.env`, `.claude`, `data/processed`, caches and
  docs/tests are excluded). No future commit hash is claimed.
* **Protocol.** 3 independent runs; each creates a fresh scratch database, migrates to head,
  ingests the committed 240-product seed, runs `ANALYZE`, measures, and drops the database. 12
  queries; per query one cold first execution (recorded separately), 20 untimed warm-ups, then 200
  timed executions from a single sequential in-process client (FastAPI `TestClient`), `top_k = 10`.
  **2,400 timed samples per run, 7,200 across three runs, n = 200 per query per run**, plus 36 cold
  samples. Nearest-rank percentiles.
* **Raw samples are preserved**: `<experiment_id>.samples.jsonl` holds all 7,236 samples (header
  line + 7,200 timed + 36 cold), each with experiment id, run, query, query index, sample index,
  sequence, UTC timestamp, status code, result count, `lexical_ms`, `app_total_ms`,
  `serialization_ms`, `request_ms` and a diagnostic outlier flag. The summary was recomputed
  exactly from them (`scripts/benchmark_lexical.py --verify` printed "OK"). All statuses were 200.
* **Metrics.** `lexical_ms`: SQL and row fetch, reported by the response. `app_total_ms`: handler
  processing through result-model construction. `serialization_ms`: Pydantic `model_dump_json` of
  the parsed response, measured separately. `request_ms`: the in-process `TestClient.get` call
  (routing, validation, handler, framework response serialization). **The `request_ms` interval
  ends before `response.json()`, so client JSON decoding is not included, and it is not network
  latency: no socket and no real server are involved.**

Per-run overall results (ms, n = 2,400 per metric per run):

| Run | Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1 | lexical_ms | 3.98 | 6.445 | 8.563 | 61.37 |
| 1 | app_total_ms | 4.05 | 6.53 | 8.705 | 61.4 |
| 1 | serialization_ms | 0.036 | 0.064 | 0.085 | 0.736 |
| 1 | request_ms | 7.114 | 11.593 | 15.409 | 66.848 |
| 2 | lexical_ms | 3.857 | 5.43 | 6.192 | 8.645 |
| 2 | app_total_ms | 3.925 | 5.518 | 6.268 | 8.735 |
| 2 | serialization_ms | 0.035 | 0.058 | 0.074 | 0.161 |
| 2 | request_ms | 6.936 | 9.538 | 10.764 | 13.175 |
| 3 | lexical_ms | 4.071 | 6.34 | 8.03 | 55.016 |
| 3 | app_total_ms | 4.129 | 6.414 | 8.11 | 55.092 |
| 3 | serialization_ms | 0.036 | 0.062 | 0.085 | 0.282 |
| 3 | request_ms | 7.282 | 11.064 | 13.566 | 59.107 |

Pooled across the three runs (labelled aggregate, n = 7,200 per metric; per-run figures are
primary): `lexical_ms` p50 3.953 / p95 6.055 / p99 7.834; `app_total_ms` 4.022 / 6.14 / 7.935;
`serialization_ms` 0.036 / 0.062 / 0.082; `request_ms` 7.074 / 10.795 / 13.164.

Cold first executions (`request_ms`, one per query per run): the first query of every run
(`laptop`) took 41.673, 43.61 and 45.043 ms; every other first execution was between 6.571 and
12.983 ms. Per-query tables are in the generated artifact (`data/processed/search_benchmark/`,
git-ignored).

**Diagnostic outliers** (rule: `request_ms` or `lexical_ms` above 10 times the median of the same
run and query; diagnostic only, every flagged sample stays in all percentiles above). Two samples
were flagged, both on `lexical_ms` (the delay is inside the SQL and fetch time):

* run 1, `zzqxv`, sample 127, 2026-10-01T09:55:19.740129Z: `request_ms` 66.848,
  `lexical_ms` 61.37, `app_total_ms` 61.4;
* run 3, `8gb laptop`, sample 27, 2026-10-01T09:56:15.099281Z: `request_ms` 59.107,
  `lexical_ms` 55.016, `app_total_ms` 55.092.

Their cause was not established. The largest gap between consecutive recorded samples in any
run was 0.329 s, so no sample was stalled for seconds.

**Standby check.** The Windows System log shows the machine entering Modern Standby at 15:24:16
local time and exiting at 15:24:58. The benchmark started at 15:24:32 (09:54:32Z), so run 1's
setup (database creation, migration, ingestion) overlapped the end of that standby interval; the
first recorded sample is at 15:25:01.96 local, after the exit, and no later standby event
occurred before the run finished (about 15:26:34). No recorded sample fell inside a standby
interval, and none shows a stall. `powercfg /requests` requires administrator rights and was not
run; no power setting was changed.

Environment: Windows 11 (10.0.26300), Intel64 Family 6 Model 140 (8 logical CPUs), Python 3.12.3,
PostgreSQL 16.15 in a Docker container on the same host (`pgvector/pgvector:0.8.6-pg16`),
`shared_buffers` 128MB, `work_mem` 4MB, `jit` on, `max_parallel_workers_per_gather` 2, `fsync` and
`synchronous_commit` on. Configuration: `simple`, `plainto_tsquery`, `ts_rank`, weights
D 0.1 / C 0.2 / B 0.4 / A 1.0, normalization 0, `DOCUMENT_VERSION` 1, `SEARCH_VERSION`
`v0_lexical`, `SEARCH_LEXICAL_K` 50, `SEARCH_DEFAULT_TOP_K` 10, `SEARCH_MAX_QUERY_LENGTH` 200,
pool size 5 / overflow 5. Catalog: 240 products, 240 documents (seed checksum prefix
`2c1c5fa43eec5f73`).

Caveats: 240 synthetic rows; a Windows host with a containerized database (loopback and driver
overhead are included); one sequential client, no concurrency; in-process client; no cache
control beyond the warm-up. Because the plans are sequential scans, these numbers say nothing
about index performance.

### 5.2 Pre-fix, pre-commit measurements (historical, kept unchanged)

These came from the first benchmark (`benchmark_20261001T082249Z`), run from the uncommitted
working tree before the provenance and raw-sample fixes. That version recorded only the base
commit and a dirty flag and **discarded its raw samples**, so no individual sample can be
inspected.

| Run | Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1 | lexical_ms | 3.811 | 5.113 | 6.282 | 8.633 |
| 1 | request_ms | 6.698 | 8.783 | 10.781 | 13.477 |
| 2 | lexical_ms | 3.854 | 5.079 | 6.238 | 37.854 |
| 2 | request_ms | 6.779 | 8.811 | 11.079 | 41.915 |
| 3 | lexical_ms | 3.929 | 5.876 | 9.058 | 113.112 |
| 3 | request_ms | 6.915 | 10.071 | 15.149 | **98,931.725** |

Pooled (n = 7,200): `lexical_ms` 3.852 / 5.346 / 7.183; `request_ms` 6.776 / 9.171 / 12.111.
The other metrics of that run (`app_total_ms`, `serialization_ms`) were reported alongside and are
omitted here as superseded.

**The 98.9-second sample (run 3, query `256gb ssd laptop`) is kept in those figures.** Review
finding: it aligned strongly with a Windows Modern Standby interval (System log: entered
13:50:38, exited 13:52:33 local; run 3 began a few seconds before the entry). A database statement
timeout (30 s) or a pool timeout (10 s) would have aborted the run, so the delay was outside SQL
execution. It could not be paired with its same-sample `lexical_ms` (the maximum `lexical_ms` of
that query was 113.112 ms), because the old benchmark discarded raw samples; the cause is
therefore classified as a host scheduling anomaly that is strongly supported but not proven per
sample. The corrected benchmark (5.1) preserves raw samples, UTC timestamps and source
fingerprints so such cases can be examined.

### 5.3 Real-HTTP smoke (manual demo, not a benchmark)

On 2026-10-01 the application ran on `127.0.0.1:8000` (uvicorn) against the development
database (revision `0003`, 240 products, 240 documents, index `CURRENT`), and each smoke query
was sent once over real HTTP with `top_k = 5`. The `lexical_ms` values the responses reported were
between 6.947 and 14.02 ms and `total_ms` between 7.058 and 14.164 ms (`!!!` was 7.981 and 8.015 ms;
the two zero-result queries about 10.7 to 11.6 ms). These are **single, sequential requests, not
repeated and not warmed up**: no percentile is derived from them, and they are not comparable with
the benchmark in 5.1 (a different process, a one-off sample, no controlled protocol). The
smoke queries matched the consistency expectations in section 3 (descending scores, ties by
`product_id`, no duplicates, `top_k` respected, `iphone` and `zzqxv` empty with HTTP 200).

Three different things are therefore reported separately in this document, and none of them is a
relevance evaluation:
* **consistency checks** (section 3): the results agree with an independent word-matching oracle;
* **benchmark latency** (5.1): an in-process, controlled, repeated measurement;
* **real-HTTP smoke latency** (5.3): single requests from the manual demo.

Relevance evaluation is not available: there is no Golden Dataset until Milestone 9.

## 6. Not done in V0

Embeddings and vector retrieval (M4), hybrid/RRF (M5), query understanding and Hinglish handling
(M6), structured filters (M7), reranking (M8), Golden Dataset metrics (M9/M10), pagination,
stemming, synonyms, per-category attributes in results, a repeated real-HTTP latency benchmark.
