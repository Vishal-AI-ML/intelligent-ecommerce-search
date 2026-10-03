# Milestone 5: hybrid retrieval (lexical + dense + Reciprocal Rank Fusion, V1)

- Status: Milestone 5 implemented. `search_version = "v1_hybrid"`, fusion method `rrf`,
  `rrf_k = 100` with `rrf_k_status = "provisional"`.
- Related: ADR-002 (fusion decision and the M5 `rrf_k` record), ADR-006 (embedding model),
  `docs/architecture.md` §3, `docs/spec.md` FR-RET-3 and §9.1, `docs/search-v0-baseline.md`,
  `docs/search-dense-m4.md`.
- **No relevance or quality claim is made.** There is no Golden Dataset yet (Milestone 9). Every
  number below is a deterministic attribute-consistency proxy, an overlap or composition count,
  a determinism check or a latency measurement on a **synthetic, templated 240-product catalog**
  on one laptop. The proxy is not relevance, not quality ground truth and not a human label.
- Status note (Milestone 6, 2026-10-03): the runs recorded here predate Milestone 6; their
  queries, results and conclusions are unchanged. M6 added deterministic query understanding
  that `/search/hybrid` reports for information only (`docs/query-understanding-m6.md`); it does
  not change hybrid retrieval, RRF, `rrf_k`, ranks or scores. Retrieval still treats Hinglish
  and filler words as plain query text, as before Milestone 6, until Milestone 7 or another
  approved version changes it. Where this document says that Hinglish or query understanding is
  Milestone 6, read it as parsing only.

## 1. Scope

* New `GET /search/hybrid` and `POST /search/hybrid` (V1). `GET`/`POST /search` remains lexical
  V0 (`v0_lexical`), `GET`/`POST /search/dense` remains dense-only (`dense_only`), and `/health`
  is unchanged.
* No schema migration: hybrid search reads the existing `product_search_documents` (revision
  `0003`) and `product_embeddings` (revision `0004`) tables. Dense retrieval is still the exact
  cosine scan; no persistent ANN index exists (ADR-006).
* Not in M5: query understanding (M6), structured filters (M7, `applied_filters` is always
  empty), reranking (M8), the Golden Dataset and retrieval metrics (M9, M10).

## 2. Algorithm

Code: `search/hybrid.py` (`fuse_rrf`, `read_sources`) and `api/hybrid.py`.

1. **Validate** the query text and `top_k` exactly as `/search/dense` does (section 3).
2. **Model work first.** Load the embedding model if needed, check the query against the
   256-token limit and encode it. No database connection is checked out during this step.
3. **One snapshot.** Run the existing `lexical_search` (depth `SEARCH_LEXICAL_K`, default 50)
   and `dense_search` (depth `SEARCH_DENSE_K`, default 50) in one PostgreSQL transaction whose
   first statement is `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY`, so both lists
   come from the same snapshot. The setting is transaction-local; no engine or connection
   setting changes. The transaction ends (commit on success, rollback on error) before fusion.
4. **Fuse** with Reciprocal Rank Fusion, equal source weights:

   ```text
   rrf_score(d) = sum over sources s containing d of 1 / (rrf_k + rank_s(d))
   ```

   `rank_s(d)` is the existing 1-based positional rank of `d` in that source's list (not a
   score). Scores are summed as exact `Fraction` values, so ordering never depends on float
   rounding; the order is `rrf_score` descending, then `product_id` ascending. The input lists
   are checked: ranks must be contiguous 1..n in list order and product ids unique.
5. **Truncate** the fused union to `SEARCH_CANDIDATE_K` (default 50) candidates and return the
   first `top_k` of them. Each result keeps its lexical rank and score and its dense rank and
   score; a source in which the product did not appear gives `null` rank and score.

Behaviour at the edges:

* **Dense capability unavailable: fail closed.** A missing model snapshot, a model load or
  encode failure, a missing table or any database error returns the same fixed
  `503 {"detail": "search unavailable"}` as `/search/dense`. Hybrid search never silently
  degrades to lexical-only results. The log records only a reason code and an operator hint.
* **Incomplete embeddings.** Only embeddings current for the active model and the product's
  current content take part (the M4 freshness rule). While embeddings are missing or stale,
  the dense list can be shorter (`dense_hit_count`), so fewer fused candidates may exist. This
  is the normal dense contract, not a fallback, and the response does not claim completeness.
* **No searchable text.** A query with no letter or digit returns `200` with no results
  without loading the model or touching the database: `dense_status =
  "skipped_no_searchable_text"`, `tsquery` null, every hit, overlap, fused and candidate count
  0, and every stage timing except `total_ms` null.
* **Scores.** `rrf_score` is a rank-fusion value, not a relevance probability. `lexical_score`
  (`ts_rank`) and `dense_score` (cosine similarity) are uncalibrated and reported only for
  inspection. Dense retrieval has no similarity threshold, so every query with searchable text
  receives dense neighbours.

## 3. API contract

`GET /search/hybrid?q=...&top_k=...` and `POST /search/hybrid` with body
`{"query": "...", "top_k": n}` (unknown body fields are rejected) share one code path.

* Validation as `/search` and `/search/dense`: blank, control, surrogate and invisible-format
  characters and text over `SEARCH_MAX_QUERY_LENGTH` are `422` (input never echoed), as is a
  query over the model's 256-token limit. `top_k` defaults to `SEARCH_DEFAULT_TOP_K` (10) and
  must be between 1 and `SEARCH_CANDIDATE_K` (50).
* Settings: `SEARCH_CANDIDATE_K` (default 50, must not exceed `SEARCH_LEXICAL_K +
  SEARCH_DENSE_K`) and `SEARCH_RRF_K` (default 100), both forwarded to the `api` container by
  Compose; source depths reuse `SEARCH_LEXICAL_K` and `SEARCH_DENSE_K`.

Response (`HybridSearchResponse`):

| Field | Meaning |
|---|---|
| `query` | The query after whitespace normalization |
| `search_version` | `"v1_hybrid"` |
| `document_version` | Lexical document version |
| `tsquery` | Generated tsquery text; `null` when no SQL ran |
| `embedding_model_id`, `embedding_model_revision`, `embedding_dimension`, `embedding_text_version` | Active embedding configuration |
| `distance_metric` | `"cosine"` |
| `fusion` | `method` (`"rrf"`), `rrf_k`, `rrf_k_status` (`"provisional"`), `lexical_k`, `dense_k`, `candidate_k` |
| `dense_status` | `"used"` or `"skipped_no_searchable_text"` |
| `lexical_hit_count`, `dense_hit_count` | Source list lengths (at most `lexical_k` / `dense_k`) |
| `overlap_count` | Products present in both source lists |
| `fused_count` | Unique products in the union of both lists (before truncation) |
| `candidate_count` | Fused candidates kept (at most `candidate_k`) |
| `top_k`, `result_count` | Requested maximum; results in this response |
| `results[]` | `rank` (1-based position in this response), `rrf_score`, `lexical_rank`, `lexical_score`, `dense_rank`, `dense_score`, then the product fields of `/search/dense` (`product_id`, `title`, `brand`, `category`, `subcategory`, `description`, `price` as a decimal string, `currency`, `rating`, `review_count`, `availability`) |
| `applied_filters` | Always empty |
| `latency_ms` | `lexical_ms`, `model_load_ms`, `query_embedding_ms`, `vector_ms`, `rrf_ms`, `total_ms` |

Null semantics in `latency_ms`: `lexical_ms`, `vector_ms`, `query_embedding_ms` and `rrf_ms` are
`null` when that stage did not run; `model_load_ms` is `null` unless this request loaded the
model. `total_ms` covers handler entry through result-model construction and excludes request
validation and framework response serialization. `serialization_ms` is not reported by the API;
the benchmark harness measures it separately (section 6).

## 4. Phase S: provisional `rrf_k` selection

**Experiment** `hybrid-m5-s-20261002T130708Z-2c997cb9` (2026-10-02 13:07:08Z to 13:08:04Z).
Harness commit `c59475cbf9a590fb5a1757b5fc82d624b346f22b` (clean tree, source tree SHA-256
`d61896c14cd0d7b34368bae5085ebe74f6f55354348fee84f1c72c940db797b6`); pinned production commit
`f7cb2757e669cb58b281c9ac3d8a37b6da7e1a7a`. Artifacts (git-ignored, not committed; SHA-256):
`.json` `5dd84db54496ebb008d02e48d6a3631318be52ef13d91ad25f10791b27e2cea1`, `.md`
`5a1e53b7cca5075ae698507fddde5f8465b22675db18ebdc60c75dcf213de345`, `.samples.jsonl`
`733f11e3a646c18daad0fb0d3718254087aa4490baec6d739054e1a4c42dd8a4` (276 records plus the
header).

**Frozen protocol (committed before execution in `scripts/benchmark_hybrid.py`).** 17 queries
(query-set SHA-256 `ea1df08db89a8bceb0e5b6437f18e75a30aed997fb70d727d080b441ebbd78d3`) with an
explicit deterministic expectation table (SHA-256
`ac5647876b33397bc0b0cc2d60e60357af9285daf65f5973e5c675f824c8d643`). 15 queries are checked;
`iphone` and `zzqxv` have no approved explicit expectation and are diagnostic only (excluded
from every aggregate). Expectations come from nine approved terms only: `laptop`, `phone`,
`shoes`, `headphones` (category), `hp`, `apple`, `nike` (brand), `anc` and `noise cancelling`
(`anc = true`). Nothing is inferred at run time; for example `8gb laptop` checks only the
laptop category, not RAM.

**Proxy.** Per checked query: the number of the top 10 results whose catalog facts (category,
brand, anc) equal every expectation, divided by 10 (missing results count as not satisfying);
the aggregate is the exact mean over the 15 checked queries.

**Procedure.** Two runs, each in a fresh subprocess with a fresh scratch database
(`ecommerce_search_bench_*`) and 240 current embeddings. Per run, each query's lexical and
dense lists were retrieved once with production `read_sources` (depth 50, one snapshot), and
those same lists were fused with production `fuse_rrf` for every value of the grid
`[1, 5, 10, 20, 40, 60, 100]`; an independent exact-fraction oracle re-checked every fused id,
rank and score. Decision rule (predeclared): highest mean proxy; an exact tie goes to the
largest tied `rrf_k`; no value is proposed unless both runs are identical.

**Result.**

| `rrf_k` | 1 | 5 | 10 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|
| Mean proxy, run 1 and run 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

* Both runs were identical (every source list and every fused list of every query and
  `rrf_k`).
* **All seven values tied at 1.0.** The predeclared tie rule therefore selected `rrf_k = 100`,
  the largest tied value. The harness recorded this as `candidate_proposed_pending_review`; it
  was adopted separately in commit `86f7b68f39155a002390bffc110036ad75dbb326` with status
  `provisional`.
* **The proxy was saturated and gives no evidence that 100 is better than any other grid
  value.** The grid did change rankings (11 of the 17 queries had more than one distinct top-10
  list across the seven values), but every one of those lists satisfied every explicit
  expectation. The choice is a tie-break, not a relevance result.
* Milestone 10 must revisit `rrf_k` on the human-reviewed Golden Dataset (ADR-002).

## 5. Phase CL: lexical, dense and hybrid comparison

**Experiment** `hybrid-m5-cl-20261002T155427Z-8636430a` (2026-10-02 15:54:27Z to 16:10:24Z).
Harness commit `691600937ad3cbe6abb47ae6839683726d72fd81` (clean tree, source tree SHA-256
`bf34d2851d09b03c311c1014e548858fe8f307b7bbe0a1f124269fe4d625c67e`); pinned runtime commit
`86f7b68f39155a002390bffc110036ad75dbb326` with settings lexical/dense/candidate depth 50 and
`rrf_k` 100 (`provisional`). Artifacts (git-ignored, not committed; SHA-256): `.json`
`9a0626dd6564df527235076c05173dc01ffe608b519b4a5e70b301ee656f4d3e`, `.md`
`008f2811300d92219215e59749e5962c1d76b6c73a3b04c7aa314d915ecb7653`, `.samples.jsonl`
`c313fbe80887544471ba0f48b7c7cf11c963781fab0035ddebba0cabf66253de` (30,912 records plus the
header).

**Identity.** Dataset `synthetic-seed` version 1, checksum
`2c1c5fa43eec5f7346df40c17e57c07f6594ab1833c5d9d4c862e6d2e8bbbbc8`, 240 records; 240 products
and 240 embeddings per run; `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; catalog-facts SHA-256
`d40c037b8982cdd1817816c2a3ab15e483e2bcfbcf7c75cc04b05f24d8db21d5` in every run; the same 17
queries and expectation table as Phase S.

**Protocol (predeclared).** Three runs, each in a fresh subprocess with a fresh scratch
database. In-process `GET /search`, `GET /search/dense` and `GET /search/hybrid` at
`top_k = 10`; the endpoint order rotates per run (run 1 search > dense > hybrid, run 2
dense > hybrid > search, run 3 hybrid > search > dense). Per endpoint and query: 1 cold
request, 20 untimed warm-ups, 200 timed requests. The comparison uses the result lists of the
same runs.

### 5.1 Comparison (provisional, non-authoritative)

All values below were identical in each of the three runs.

| Endpoint | Mean proxy (15 checked queries) | Queries with zero results (of 17) |
|---|---|---|
| `/search` (V0) | 11/15 = 0.733 | 6 |
| `/search/dense` | 73/75 = 0.973 | 0 |
| `/search/hybrid` (V1) | 1 = 1.0 | 0 |

* **Zero results.** Lexical search returned nothing for `iphone`, `zzqxv`,
  `coding ke liye laptop`, `student laptop`, `lightweight laptop` and
  `noise cancelling headphones` (strict-AND FTS; four of these are checked queries and score 0,
  which accounts for 11/15). Dense and hybrid returned 10 results for all 17 queries, including
  the diagnostic nonsense query `zzqxv`, because dense retrieval has no similarity threshold.
* **Dense below 1.0.** `apple phone` (7/10) and `nike shoes` (9/10) are the only checked queries
  where the dense top 10 included products outside the expected brand.
* **Mean unordered top-10 overlap** over the 17 queries: search–dense 2.94 (50/17),
  search–hybrid 5.06 (86/17), dense–hybrid 7.59 (129/17). For the six lexical zero-result
  queries the hybrid list is identical, in order, to the dense list.
* **Hybrid source composition** over 170 hybrid results (17 × 10): 109 in both source lists,
  0 lexical-only, 61 dense-only (60 of them from the six lexical zero-result queries, 1 from
  `premium phone`).
* **Determinism.** Within each run every cold and timed request returned the recorded result
  list; the warm-ups were checked the same way by the harness control flow but not recorded.
  The result lists of every endpoint and query were identical across the three runs.
* **Lexical ties.** Every one of the 11 queries with lexical results had tied `ts_rank` values
  within the `/search` top 10 (for `laptop` all ten share one score), and the same 11 queries
  had tied lexical scores within the hybrid top 10. Tied products are ordered by `product_id`,
  so their lexical positions, and therefore the lexical rank RRF consumes, reflect identifier
  order rather than a stronger text match.

**Interpretation.** The proxy only checks explicit category, brand and `anc` facts on a
templated catalog. It does not measure relevance, intent (for example "student" or
"lightweight") or ranking quality. Hybrid's 1.0 means that its top 10 never violated an
explicit category/brand/anc expectation of these 15 queries; it is not evidence that hybrid is
more relevant than lexical or dense retrieval, and the Phase-CL result does not re-select
`rrf_k`. Authoritative comparison of V0 and V1 is Milestone 10.

### 5.2 Latency (ms, per run)

Definitions: `request_ms` is the in-process FastAPI TestClient `GET` call (it ends before
`response.json()`; it is not network latency). `app_total_ms` is the API's `latency_ms.total_ms`.
Stage timings are the API's own fields. `serialization_ms` is measured directly: the parsed
response model's `model_dump_json()` is timed on its own, outside `request_ms`; the difference
`request_ms - app_total_ms` is framework overhead and is not called serialization.

Each row summarizes 3,400 timed samples (17 queries × 200). Percentiles are nearest-rank. No
sample or outlier was excluded. `model_load_ms` is null on every timed sample and is reported
only for the cold request below.

**Run 1 (search > dense > hybrid)**

| Endpoint | Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `search` | `request_ms` | 8.192 | 14.069 | 19.001 | 72.626 |
| `search` | `app_total_ms` | 4.662 | 8.193 | 12.052 | 46.221 |
| `search` | `lexical_ms` | 4.595 | 8.097 | 11.898 | 46.072 |
| `search` | `serialization_ms` | 0.037 | 0.073 | 0.109 | 0.455 |
| `dense` | `request_ms` | 25.899 | 39.114 | 52.991 | 369.159 |
| `dense` | `app_total_ms` | 20.696 | 32.235 | 44.614 | 357.409 |
| `dense` | `query_embedding_ms` | 14.044 | 23.474 | 30.843 | 344.369 |
| `dense` | `vector_ms` | 6.396 | 8.905 | 13.004 | 58.465 |
| `dense` | `serialization_ms` | 0.059 | 0.114 | 0.156 | 0.653 |
| `hybrid` | `request_ms` | 31.831 | 47.291 | 67.328 | 128.644 |
| `hybrid` | `app_total_ms` | 27.84 | 41.882 | 60.466 | 122.652 |
| `hybrid` | `lexical_ms` | 2.963 | 5.311 | 7.736 | 48.555 |
| `hybrid` | `query_embedding_ms` | 14.776 | 24.023 | 30.76 | 93.426 |
| `hybrid` | `vector_ms` | 3.902 | 5.97 | 7.573 | 59.014 |
| `hybrid` | `rrf_ms` | 0.778 | 1.636 | 2.241 | 3.344 |
| `hybrid` | `serialization_ms` | 0.076 | 0.151 | 0.19 | 0.89 |

**Run 2 (dense > hybrid > search)**

| Endpoint | Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `dense` | `request_ms` | 22.968 | 30.855 | 37.757 | 78.849 |
| `dense` | `app_total_ms` | 18.179 | 25.681 | 31.077 | 73.603 |
| `dense` | `query_embedding_ms` | 12.05 | 18.789 | 23.166 | 60.727 |
| `dense` | `vector_ms` | 5.919 | 7.481 | 8.918 | 43.739 |
| `dense` | `serialization_ms` | 0.053 | 0.106 | 0.143 | 0.769 |
| `hybrid` | `request_ms` | 28.378 | 37.423 | 46.122 | 100.46 |
| `hybrid` | `app_total_ms` | 24.704 | 32.909 | 40.651 | 96.62 |
| `hybrid` | `lexical_ms` | 2.9 | 4.301 | 5.085 | 12.445 |
| `hybrid` | `query_embedding_ms` | 12.075 | 19.016 | 23.159 | 78.852 |
| `hybrid` | `vector_ms` | 3.646 | 5.125 | 6.091 | 40.275 |
| `hybrid` | `rrf_ms` | 0.695 | 1.439 | 1.936 | 2.565 |
| `hybrid` | `serialization_ms` | 0.071 | 0.144 | 0.181 | 0.291 |
| `search` | `request_ms` | 9.313 | 13.881 | 16.424 | 51.489 |
| `search` | `app_total_ms` | 5.017 | 7.782 | 9.289 | 37.151 |
| `search` | `lexical_ms` | 4.942 | 7.621 | 9.126 | 36.993 |
| `search` | `serialization_ms` | 0.039 | 0.093 | 0.141 | 0.772 |

**Run 3 (hybrid > search > dense)**

| Endpoint | Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `hybrid` | `request_ms` | 27.941 | 38.211 | 48.039 | 94.51 |
| `hybrid` | `app_total_ms` | 24.367 | 33.741 | 42.423 | 90.724 |
| `hybrid` | `lexical_ms` | 2.828 | 4.309 | 5.464 | 50.62 |
| `hybrid` | `query_embedding_ms` | 12.047 | 19.628 | 23.762 | 74.699 |
| `hybrid` | `vector_ms` | 3.557 | 5.075 | 6.227 | 48.57 |
| `hybrid` | `rrf_ms` | 0.682 | 1.412 | 1.903 | 3.403 |
| `hybrid` | `serialization_ms` | 0.07 | 0.14 | 0.167 | 0.253 |
| `search` | `request_ms` | 9.249 | 12.355 | 15.248 | 61.63 |
| `search` | `app_total_ms` | 4.822 | 6.668 | 8.289 | 50.506 |
| `search` | `lexical_ms` | 4.743 | 6.553 | 8.19 | 50.466 |
| `search` | `serialization_ms` | 0.037 | 0.094 | 0.134 | 0.239 |
| `dense` | `request_ms` | 23.892 | 34.981 | 46.084 | 110.233 |
| `dense` | `app_total_ms` | 19.071 | 29.319 | 37.355 | 103.386 |
| `dense` | `query_embedding_ms` | 12.846 | 21.923 | 25.369 | 87.846 |
| `dense` | `vector_ms` | 6.04 | 8.175 | 10.976 | 50.218 |
| `dense` | `serialization_ms` | 0.054 | 0.109 | 0.135 | 0.206 |

* **Cold model load.** The model loaded on the first request in each run that needed it, the
  cold `laptop` request: `/search/dense` in runs 1 and 2 (`model_load_ms` 199.114 and 211.94;
  cold `request_ms` 253.4 and 275.675) and `/search/hybrid` in run 3 (`model_load_ms` 204.847;
  cold `request_ms` 289.488). Cold requests are excluded from the timed percentiles.
* **Run variance is kept.** Run 1 has the widest tails (for example hybrid `request_ms` p99
  67.328 versus 46.122 and 48.039 in runs 2 and 3, and a dense `query_embedding_ms` maximum of
  344.369). Runs are reported separately and not averaged.
* `lexical_ms` inside hybrid (p50 2.828 to 2.963) is lower than `/search` `lexical_ms` (p50
  4.595 to 4.942) in every run. The harness does not isolate the cause, and no explanation is
  claimed.
* No universal latency target is set or met: these are in-process numbers for one sequential
  client on one laptop with a 240-product catalog.

## 6. Verification and host validity

* **Host.** Windows 11 (10.0.26300), Intel64 Family 6 Model 140 (8 logical CPUs), Python
  3.12.3, PostgreSQL 16.15 with pgvector 0.8.6 in Docker on the same host. Both experiments ran
  on AC power with a keep-awake request held (recorded during execution; not a field of the
  artifacts). The artifacts record the Windows Kernel-Power query for each experiment window
  as available with 0 sleep/standby events.
* **Sampling gaps (Phase CL).** Maximum gap between consecutive recorded samples: 1.481 s
  (run 1), 0.779 s (run 2) and 0.878 s (run 3).
* **Raw counts.** Phase CL: per run and endpoint 17 cold and 3,400 timed samples (30,753
  samples in total), 153 result-list records, 3 run and 3 facts records, all `200`. Phase S:
  238 fused-list records (17 queries × 7 values × 2 runs), 34 source-list records, 2 run and 2
  facts records.
* **Warm-ups.** The 20 warm-ups per endpoint and query were executed by the committed harness
  control flow (with the same result-list check) but were not individually recorded in the raw
  JSONL, so their count cannot be independently reconstructed from the artifact.
* **Recomputation.** The committed `--verify` mode recomputed both summaries from their raw
  records and passed. A separate script that does not use the harness's derivation recomputed
  the Phase-CL figures with 0 mismatches across 8,354 compared values.
* **Isolation.** Every run used its own scratch database (`ecommerce_search_bench_*`) that the
  harness dropped afterwards; the development database was not touched. The model was loaded
  from the verified local snapshot with outbound network refused. The Phase-S artifacts were
  unchanged by Phase CL (same SHA-256 values as above).

## 7. Limitations

* 240 synthetic, templated products; one laptop; an in-process client; one sequential client;
  no concurrency. Nothing here predicts behaviour at larger scale or under load.
* No relevance labels and no Golden Dataset: the attribute-consistency proxy covers only
  explicit category, brand and `anc` expectations, and it saturated in Phase S.
* English-only embedding model: Hinglish such as `coding ke liye laptop` is not understood
  (query understanding is Milestone 6); its hybrid results come entirely from dense retrieval.
* Dense retrieval has no similarity threshold, so dense and hybrid always return neighbours,
  even for nonsense queries.
* `rrf_k = 100` is provisional, chosen by a tie rule; Milestone 10 re-decides it on the
  Golden Dataset. Source depths and `candidate_k` were not tuned.
* Lexical ties are broken by `product_id`, which feeds identifier order into the lexical ranks
  that RRF consumes.
