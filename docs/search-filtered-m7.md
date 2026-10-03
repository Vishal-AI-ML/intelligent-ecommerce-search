# Milestone 7: structured filtering (V2)

- Status: Milestone 7 implemented. `GET`/`POST /search/filtered`, `search_version =
  "v2_filtered"`, filter policy `fp-1` (`FILTER_POLICY_VERSION`), fusion unchanged from V1 (`rrf`,
  `rrf_k = 100`, `rrf_k_status = "provisional"`).
- Related: `docs/spec.md` FR-RET-4, FR-API-2, §7 and §14 (Q5, Q9), `docs/architecture.md` §3.1,
  `docs/query-understanding-m6.md` (the only filter source), `docs/search-hybrid-m5.md` (the
  unchanged V1 retrieval and fusion), ADR-002, ADR-003.
- **No relevance, quality, Constraint Satisfaction Rate or extraction-accuracy claim is made.**
  The catalog is the **synthetic 240-product** seed (80 laptops, 60 phones, 50 shoes, 50
  headphones). The tests and the demo below show deterministic filter correctness and contract
  conformance: that the filters which are applied are exactly the ones fp-1 prescribes and that
  every returned product satisfies them. They do not show that V2 returns better products than
  V1. Golden Dataset labelling and the V0–V3 evaluation are Milestones 9 and 10. **No M7 latency
  benchmark was run**, so no latency figure is reported here.

## 1. Scope and relationship to earlier versions

| Endpoint | Version | M7 change |
|---|---|---|
| `GET`/`POST /search` | V0 `v0_lexical` | none |
| `GET`/`POST /search/dense` | `dense_only` (component) | none |
| `GET`/`POST /search/hybrid` | V1 `v1_hybrid` | none (its M6 `query_understanding` block stays `usage: "informational"` and `applied_filters` stays `[]`) |
| `GET`/`POST /search/filtered` | V2 `v2_filtered` | **new, additive** |

* V2 = V1 + structured filtering. It reuses the V1 lexical and dense statements, the V1 snapshot
  transaction and the unchanged `fuse_rrf`.
* **M6 query understanding is the only filter source.** The request has no explicit filter
  parameters: `FilteredSearchRequest` has only `query` and `top_k` (`extra="forbid"`), and the GET
  route only `q` and `top_k`.
* **Retrieval text is unchanged.** The normalized query string goes to `plainto_tsquery` and to
  the embedder exactly as in V1; parsed terms are never stripped or rewritten.
* No migration, index, setting, dependency, Docker or Compose change. The migration head is
  still `0004`.

## 2. Request flow (shared GET/POST path, `src/ecommerce_search/api/filtered.py`)

```text
normalize the query and validate top_k (1..SEARCH_CANDIDATE_K, default 10)   -> 422 on failure
 -> deterministic parse + DecisionProvider + validation (query_understanding_ms)
    any failure: fixed 503, reason query_understanding_failed
 -> derive_filters(decision) under fp-1, pure (filter_translation_ms)
    any failure: fixed 503, reason filter_policy_failed
 -> no letter or digit: 200, no results, no model, no SQL (query_understanding still reported)
 -> load the model if needed, token check, encode the query (no DB connection held)
 -> one REPEATABLE READ READ ONLY transaction:
      filtered lexical SQL (LIMIT SEARCH_LEXICAL_K) + filtered dense SQL (LIMIT SEARCH_DENSE_K)
 -> transaction ends
 -> unchanged RRF over the two filtered lists, keep candidate_k = 50, return the first top_k
```

Query understanding and filter translation run once per request, before any model or database
work, so a parse or policy failure checks out no connection.

## 3. Filter policy `fp-1` (`src/ecommerce_search/filtering/policy.py`)

`derive_filters` is pure (no I/O, settings, database, model, clock, randomness or logging) and
returns a frozen `FilterDerivation`: a closed `FilterSpec`, the `applied_filters` derived from it
and the `ignored_constraints`.

* **Source gate.** Only provider `deterministic` at `provider_version = "qu-1"`, parser `qu-1`
  and lexicon `1` may create filters. Model-derived decisions cannot be hard filters before
  Milestone 12 (`docs/spec.md` §7). Anything else is a `FilterPolicyError` (→ fixed 503).
* **Evidence gate.** A parsed field becomes a filter only when it is non-null and a matched term
  carries that field with the same canonical value. A non-null field without such evidence is a
  contract violation (`FilterPolicyError`), never applied silently.
* **Filterable fields** (fixed order): `category`, `brand`, `ram_gb`, `storage_gb`,
  `storage_type`, `storage_interface`, `min_price`, `max_price`.

| Field | Operator | Matching (SQL) |
|---|---|---|
| `category` | `eq` | `products.category` exact (`laptop`, `phone`, `shoes`, `headphones`) |
| `brand` | `eq` | `products.brand` exact, canonical brand-dictionary spelling |
| `ram_gb` | `eq` | exact on a `laptop_specs` **or** `phone_specs` row of the product |
| `storage_gb` | `eq` | exact on a `laptop_specs` **or** `phone_specs` row of the product |
| `storage_type` | `eq` | exact on the product's `laptop_specs` row (`SSD`, `HDD`) |
| `storage_interface` | `eq` | exact on the product's `laptop_specs` row (only `NVME` is ever produced) |
| `min_price` | `gte` | `products.price >= min_price` (inclusive) |
| `max_price` | `lte` | `products.price <= max_price` (inclusive) |

* **Exact equality, no tolerance.** `8gb ram` matches `ram_gb = 8` only; a RAM value never matches
  storage and vice versa. `1tb` is `1024` GB (M6 normalization).
* **SSD and NVMe.** `ssd` filters only the medium, so NVMe and SATA SSDs both match (FR-QU-4).
  `nvme` gives `storage_type = SSD` **and** `storage_interface = NVME`. Phones have no storage
  type, so `phone ssd` matches nothing.
* **Unknown never satisfies.** A product without a spec row, or with a NULL spec value, does not
  satisfy a spec filter. It still satisfies non-spec filters (category, brand, price).
* **Conflicts.** A field named by any parser conflict is not applied (`conflict`). A conflicted
  `storage_type` also blocks the NVMe interface that depends on it.
* **Family-scoped ambiguity suppression** (`ambiguous_family`). A capacity ambiguity
  (`bare_capacity`, `ambiguous_memory_capacity`, `capacity_multiple_roles`) blocks both `ram_gb`
  and `storage_gb`; a price ambiguity (`price_without_bound`) blocks both `min_price` and
  `max_price`; `unsupported_grouped_number`, `unsupported_range`, `unsupported_numeric_compound`
  and `out_of_range_value` block both numeric families. Category, brand and storage type are
  never blocked by a numeric ambiguity. A suppressed field is reported only when the parser had
  set it.
* **Informational only.** `semantic_intent` and `attributes.price_preference` (`sasta`) are never
  filters; when present they are reported as `informational_only`.
* **Literal brand/category pairs.** `nike laptop` applies `category = laptop` and `brand = Nike`
  even though no such product exists; the answer is zero results, not a guess.
* Any rule change requires a new `FILTER_POLICY_VERSION`.

## 4. Filtered retrieval (`src/ecommerce_search/search/filtered.py`)

* **In the SQL, before ranking and LIMIT.** Each active field adds one constant fragment from
  the closed `FILTER_FRAGMENTS` table to the WHERE clause of the existing V0 lexical / M4 dense
  statement. Spec filters are correlated `EXISTS` subqueries, so a product is never duplicated.
  ORDER BY, tie-breaks, scores and `LIMIT` are those of the existing statements; ranks are
  positional within the filtered list. With an empty `FilterSpec` the statements are exactly the
  existing V1 statements.
* **One table for both sources.** Lexical and dense use the same fragments, so they share the
  same filter semantics.
* **Bound values only.** Values are always named bind parameters (prices as `Decimal`); the
  statement text depends only on which fields are set.
* **No post-filtering, retry, relaxation or fallback.** There is no Python post-filter, no second
  query without filters and no fallback to V1. An empty list is a result.
* **One snapshot.** `read_filtered_sources` runs both reads in one `REPEATABLE READ READ ONLY`
  transaction (the same `SNAPSHOT_SQL` as V1) and refuses a session that already has an open
  transaction. It commits on success, rolls back on error and releases the connection either way.
* **Dense freshness unchanged.** The current-embedding predicates of M4 are kept, so missing or
  stale embeddings only shorten the dense list.
* **Fusion unchanged.** `fuse_rrf` with equal weights, 1-based positional ranks, exact-fraction
  scores, `product_id` tie-break and `rrf_k = 100` (**provisional**, ADR-002; re-decided in
  Milestone 10). `candidate_k` stays 50.

## 5. API contract (`FilteredSearchResponse`)

```bash
curl "http://127.0.0.1:8000/search/filtered?q=hp+laptop+8gb+ram&top_k=5"
curl -X POST http://127.0.0.1:8000/search/filtered -H "Content-Type: application/json" -d '{"query": "laptop 8gb 256 ssd", "top_k": 5}'
```

Fields shared with V1: `query`, `document_version`, `tsquery`, `embedding_model_id`,
`embedding_model_revision`, `embedding_dimension`, `embedding_text_version`, `distance_metric`,
`fusion`, `dense_status`, `lexical_hit_count`, `dense_hit_count` (both counted **after**
filtering), `overlap_count`, `fused_count`, `candidate_count`, `top_k`, `result_count` and
`results` (the V1 `HybridSearchResult` shape with both source ranks and scores).

V2-specific fields:

* `search_version`: `"v2_filtered"`.
* `filter_policy`: `{"version": "fp-1"}`.
* `applied_filters`: list of `{"field", "operator", "value"}` in the fixed field order; operator
  `eq`, `gte` or `lte`; value the M6 canonical string (`"laptop"`, `"HP"`, `"8"`, `"SSD"`,
  `"NVME"`, prices with two decimals such as `"40000.00"`).
* `ignored_constraints`: list of `{"field", "reason"}` with reason `conflict`,
  `ambiguous_family` or `informational_only`, in M6 field order.
* `query_understanding`: `{"usage": "filter_source", "provider": "deterministic",
  "provider_version": "qu-1", "understanding": {...}}`.
* `latency_ms`: `lexical_ms`, `model_load_ms`, `query_embedding_ms`, `vector_ms`, `rrf_ms` (null
  when the stage did not run), `query_understanding_ms`, `filter_translation_ms` and `total_ms`
  (always present on 200). Filtering runs inside the source SQL, so its cost is part of
  `lexical_ms` and `vector_ms`; there is no separate `filtering_ms`.

**Zero results.** When no product satisfies the applied filters the response is `200` with
`results = []`, `candidate_count = 0` and `applied_filters` exactly as derived. It is never
relaxed.

**Status codes.**

| Status | When |
|---|---|
| `422` | blank, over-long or unsafe query; `top_k` outside 1..`SEARCH_CANDIDATE_K`; unknown POST field; query over the model token limit (standard validation body, never echoing the input) |
| `503` `{"detail": "search unavailable"}` | query understanding failed (`query_understanding_failed`), filter translation failed (`filter_policy_failed`), model unavailable (`snapshot_missing`, `load_failed`, `dimension_mismatch`, `config_mismatch`, `invalid_output`, `encode_failed`), missing table (`table_missing`) or any other SQLAlchemy error (`database_error`). The log records only the fixed reason code and, where one exists, a fixed operator hint |
| `500` | any other, unexpected exception (for example a programming error during the read, or a malformed source list rejected by `fuse_rrf`): the framework's generic `Internal Server Error`, not disguised as 503; the transaction is rolled back or already ended and the connection released |

## 6. Evidence

Commits on `main`:

* `f1be5abff97736ec91960aec5874801ce3f2ccc8` `feat(m7): add deterministic filter policy fp-1`
* `a5cc37454f7fd9886fbc14909355ee1b91f4040b` `feat(m7): add filtered lexical and dense retrieval`
* `3d2bc22fc8c0d09d9b213e6b2e2b37ec3c352a10` `feat(m7): expose V2 filtered hybrid search`

Tests (developer-written; expectations predeclared by hand from fp-1 and the seed listing):

* `tests/unit/search/test_filter_policy.py`: the fp-1 expectation table, contract violations,
  determinism, ambiguity-reason classification and import isolation.
* `tests/unit/search/test_filtered_sql.py`: the closed fragment table, identical fragments for
  both sources, statement text independent of values, bound parameters, and empty filters
  reproducing the existing statements.
* `tests/integration/filter_support.py`: a 46-row truth table (26 parsed queries, 20 direct
  `FilterSpec` boundary rows) and an independent catalog oracle (plain `SELECT`s plus Python
  predicates), over a deliberately damaged scratch catalog (deleted spec rows, NULL spec values,
  one stale and one missing embedding).
* `tests/integration/test_filtered_retrieval.py`: SQL eligibility equals the oracle for every row
  and both sources, filtered results are the oracle-filtered prefix of the unfiltered order,
  `EXPLAIN` places filter predicates below the `LIMIT`, and both reads share one snapshot.
* `tests/unit/search/test_filtered_api.py`, `tests/integration/test_filtered_api_integration.py`:
  the API contract, GET/POST parity, exact-fraction RRF oracle, empty-filter V2 = V1, the
  422/503/500 boundaries, no retry and zero connection checkouts on a policy failure.

Validation reused from M7 Session F on the production tree `3d2bc22` (not re-run in Session H,
which changed no executable line): 2191 unit tests passed (`not integration and not real_model`),
641 integration tests passed (`integration and not real_model`), 5 `real_model` tests passed
offline.

### 6.1 Scratch-database demo (Session H)

A throwaway database was migrated to `0004`, seeded with the 240-product catalog, given fake
embeddings (`FakeEmbedder`, not MiniLM) and dropped afterwards. Every check below passed (0
failures). This is contract and filter-correctness evidence, **not relevance**: with a fake
embedder the dense order means nothing.

* All 26 query-backed truth-table rows were sent through GET and POST `/search/filtered` (after
  the committed damage was applied): GET/POST bodies were equal without `latency_ms`; applied
  filters equalled the committed expectations; ignored constraints equalled the committed
  expectations wherever one exists (24 rows; the other two, `hp laptop 8gb ram` and
  `hdd laptop`, returned none); every returned product satisfied the independent catalog
  predicate; and the source lists, counts and exact-fraction RRF ranks and scores were
  recomputed independently from the **unfiltered** V0/M4 reads filtered in Python. The 20 direct
  `FilterSpec` rows (price boundaries, RAM-versus-storage, missing spec rows) are not API queries
  and rely on the committed retrieval tests above.
* Empty-filter queries (`nvme hdd`, `above 30k under 20k`, `memory 8gb`, `zzqxv`,
  `comfortable gift`, `!!!`) returned the V1 body exactly, apart from the version, filter and
  query-understanding fields.
* `/search`, `/search/dense` and `/search/hybrid` returned exactly the direct V0, dense and RRF
  results and carry no V2 fields; `/search/hybrid` still reports `usage: "informational"` and
  `applied_filters: []`.

Examples on the undamaged seed catalog (`top_k = 50`; counts are lexical / dense hits after
filtering):

| Query | `applied_filters` | `ignored_constraints` | Eligible products | Lexical / dense | Results |
|---|---|---|---|---|---|
| `hp laptop` | category `laptop`, brand `HP` | — | 12 | 12 / 12 | 12 |
| `hp laptop 8gb ram 256gb ssd under 40k` | category, brand, ram `8`, storage `256`, type `SSD`, max `40000.00` | — | 1 | 0 / 1 | 1 |
| `laptop 8gb 256 ssd` | category, ram `8`, storage `256`, type `SSD` | — | 15 | 0 / 15 | 15 |
| `coding ke liye laptop` | category `laptop` | `semantic_intent` informational_only | 80 | 0 / 50 | 50 |
| `sasta phone` | category `phone` | `price_preference` informational_only | 60 | 0 / 50 | 50 |
| `ssd` | type `SSD` | — | 68 | 50 / 50 | 50 |
| `nvme` | type `SSD`, interface `NVME` | — | 55 | 50 / 50 | 50 |
| `above 30k under 20k` | — | `min_price`, `max_price` conflict | 240 | 0 / 50 | 50 |
| `16gb ram laptop ₹40,000` | category `laptop` | `ram_gb` ambiguous_family | 80 | 0 / 50 | 50 |
| `nike laptop` | category `laptop`, brand `Nike` | — | 0 | 0 / 0 | 0 |
| `from 20k ke andar` | min `20000.00`, max `20000.00` | — | 0 | 0 / 0 | 0 |
| `!!!` | — | — | (no retrieval) | — / — | 0 |

The `nike laptop` and `from 20k ke andar` rows show valid filters with no eligible product: a
`200` with no results and the filters still reported. The zero lexical counts show the strict-AND
limitation below.

## 7. Limitations

* **Strict-AND lexical text.** The lexical query is still `plainto_tsquery` over the full,
  unchanged text, so words such as `under`, `40k`, `ke`, `liye` or `sasta` must also appear in
  the document. Filtered queries with such words often get no lexical candidates and are served
  by the dense side alone (see the table). This is the V0 behaviour, not a filter effect; M7
  does not strip parsed terms.
* **Missing or stale embeddings** silently shorten the dense list, as in V1. Combined with the
  strict-AND lexical side, an eligible product with a stale embedding can be missing from the
  results (the truth-table `full` row demonstrates this on the damaged catalog). Check
  `embed-status --require-current` before relying on completeness.
* **Dense has no threshold.** Within the filtered set the dense side returns up to `dense_k`
  products for any query, so V2 can still return weak matches that satisfy the filters.
* **Coverage is the M6 `qu-1` lexicon.** Anything the parser leaves unresolved, ambiguous or
  conflicting is not filtered; ranges such as `20k to 40k`, grouped numbers and unbound
  capacities apply no numeric filter.
* **Missing or NULL catalog data excludes a product** under spec filters even if the real
  product would match.
* **No explicit filters** and no model-derived filters (Milestone 12 at the earliest).
* No relevance, Constraint Satisfaction Rate or zero-result-rate figure exists before the Golden
  Dataset (Milestones 9 and 10); no M7 latency benchmark was run; the exact dense scan and the
  decision to add no index for the filter columns are unmeasured choices at 240 products and
  must be re-evaluated when the catalog grows or a latency measurement shows a need.
