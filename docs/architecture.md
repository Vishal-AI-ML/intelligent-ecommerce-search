# Architecture

- Status: Milestone 0 design; implemented parts are marked per section (M1 foundation, M2 catalog, M3 lexical V0, M4 dense retrieval)
- Date: 2026-09-30
- Related: `docs/spec.md`, `docs/data-quality.md`, `docs/decisions/`

## 1. Principles

1. Deterministic extraction before inference.
2. Bounded, typed decisions instead of unrestricted generation.
3. Confidence gating instead of blind automation.
4. Human-reviewed labels instead of generated ground truth.
5. Measured improvement instead of assumed performance.
6. Simple architecture instead of unnecessary infrastructure. PostgreSQL is
   the single datastore for catalog, lexical search and vectors (ADR-001).
7. The core must work with no remote model. Remote providers are optional
   add-ons behind typed interfaces (ADR-003, ADR-005).

## 2. Component overview

```text
                    +-------------------------------+
  Buyer query  ---> |  FastAPI (GET/POST /search)   |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |  Query understanding          |
                    |  - deterministic parser       |
                    |  - DecisionProvider (ADR-003) |
                    |  - confidence gate            |
                    +---------------+---------------+
                                    |
             +----------------------+----------------------+
             |                                             |
  +----------v-----------+                     +-----------v----------+
  | Lexical retrieval    |                     | Dense retrieval      |
  | PostgreSQL FTS       |                     | local embeddings +   |
  |                      |                     | pgvector (ADR-004)   |
  +----------+-----------+                     +-----------+----------+
             |                                             |
             +----------------------+----------------------+
                                    |
                    +---------------v---------------+
                    | Safe structured filters       |
                    | RRF fusion (ADR-002)          |
                    | Optional cross-encoder rerank |
                    +---------------+---------------+
                                    |
                           ranked products
                            + telemetry
```

Shared components: taxonomy, attribute schema, brand dictionary,
normalization rules, Hinglish phrase dictionary, provider interfaces,
confidence policies and evaluation infrastructure.

## 3. Buyer search pipeline

```text
Query
 -> deterministic parser
 -> optional Jev bounded decisions
 -> confidence gate
 -> lexical + dense retrieval
 -> safe structured filters
 -> Reciprocal Rank Fusion
 -> optional cross-encoder reranker
 -> ranked products + telemetry
```

Notes:

- The diagram shows the logical order. Safe filters may be pushed into the
  lexical and dense SQL queries when that is safe and beneficial. The physical
  placement is decided and measured in Milestone 7.
- Hard-filter rule (`docs/spec.md` §7): explicit constraints parsed
  deterministically with high reliability may become hard filters. A
  model-derived decision may become a hard filter only after a confidence
  threshold is justified from human-reviewed data in Milestone 12. Before
  that, model-derived decisions may influence ranking or retrieval routing but
  not hard filtering.
- Initial candidate sizes (configurable): `lexical_k=50`, `dense_k=50`,
  `candidate_k=50`. Rerank the top 20 and return 10. `rrf_k` is configurable
  and has no initial value. Milestone 5 records a provisional value in
  configuration, marked provisional (ADR-002).
- Each result keeps its lexical rank, dense rank (either may be absent), fused
  score and, when reranking runs, reranker score.

### 3.1 Search versions

| Version | Pipeline | Milestone |
|---|---|---|
| V0 | Lexical (PostgreSQL FTS) | 3 |
| V1 | Lexical + dense + RRF | 5 |
| V2 | V1 + structured filtering | 7 |
| V3 | V2 + reranking | 8 |
| V4 | V3 + confidence-gated Jev decisions | 12 |

**Milestone 3 implemented V0** (`search_version = "v0_lexical"`): PostgreSQL FTS only, see
`docs/search-v0-baseline.md`. The search version is configurable and reported in every search response, so
versions can be compared on the same catalog and Golden Dataset.

**Milestone 4 added dense-only retrieval** (`search_version = "dense_only"`, `GET`/`POST
/search/dense`), which is a component, not a V-numbered version: V1 (lexical + dense + RRF) is
Milestone 5. See `docs/search-dense-m4.md`.

Evaluation order: Milestones 3 to 8 use provisional, non-authoritative smoke
queries. Human-reviewed labels arrive in Milestone 9, and Milestone 10 re-runs
V0 through V3 on the human-reviewed Golden Dataset. V4 is evaluated in
Milestone 12 and is enabled by default only if that evaluation shows a
measured benefit and acceptable safety, latency and cost. Provisional tuning
and authoritative evaluation are always reported separately
(`docs/spec.md` §9.1).

## 4. Seller listing pipeline (Level 2)

```text
Raw listing
 -> deterministic normalization
 -> attribute candidates
 -> optional Jev bounded decisions
 -> confidence/risk gate
 -> auto_accept | accept_with_warning | manual_review
 -> normalized catalog record
```

- Raw input is stored unchanged before any processing.
- Uncertain model names are never rewritten automatically.
- No automatic path produces a rejection. Rejection happens only through human
  review.

## 5. Data model

The concrete schema is implemented in Milestone 2. This section fixes the
design constraints, not the DDL.

### 5.1 Product (normalized catalog record)

- Core columns: `product_id`, `seller_id`, `title`, `description`,
  `category`, `subcategory`, `brand`, `price`, `currency`, `rating`,
  `review_count`, `availability`, `source_type`, `created_at`, `updated_at`.
- Frequently filtered attributes get proper typed columns (at minimum those
  used by V2 filters: `ram_gb`, `storage_gb`, `storage_type`, plus core
  `category`, `brand`, `price`).
- Other category attributes (`processor`, `gpu`, `screen_size_inches`,
  `operating_system`, `weight_kg`, `camera`, `battery_mah`, `size`, `color`,
  `material`, `gender`, `wireless`, `anc`, `battery_life_hours`,
  `connectivity`) are typed columns or validated JSONB, decided in
  Milestone 2 by filtering needs.
- Attributes that do not apply to a category stay null.
- `storage_type` taxonomy is an open question: NVMe is an interface/protocol
  and NVMe products are normally SSDs. Milestone 6 or 7 decides whether
  `storage_type` is SSD/HDD with a separate interface field, or NVMe is an SSD
  subtype. An explicit `ssd` query must not exclude NVMe SSDs
  (`docs/spec.md` §14, item 9).
- Provenance: `source_type`, plus the dataset reference and a synthetic flag
  (see `docs/data-quality.md`).

#### Milestone 2 implementation

Implemented tables (`src/ecommerce_search/models/catalog.py`, migration `0002`):
`catalog_datasets` (provenance, checksum, versions), `raw_catalog_records` (exact source line,
never updated), `products` (normalized core fields), `laptop_specs`, `phone_specs`,
`shoe_specs`, `headphone_specs` (typed attribute tables, 1:1 with a product, tied to the
product's category by a composite foreign key) and `catalog_reviews` (append-only human review
outcomes, each row carrying the reviewed `product_content_sha256` and the sample manifest
hash). All attributes are typed columns; there is no JSONB attribute bag. Dataset versions are
validated positive integers (CHECK) so ordering is numeric. Enumerated values
are TEXT plus named CHECK constraints, kept equal to `catalog/taxonomy.py` by tests.
`storage_type` and `storage_interface` are separate (see `docs/spec.md` §14, item 9). Milestone 2 created only
constraint-backing indexes (Milestone 3 adds one GIN search index, below). There are no
embedding or inferred-decision tables.

#### Milestone 3 implementation

`product_search_documents` (`models/search.py`, migration `0003`): one row per product with
`product_id` (PK, FK to `products`), `document_version`, `source_content_sha256` (the product's
`content_sha256` when built), a weighted `search_vector` (`tsvector`) and `built_at`, plus a GIN
index. It is a **derived, rebuildable** table (nothing in it is a source of truth) and there is no
search column on `products`. Documents are built by `search/documents.py` from validated catalog
fields only, written by ingestion inside the same transaction and by the explicit
`python -m ecommerce_search.search reindex --database NAME`. Writers take one advisory lock
(`search/indexing.py`), always after any dataset lock. Migration `0003` is schema only.

#### Milestone 4 implementation

`product_embeddings` (`models/embeddings.py`, migration `0004`): one row per product with
`product_id` (PK, FK to `products`), `embedding` (pgvector `vector(384)`), `model_id`,
`model_revision` (40-hex commit), `dimension`, `normalized`, `embedding_text_version`,
`embedding_config_sha256`, `source_content_sha256` (the product's `content_sha256` when
embedded), `embedding_text_sha256` and `embedded_at`. Named CHECK constraints enforce the
dimension, a non-zero vector and unit norm when flagged normalized. It is a **derived,
rebuildable** table generated only by the explicit
`python -m ecommerce_search.search embed --database NAME` (ingestion never loads the model).
Writers take the embedding advisory lock, always after any dataset or lexical-search lock, and
hold no product row locks. Only the primary-key index exists: no vector index (section 5.2).
Migration `0004` is schema only.

### 5.2 Search representations

- Lexical (**implemented in Milestone 3**): the weighted `tsvector` above, `simple` FTS
  configuration, `DOCUMENT_VERSION` "1" (a version bump is required when the builder, the
  section-to-weight assignment or the configuration changes), GIN index. Queries use
  `plainto_tsquery` (strict AND) and `ts_rank`.
- Dense (**implemented in Milestone 4**): the `vector(384)` column above with the pinned
  `sentence-transformers/all-MiniLM-L6-v2` model (ADR-006). Each row records the model name,
  immutable revision, dimension, normalization, embedding-text version, configuration hash,
  source content hash and generation timestamp (`embedded_at`). Catalog/dataset identity is not
  repeated on each row: a current row's content belongs to the dataset version recorded on its
  product, and experiments and database audits record the dataset id, version and checksum
  (`docs/data-quality.md` §7). Freshness is content-addressed: rows built from older product
  content or another model/configuration/text version are excluded from retrieval. Retrieval is
  an **exact** cosine-distance scan; a measured benchmark (ADR-006,
  `docs/search-dense-m4.md`) found no benefit from an HNSW index at 240 products, so no
  persistent ANN index exists. It must be re-evaluated at materially larger catalog sizes.
- Embedding and search text never contains evaluation labels.

### 5.3 Separation of raw, normalized, inferred and human values

**Provisional:** the mutability column is a proposed design. It is finalized
in Milestone 2 (catalog schema) and Milestone 13 (listing normalization).

| Layer | Content | Mutability (provisional) |
|---|---|---|
| Raw | Seller input exactly as received | Immutable |
| Normalized | Deterministic normalization output, linked to raw | Regenerated when rules change (versioned) |
| Inferred | Model decisions with provider, model version, confidence, timestamp | Append-only |
| Risk signals | Triage signals with criteria version | Append-only |
| Human outcomes | Reviewer decisions and labels with reviewer and timestamp | Append-only. Authoritative |

Milestone 2 implements the raw layer (`raw_catalog_records`), the normalized layer (`products`
and the spec tables) and the human-outcome layer for the catalog review sample
(`catalog_reviews`, append-only via triggers). The inferred and risk-signal layers are not
implemented yet.

### 5.4 Query understanding

`QueryUnderstanding` holds typed optional fields: `raw_query`, `category`,
`brand`, `ram_gb`, `storage_gb`, `storage_type`, `min_price`, `max_price`,
`semantic_intent`, `retrieval_strategy`, `attributes`. Unset means unknown.
Unknown is never turned into a guessed value.

### 5.5 Evaluation and experiment records

Stored under `evals/` (created in a later milestone), separate from application
code: golden queries, human labels, decision labels, experiment records and
generated reports. The fields each experiment record carries are listed in
`docs/spec.md` §9.

## 6. Decision layer

- `DecisionProvider` interface (ADR-003), conceptually
  `understand(query, deterministic_result) -> DecisionResult`.
- Implementations: `DeterministicDecisionProvider` (default, local),
  `JevDecisionProvider` (optional, Milestone 11, ADR-005), and an optional
  `LLMDecisionProvider` used only as an experimental comparison.
- `DecisionResult` is validated with Pydantic. Each decision is one of a fixed
  set of bounded decision types with a closed label set, a raw provider
  probability and a gate confidence.
- Raw provider probability is provider output. Gate confidence is the value
  the confidence policy consumes. They are separate fields and are logged
  separately. Jev's raw probability must not become gate confidence until its
  meaning and calibration are verified in Milestone 12 (ADR-003, ADR-005).
- The confidence gate sits outside providers. Providers propose and the gate
  decides. Gate thresholds are configuration, experimental until
  human-reviewed data exists.
- Milestone 11 implements the optional Jev provider, Milestone 12 evaluates
  it, and it is enabled in the default/V4 configuration only if Milestone 12
  shows a measured benefit and acceptable safety, latency and cost.
- Failure handling: timeout, transport error, schema-invalid output,
  out-of-set label or disabled provider all lead to the deterministic result,
  and the fallback reason is recorded.

## 7. Configuration and secrets

- Typed configuration via Pydantic settings from environment variables
  (Milestone 1).
- Configurable values include: search version; candidate sizes
  (`lexical_k`, `dense_k`, `candidate_k`, `rrf_k`, rerank depth, result count);
  embedding model and revision; reranker model and on/off; decision provider
  enablement, endpoint, model, timeout, retries and thresholds.
- Milestone 4 settings: `SEARCH_DENSE_K` (maximum dense `top_k`), `EMBEDDING_MODEL_ID` and
  `EMBEDDING_MODEL_REVISION` (must equal a reviewed registry entry; the default is the ADR-006
  pin), `EMBEDDING_MODELS_DIR` (default: the git-ignored repository `models/`; the `api`
  container uses its read-only `/models` mount) and `EMBEDDING_BATCH_SIZE`.
- Credentials only in environment variables. `.env` is git-ignored and only
  `.env.example` may be committed. Secrets are never logged.

## 8. Telemetry

- Stage timings: `query_understanding_ms`, `jev_decision_ms`, `lexical_ms`,
  `vector_ms`, `filtering_ms`, `rrf_ms`, `reranking_ms`, `serialization_ms`,
  `total_ms`. Aggregated as P50, P95 and P99.
- Response metadata: search version, provider, reranker state, latency,
  result count, applied safe filters.
- Milestone 4 dense responses report `model_load_ms` (only when the model loaded in that
  request), `query_embedding_ms`, `vector_ms` and `total_ms`. `serialization_ms` is measured by
  the benchmark harness (timing `model_dump_json()` separately), as in M3, not by the API.
- Provider telemetry: provider, pinned model version, latency, raw provider
  probabilities and gate-confidence values (separate fields), fallback reason
  and returned cost where available.
- Full observability (request IDs, structured logs, aggregate metrics) is
  Milestone 17.

## 9. API surface

| Endpoint | Milestone |
|---|---|
| `GET /health` | 1 |
| `GET /search?q=...`, `POST /search` | 3 (V0 lexical implemented; later milestones add response fields as they become real) |
| `GET /search/dense?q=...`, `POST /search/dense` | 4 (dense-only retrieval implemented; not a V-numbered version) |
| `POST /listings/analyze`, `POST /listings/normalize` | 13–14 (provisional) |
| `GET /reviews/pending`, `POST /reviews/{review_id}/decision` | Listing milestones (provisional) |

The listing endpoint milestone mapping is provisional. The master plan does
not assign endpoints to milestones, and the mapping is confirmed when
Milestones 13 to 15 are planned.

## 10. Repository layout

Directories are created only when their milestone begins. The intended layout
is in `docs/MASTER_PLAN.md` §17. Milestone 0 adds only `docs/*.md` and
`docs/decisions/`.

## 11. Deployment

- Local development: Docker Compose with PostgreSQL + pgvector (Milestone 1).
- Milestone 4: model weights are never copied into the image; Compose mounts the repository
  `models/` directory read-only into the `api` service only (`./models:/models:ro`). Without a
  snapshot, dense search returns the fixed 503 and lexical search and `/health` keep working
  (verified in Docker; `docs/search-dense-m4.md` §7).
- AWS (ECR, ECS/Fargate, PostgreSQL with pgvector, Secrets Manager,
  CloudWatch, Terraform) only in Milestone 19, after local stability and with
  approval before chargeable actions. No EKS.

## 12. Architecture decision records

| ADR | Decision |
|---|---|
| [ADR-001](decisions/ADR-001-postgresql-pgvector.md) | PostgreSQL + pgvector as the single search datastore |
| [ADR-002](decisions/ADR-002-hybrid-retrieval-rrf.md) | Hybrid lexical + dense retrieval fused with RRF |
| [ADR-003](decisions/ADR-003-decision-provider.md) | `DecisionProvider` abstraction with a deterministic default |
| [ADR-004](decisions/ADR-004-local-embedding-strategy.md) | Configurable local embedding models |
| [ADR-005](decisions/ADR-005-jev-bounded-decision-provider.md) | Jev as an optional bounded-decision provider |
| [ADR-006](decisions/ADR-006-embedding-model-selection.md) | MiniLM embedding model (pinned revision) and exact scan at the current scale |
