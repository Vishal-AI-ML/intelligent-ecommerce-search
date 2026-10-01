# Specification — Intelligent E-commerce Search & Decision Engine

- Status: Milestone 0 specification
- Date: 2026-09-30
- Primary source: `docs/MASTER_PLAN.md`
- Related: `docs/architecture.md`, `docs/data-quality.md`, `docs/decisions/`

This document states what the system must do and how completion is judged. It
does not describe implemented behavior; nothing described here exists in code
yet. Where `docs/MASTER_PLAN.md` and this document disagree, the master plan
wins and this document must be corrected.

## 1. Problem statement

Marketplace listings are inconsistent across sellers: misspellings, missing
attributes, conflicting units, misleading claims, policy risks and duplicates.
Buyers submit short queries that mix category, brand, exact specifications,
price and semantic intent, for example:

```text
laptop 8gb 256 ssd
hp laptop under 50k
coding ke liye laptop
nike shoes under 5k
anc headphones
iphone 15
```

A purely semantic retriever can return products that look similar but violate
explicit constraints. For `laptop 8gb 256 ssd`, RAM, storage and storage type
are explicit constraints. For `coding laptop`, "coding" is semantic intent.

The system therefore combines deterministic query understanding, lexical and
dense retrieval, safe structured filtering, Reciprocal Rank Fusion, optional
reranking and, later, optional confidence-gated bounded model decisions, all
measured against human-reviewed evaluation data.

## 2. Goals

1. Return relevant products for short, mixed-intent buyer queries, including
   common Hinglish phrasing.
2. Respect explicit, reliably parsed constraints (category, brand, model, RAM,
   storage, storage type, price).
3. Treat vague or subjective intent (gaming, coding, student, lightweight,
   premium, comfort, travel, office use) as ranking signals, not filters,
   unless reliable structured data supports filtering.
4. Make every retrieval stage measurable: quality metrics per search version
   and per-stage latency at P50, P95 and P99.
5. Run fully locally without any remote model provider.
6. Keep every model-driven decision bounded, typed, confidence-gated and
   reversible by a human.
7. Preserve raw seller data separately from normalized, inferred and
   human-reviewed values.

## 3. Non-goals

- A generic RAG chatbot or an unrestricted LLM wrapper.
- Free-form generated answers, product descriptions or recommendations.
- Using any model to invent attributes, produce ground-truth labels, make final
  legal conclusions, or automatically reject listings or ban sellers.
- Paid embedding APIs as a core dependency.
- Elasticsearch, OpenSearch, Qdrant, Pinecone, Kafka, Airflow, Kubernetes, or
  multiple databases for the same search workload, unless measurements prove
  PostgreSQL insufficient.
- A universal latency promise not backed by measurement.
- Frontend, AWS or Terraform work before Milestones 18 and 19.
- Multi-currency conversion, personalization, and learning-to-rank from user
  behavior. None of these are in the master plan.

## 4. Scope levels

| Level | Content | Gate |
|---|---|---|
| 1 — Core search engine | Catalog, catalog quality, PostgreSQL FTS, local embeddings + pgvector, hybrid retrieval with RRF, deterministic query understanding, structured filtering, cross-encoder reranking, human-reviewed Golden Dataset, evaluation, FastAPI, tests, measured latency | Portfolio-complete when done |
| 2 — Marketplace decision layer | Jev decision provider, query classification, retrieval-strategy routing, ambiguity and out-of-catalog detection, listing-category classification, attribute-candidate validation, policy-risk triage, confidence-based review routing | Only after Level 1 is stable |
| 3 — Production extensions | Regression CI, observability, frontend, production Docker, AWS, Terraform | Only after evaluation is stable |

## 5. Actors and workflows

- **Buyer:** submits a search query and receives ranked products plus search
  metadata.
- **Seller:** submits raw listings (Level 2). Seller content is untrusted input.
- **Human reviewer:** labels search relevance, reviews the catalog sample,
  labels decision data and resolves listings routed to manual review. Only a
  human produces authoritative labels.
- **Operator/developer:** runs ingestion, evaluation and experiments, and
  chooses configuration from measured results.

## 6. Functional requirements

### 6.1 Product catalog

- FR-CAT-1: Core fields: `product_id`, `seller_id`, `title`, `description`,
  `category`, `subcategory`, `brand`, `price`, `currency`, `rating`,
  `review_count`, `availability`, `source_type`, `created_at`, `updated_at`.
- FR-CAT-2: Category attributes:
  - Laptop: `ram_gb`, `storage_gb`, `storage_type`, `processor`, `gpu`,
    `screen_size_inches`, `operating_system`, `weight_kg`.
  - Phone: `ram_gb`, `storage_gb`, `camera`, `battery_mah`,
    `screen_size_inches`, `operating_system`.
  - Shoes: `size`, `color`, `material`, `gender`.
  - Headphones: `wireless`, `anc`, `battery_life_hours`, `connectivity`.
- FR-CAT-3: Irrelevant category attributes are left empty, never populated
  with defaults.
- FR-CAT-4: Frequently filtered fields have proper columns. Validated
  long-tail attributes may use JSONB.
- FR-CAT-5: Every record carries provenance, including whether it is
  synthetic (see `docs/data-quality.md`).
- FR-CAT-6: Catalog quality checks run before any evaluation.

### 6.2 Query understanding

- FR-QU-1: A typed `QueryUnderstanding` model with optional fields
  `raw_query`, `category`, `brand`, `ram_gb`, `storage_gb`, `storage_type`,
  `min_price`, `max_price`, `semantic_intent`, `retrieval_strategy`,
  `attributes`.
- FR-QU-2: A deterministic parser runs first and is always available.
- FR-QU-3: Supported numeric forms include `8gb`, `8 gb`, `16gb`, `256gb`,
  `512gb`, `1tb`, `40k`, `₹40000` and `rs 40000`. `40k` normalizes to `40000`
  and `1tb` normalizes to `1024` GB.
- FR-QU-4: Supported terms include RAM, memory, storage, SSD, NVMe and HDD.
  How `storage_type` and NVMe relate is an open question (section 14, item 9).
  Whatever taxonomy is chosen, an explicit `ssd` query must not exclude NVMe
  SSDs.
- FR-QU-5: Hinglish support is defined in section 7.
- FR-QU-6: The parser never invents a missing constraint. An absent value
  stays `null`.
- FR-QU-7: The parser stays small, typed and unit-tested. It must not grow
  into hundreds of fragile regular expressions.

### 6.3 Retrieval and ranking

- FR-RET-1 (V0): Lexical retrieval with PostgreSQL Full Text Search over
  brands, model names, exact terms and specifications, with suitable indexes.
- FR-RET-2: Dense retrieval with a configurable local
  sentence-transformers-compatible embedding model stored in pgvector.
- FR-RET-3 (V1): Hybrid retrieval that fuses lexical and dense candidate lists
  with Reciprocal Rank Fusion. Initial configurable values: `lexical_k=50`,
  `dense_k=50`, `candidate_k=50`. The RRF constant `rrf_k` is configurable,
  and its initial value stays unspecified until measured evaluation
  (see ADR-002). Milestone 5 selects a provisional value, records it in
  configuration and marks it provisional (section 9.1). Source ranks and the
  fused score are preserved per result.
- FR-RET-4 (V2): Safe structured filtering on category, brand, RAM, storage,
  storage type, minimum price and maximum price.
- FR-RET-5 (V3): Optional configurable cross-encoder reranking with on/off
  modes. Initial plan: 50 fused candidates, rerank 20, return 10. Benchmarks
  compare no reranker, rerank 10 and rerank 20.
- FR-RET-6 (V4): Confidence-gated Jev decisions (Level 2 only).

### 6.4 API

- FR-API-1: `GET /health`, `GET /search?q=...` and `POST /search`, with typed
  requests and responses and OpenAPI documentation.
- FR-API-2: Search responses include metadata: search version, decision
  provider, reranker state, latency, result count and the safe filters that
  were actually applied.
- FR-API-2 status (Milestone 3): the V0 response carries only metadata that exists and is
  truthful: `search_version`, `document_version`, requested `top_k`, `result_count` (results in
  this response, not total matches), the generated `tsquery`, `applied_filters` (always empty in
  V0) and `latency_ms` (`lexical_ms`, `total_ms`). Decision provider, reranker state and parsed
  query are **not** returned yet; they are added when those components exist (additive, no
  placeholders).
- FR-API-3: Listing endpoints (`POST /listings/analyze`,
  `POST /listings/normalize`, `GET /reviews/pending`,
  `POST /reviews/{review_id}/decision`) are deferred to their milestones.
  **Provisional:** the milestone mapping for these endpoints
  (`docs/architecture.md` §9) is not fixed by the master plan and is confirmed
  when the listing milestones (13 to 15) are planned.

### 6.5 Decision providers

- FR-DEC-1: A `DecisionProvider` interface conceptually shaped as
  `understand(query, deterministic_result) -> DecisionResult` (ADR-003).
- FR-DEC-2: `DeterministicDecisionProvider` is the default and needs no
  network access.
- FR-DEC-3: Provider outputs are validated with Pydantic. Invalid, late or
  failed outputs fall back safely to the deterministic result.
- FR-DEC-4: Jev is an optional bounded-decision provider (ADR-005).
  Milestone 11 implements it, Milestone 12 evaluates it, and it is enabled in
  the default/V4 configuration only if Milestone 12 demonstrates a measured
  benefit and acceptable safety, latency and cost.
- FR-DEC-5: Raw provider probability and gate confidence are distinct values.
  The raw provider probability is provider output. Gate confidence is the
  value consumed by the confidence policy. A provider's raw probability must
  not become gate confidence until its meaning and calibration are verified
  (for Jev, in Milestone 12). Both are logged, as separate fields.

### 6.6 Listing intelligence (Level 2)

- FR-LST-1: Raw seller input is preserved unchanged. Normalized facts,
  inferred decisions, risk signals and human outcomes are stored separately.
- FR-LST-2: Deterministic normalization covers casing, whitespace, known brand
  aliases, units, storage terms and numeric formats. Uncertain model names are
  never rewritten automatically.
- FR-LST-3: Exact attributes are extracted deterministically where possible.
- FR-LST-4: Automated outcomes are limited to `auto_accept`,
  `accept_with_warning` and `manual_review`. Final rejection requires human
  review.
- FR-LST-5: Risk signals (prohibited item, counterfeit indicator, misleading
  claim, suspicious review, policy mismatch, adult content, unsafe product,
  unsupported medical claim) are triage signals, not legal conclusions.

### 6.7 Annotation tool requirements

The annotation tool is implemented in Milestone 9 (search annotation) and in
the listing milestones (listing annotation). These are requirements only.
Nothing is implemented in Milestone 0. The tool records human judgments and
never generates labels.

**Search annotation (Milestone 9).** For each golden query the tool must show:

- the query;
- the parsed constraints;
- optional provider decisions, when a provider is enabled;
- the candidate products;
- each product's attributes;
- each candidate's retrieval sources (lexical, dense or both, with ranks).

It must save structured labels, one per query-candidate pair: `relevant`,
`borderline` or `not relevant`, plus the reviewer identity and a timestamp.
It must also record gold constraints (section 9.2), annotated by a human, for
each query. Labels start unset. The tool never pre-fills a label.

**Listing annotation (deferred to the listing milestones).** Requirements to
be satisfied when those milestones are planned. The tool must show:

- the raw listing and the normalized listing;
- extracted attributes;
- risk signals;
- confidence values;
- the proposed review route.

It must save a structured human outcome. The outcome vocabulary is decided in
the listing milestones. No listing annotation is built before then.

## 7. Hard versus soft constraints

| Kind | Examples | Treatment |
|---|---|---|
| Potential hard constraint | Explicit category, brand, model, RAM, storage, storage type, min/max price | May become a SQL/structured filter when explicitly stated and parsed deterministically with high reliability. A model-derived decision may do so only after the Milestone 12 validation below |
| Soft signal / semantic intent | gaming, coding, student, lightweight, premium, comfort, travel, office use, `sasta` | Influences retrieval and ranking only. Never a hard filter unless reliable structured data supports it and a measured experiment justifies it |
| Unknown / absent | Any value not present in the query | Remains `null`. Never inferred into a filter |

Rules:

1. **Deterministic extraction.** An explicit constraint parsed by the
   deterministic parser with high reliability may become a hard filter.
2. **Model-derived decisions.** A model-derived decision may become a hard
   filter only after a confidence threshold has been justified from
   human-reviewed data in Milestone 12. Before that validation, model-derived
   decisions may influence ranking or retrieval routing but not hard
   filtering. After validation, low-confidence decisions still never become
   hard filters, and medium-confidence decisions avoid risky hard filters and
   use broader retrieval.
3. Contradictory constraints (for example `min_price > max_price`) are
   detected and reported. They are never silently resolved by guessing.
4. A zero-result outcome caused by filtering is measured and reported in
   evaluation (zero-result rate).

### 7.1 Hinglish support

The initial **minimum** supported phrase set is exactly the one listed in the
master plan: `ke andar`, `ke under`, `wala`, `ka`, `sasta`, `coding ke liye`,
`gaming ke liye`.

The phrases are handled through an extensible, versioned phrase dictionary,
not through ad-hoc regular expressions. Each entry maps a normalized phrase to
a small typed role. **Provisional:** the roles below are proposed
interpretations, not verified behavior. They are confirmed with tests in
Milestone 6 and corrected there if the tests show otherwise.

| Phrase | Intended role |
|---|---|
| `ke andar`, `ke under` | Price upper-bound marker, meaning the same as `under`. Only produces `max_price` when attached to an explicit price value (for example `50k ke andar`) |
| `wala`, `ka` | Connective/possessive tokens. No constraint on their own |
| `sasta` | Soft "low price" preference. Never produces a numeric `max_price`, because no value was stated |
| `coding ke liye`, `gaming ke liye` | Semantic-intent marker (`coding`, `gaming`). Never a hard filter |

Extending the dictionary means adding entries with tests. Milestone 0 does not
add a large phrase list, and it does not implement the dictionary.

## 8. Confidence gating and human review

- Every model-driven decision carries a raw provider probability (provider
  output) and a gate confidence (the value the policy consumes), stored and
  logged separately (FR-DEC-5). A raw probability is not gate confidence until
  its meaning and calibration are verified. For Jev that is Milestone 12.
- Every model-driven decision passes through the gate before it can affect
  retrieval or listing outcomes.
- Gate bands:
  - **High:** the decision may be accepted. It may become a filter only where
    the rules in section 7 allow, which for model-derived decisions means only
    after Milestone 12 has justified a threshold from human-reviewed data.
    Before that, it may influence ranking or retrieval routing only.
  - **Medium:** the decision may inform ranking or strategy, but must not
    create risky hard filters. Retrieval is broadened.
  - **Low:** fall back to the deterministic result, or request clarification.
- Initial thresholds are experimental and are not specified in this document.
  Final thresholds are selected from human-reviewed data, calibration,
  coverage, abstention and mistake cost.
- Confidence is not correctness. Gate quality is measured, not assumed.
- Listings: low-risk, high-confidence cases may be auto-accepted. Uncertain
  cases are warned or routed to manual review. No automatic rejection and no
  automatic seller banning.
- Human review is required for Golden Dataset labels, the 30-product catalog
  review sample, decision-evaluation labels and final listing rejections.
  Review completion is never recorded automatically.

## 9. Evaluation

- Search versions:

  ```text
  V0 = lexical
  V1 = lexical + dense + RRF
  V2 = V1 + structured filtering
  V3 = V2 + reranking
  V4 = V3 + confidence-gated Jev decisions
  ```

- Retrieval metrics: Precision@K, Recall@K, NDCG@K, MRR, Constraint
  Satisfaction Rate, irrelevant-result rate and zero-result rate. Definitions
  are in section 9.2.
- Golden Dataset: 20 human-reviewed search queries covering unconstrained,
  brand/category, specification, price, semantic-intent and Hinglish queries.
  Relevance labels are `relevant`, `borderline` and `not relevant`. Generated
  labels are never authoritative.
- Decision evaluation (Level 2): compare deterministic, optional general LLM,
  Jev, and deterministic plus confidence-gated Jev using accuracy, precision,
  recall, F1, confusion matrix, schema validity, abstention, coverage,
  calibration, Brier score where appropriate, latency and cost per 1,000
  decisions.
- Review routing (Level 2): automation rate, review rate, unsafe auto-accept
  rate and unnecessary review rate.
- `evals/` stays separate from application code.
- Every experiment records ID, timestamp, Git commit, catalog/dataset/golden
  versions, provider and model, policy/prompt version, embedding model,
  reranker, candidate counts, thresholds, quality, latency, cost, notes and
  environment. Reports are generated from actual outputs only.

### 9.1 Evaluation order before the Golden Dataset exists

Human-reviewed labels do not exist until Milestone 9, so early milestones
cannot produce authoritative quality numbers.

- Milestones 3 to 8 may use provisional, non-authoritative smoke queries to
  check that retrieval behaves sensibly and to choose provisional settings.
  Results from them are not reported as quality metrics.
- Milestone 5 selects a provisional `rrf_k`, records it in configuration and
  marks it provisional. No `rrf_k` value is chosen in Milestone 0.
- Human-reviewed, authoritative labels arrive in Milestone 9.
- Milestone 10 re-runs V0 through V3 on the human-reviewed Golden Dataset.
- Final reporting must distinguish provisional tuning from authoritative
  evaluation, and label every number as one or the other.

### 9.2 Metric definitions

- **Constraint Satisfaction Rate (CSR).** The share of returned results whose
  structured attributes satisfy the query's gold constraints. Gold constraints
  are annotated by a human for each golden query (section 6.7). CSR is
  computed against these human-annotated gold constraints, never against the
  parser's own output, because scoring the parser against itself would
  measure nothing. Parser extraction accuracy is a separate metric, also
  measured against the gold constraints. CSR is reported both **per result at
  K** (the fraction of the top K results satisfying all gold constraints,
  averaged over queries) and **per query at K** (the fraction of queries whose
  top K results all satisfy every gold constraint). A result whose attribute
  value is missing is reported as `unknown`, not silently counted as satisfied
  or violated. The report states which way it was counted.
- **Recall@K.** Human relevance judgments come from a labelled candidate pool,
  not from exhaustive judgment of the full catalog. Recall@K is therefore
  reported as **pooled recall**: relevant items found in the top K divided by
  relevant items in the pool. The report discloses candidate-pool bias: items
  never pooled are unjudged, so a system that surfaces items outside the pool
  is under-credited and pooled recall overstates true recall.
- **Borderline mapping.** How `borderline` maps to graded or binary
  relevance for Precision, Recall, NDCG and MRR is **provisional and open**
  until the annotation policy is approved (section 14, item 10). No mapping is
  assumed here.
- **Small-sample reporting.** The initial Golden Dataset has only 20 queries.
  Reports must include per-query results alongside aggregates, and an
  explicit uncertainty warning that aggregate differences between versions on
  20 queries may not be meaningful. No significance is claimed without an
  appropriate method.

### 9.3 Tuning and reporting leakage

Any configuration choice (for example `rrf_k`, candidate sizes, rerank depth,
gate thresholds) tuned on the same queries used for final reporting yields
optimistic metrics. Every authoritative report must do one of the following,
and state which:

1. use a recorded tuning/reporting split of the queries;
2. use cross-validation, with the procedure recorded;
3. disclose explicitly that the configuration was tuned on the reporting set
   and that its metrics are optimistic.

Preferred: freeze the primary configuration before final authoritative
reporting. Split sizes and results are not decided in Milestone 0.

## 10. Latency

- No universal latency target is promised in this specification.
- Measured stages: `query_understanding_ms`, `jev_decision_ms`, `lexical_ms`,
  `vector_ms`, `filtering_ms`, `rrf_ms`, `reranking_ms`, `serialization_ms`,
  `total_ms`.
- Report P50, P95 and P99 together with the hardware, catalog size and
  configuration used. Any latency budget is set later from those
  measurements.

## 11. Security and safety

- Seller content and queries are untrusted data. They cannot override system
  instructions or change provider prompts, policies or thresholds.
- Credentials live only in environment variables. `.env` files are never
  committed. Secrets are never logged.
- Remote providers use configurable timeouts, limited retries, typed
  validation and safe fallback.
- Provider-specific code is isolated behind typed interfaces.
- The system produces no automatic rejection, no seller banning and no final
  legal conclusions.
- Required negative tests include empty input, nonsense input, contradictory
  constraints, unsupported categories, provider failure, timeouts, malformed
  provider responses, low-confidence decisions and prompt-like malicious
  seller text.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Dense retrieval returns similar products that violate explicit constraints | Deterministic parsing and safe structured filtering (V2). Constraint Satisfaction Rate is measured |
| Over-filtering causes zero results | Filter only explicit, reliably parsed constraints. Measure zero-result rate. Medium confidence broadens retrieval |
| Synthetic data inflates metrics | Label synthetic records. Record provenance. Keep evaluation labels out of embedding text. Use human-reviewed labels only |
| Fragile regular-expression parser | Small typed parser, a phrase dictionary and unit tests |
| Model confidence mistaken for correctness | Calibration, human-reviewed decision data, thresholds justified by measurement |
| Remote provider outage, latency or cost | Optional provider, timeout, limited retry, deterministic fallback, cost logging |
| Prompt injection via seller text | Treat seller text as data. Bounded typed outputs. Negative tests |
| Architecture creep (extra search engines, queues, orchestration) | Locked stack. New infrastructure only on a measured requirement |
| Fabricated or unverified claims in documentation or reports | Reports generated from actual runs. Unknowns recorded as open questions |

## 13. Acceptance criteria

### 13.1 Milestone 0 (this specification)

- [ ] `docs/spec.md`, `docs/architecture.md` and `docs/data-quality.md` exist.
- [ ] ADR-001 to ADR-005 exist under `docs/decisions/`.
- [ ] The documents cover goals, non-goals, requirements, architecture, data
  model, hard and soft constraints, evaluation, quality, Jev boundaries,
  confidence gating, human review, latency, security, risks and acceptance
  criteria.
- [ ] No application code, datasets, labels, metrics or benchmark results are
  created.
- [ ] No existing tracked file is modified.

### 13.2 Core (Level 1) definition of done

Core is done only when all of these exist: PostgreSQL/pgvector, catalog
quality, lexical/dense/hybrid retrieval, RRF, deterministic understanding,
structured filtering, reranking, a human-reviewed Golden Dataset, evaluation,
FastAPI, tests, measured latency and quality, an experiment report, and an
explanatory README.

### 13.3 Jev definition of done

Jev is done only when all of these exist: official integration, typed
validation, fallback, model/version logging, human evaluation, latency and
cost measurement, calibration, justified thresholds, and a
deterministic-versus-Jev comparison.

### 13.4 Listing intelligence definition of done

Listing intelligence is done only when raw data is preserved, normalization
and contradiction detection work, decisions are bounded, uncertainty reaches
human review, and evaluation is human-reviewed.

## 14. Open questions

1. **Initial dataset:** decided in Milestone 2: a project-authored deterministic synthetic
   seed catalog. See `docs/data-quality.md` §2.5.
2. **`rrf_k`:** configurable. The initial value is unspecified. Milestone 5
   selects a provisional value, marked provisional. Milestone 10 re-evaluates
   on the human-reviewed Golden Dataset (ADR-002, section 9.1).
3. **Embedding and reranker models:** selected in Milestones 4 and 8 from
   measurement (ADR-004).
4. **Confidence thresholds:** experimental. Selected from human-reviewed data
   in Milestone 12. Until then model-derived decisions cannot be hard
   filters (section 7).
5. **Price-bound inclusivity:** whether `under 50k` means `<= 50000` or
   `< 50000`. The proposed default is inclusive (`<=`). Confirm in
   Milestone 6.
6. **Implicit storage units:** in `laptop 8gb 256 ssd`, a bare `256` next to a
   storage term is proposed to mean 256 GB. Confirm in Milestone 6.
7. **Currency:** examples imply INR (`₹`, `rs`, `k`). Queries are assumed to be
   in the catalog currency. Multi-currency support is out of scope.
8. **Jev API behavior, model versions and output semantics:** unverified.
   API verification is deferred to Milestone 11. Probability meaning and
   calibration are evaluated in Milestone 12 (ADR-005).
9. **Storage taxonomy:** NVMe is an interface/protocol, and NVMe products are
   normally SSDs. It is undecided whether (a) `storage_type` is SSD/HDD with
   the interface (for example NVMe, SATA) as a separate field, or (b) NVMe is
   represented as an SSD subtype. Milestone 6 or 7 must decide. Either way, an
   explicit `ssd` query must not accidentally exclude NVMe SSDs, and a test
   must cover this.
   **Milestone 2 schema decision (implemented):** `storage_type` (medium: `SSD`, `HDD`) and
   `storage_interface` (`NVME`, `SATA`) are separate columns, and `NVME` requires
   `storage_type = SSD` (a database CHECK). This settles the stored representation only. The
   query-matching semantics (for example that an `ssd` query must include NVMe SSDs) remain
   deferred to Milestone 6/7.
10. **Borderline relevance mapping:** provisional and open until the
    annotation policy is approved (section 9.2).
11. **Provisional, dataset-dependent values:** any numeric range, tolerance or
    duplicate threshold that depends on the chosen dataset is provisional
    until Milestone 2 (`docs/data-quality.md` §3). **Milestone 2:** the values are chosen and
    documented as provisional in `docs/data-quality.md` §3.1.
