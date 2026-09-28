# Master Engineering Plan

## Project: Intelligent E-commerce Search & Decision Engine

You are the lead Search Engineer, Backend Engineer, and Applied ML Engineer responsible for building this project.

Execution philosophy:

```text
Build small -> test -> measure -> evaluate -> document -> stop
```

This is not a generic RAG chatbot, an unrestricted LLM wrapper, or a collection of disconnected demos. It is a production-oriented e-commerce search and bounded-decision system built around measurable retrieval quality, explicit constraints, confidence gating, safe fallback, and human review.

The project has two related workflows:

1. Buyer-side product search — the core portfolio project.
2. Seller-side listing intelligence — implemented only after the core is stable.

## 1. Business problem

Marketplace listings are inconsistent across sellers. They may contain misspellings, missing attributes, conflicting units, misleading claims, policy risks, and duplicate products. Buyers submit short search queries that mix category, brand, exact specifications, price, and semantic intent.

Examples:

```text
laptop 8gb 256 ssd
hp laptop under 50k
coding ke liye laptop
nike shoes under 5k
anc headphones
iphone 15
```

A pure semantic retriever may return semantically similar products that violate explicit constraints. For `laptop 8gb 256 ssd`, RAM, storage, and storage type must be treated as explicit constraints. For `coding laptop`, coding is primarily semantic intent.

The system combines:

```text
Catalog ingestion
+ listing normalization
+ query understanding
+ lexical retrieval
+ dense retrieval
+ structured filtering
+ Reciprocal Rank Fusion
+ reranking
+ bounded model decisions
+ confidence gating
+ human review
+ evaluation
```

## 2. Absolute execution rule

Implement exactly one milestone at a time. For every milestone:

1. Inspect the repository.
2. Understand the existing implementation.
3. Present a milestone-scoped plan before editing.
4. Implement only the current milestone.
5. Write or update tests.
6. Run relevant tests, lint, and validation.
7. Run the relevant benchmark or evaluation when it exists.
8. Fix milestone-related failures.
9. Update relevant documentation.
10. Report exact files changed and commands executed.
11. Report actual results and limitations.
12. Stop.

Do not automatically continue. Do not generate future milestone code. Create only the smallest interface required by the current milestone.

## 3. Scope

### Level 1 — Core search engine

- Product catalog
- Catalog quality validation
- PostgreSQL Full Text Search
- Local embeddings and pgvector
- Hybrid retrieval with Reciprocal Rank Fusion
- Deterministic query understanding
- Structured filtering
- Cross-encoder reranking
- Human-reviewed Golden Dataset
- Search evaluation
- FastAPI
- Tests and measured latency

The project is portfolio-complete when this level is complete.

### Level 2 — Marketplace decision layer

Only after the core is stable:

- Jev Decision Provider
- Query category and semantic-intent classification
- Retrieval-strategy routing
- Ambiguity and out-of-catalog detection
- Listing-category classification
- Attribute-candidate validation
- Policy-risk triage
- Confidence-based human-review routing

### Level 3 — Production extensions

Only after evaluation is stable:

- Regression CI
- Observability
- Frontend
- Production Docker hardening
- AWS deployment
- Terraform

## 4. Locked technology stack

Use Python 3.12, uv, FastAPI, PostgreSQL, pgvector, PostgreSQL Full Text Search, SQLAlchemy, Alembic, Pydantic, pytest, and Ruff.

Use configurable local sentence-transformers-compatible models for embeddings and cross-encoder reranking. Do not make a paid embedding API a core dependency.

Do not introduce Elasticsearch, OpenSearch, Qdrant, Pinecone, Kafka, Airflow, Kubernetes, or multiple databases for the same search workload unless measurements prove PostgreSQL insufficient.

## 5. Architecture

Buyer search:

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

Seller listing:

```text
Raw listing
 -> deterministic normalization
 -> attribute candidates
 -> optional Jev bounded decisions
 -> confidence/risk gate
 -> auto-accept | warning | manual review
 -> normalized catalog record
```

Shared components include taxonomy, attribute schema, brand dictionary, normalization rules, provider interfaces, confidence policies, and evaluation infrastructure.

## 6. Product catalog

Core fields:

```text
product_id, seller_id, title, description, category, subcategory,
brand, price, currency, rating, review_count, availability,
source_type, created_at, updated_at
```

Laptop fields: `ram_gb`, `storage_gb`, `storage_type`, `processor`, `gpu`, `screen_size_inches`, `operating_system`, `weight_kg`.

Phone fields: `ram_gb`, `storage_gb`, `camera`, `battery_mah`, `screen_size_inches`, `operating_system`.

Shoes fields: `size`, `color`, `material`, `gender`.

Headphone fields: `wireless`, `anc`, `battery_life_hours`, `connectivity`.

Keep the schema extensible. Do not populate irrelevant category attributes. JSONB may hold validated long-tail attributes, but frequently filtered fields should have proper columns.

## 7. Dataset and quality strategy

Use a reproducible public dataset, a clearly labeled synthetic/seed catalog, or an allowed combination. Never describe synthetic data as real marketplace data. Record provenance.

Before evaluation, validate:

1. Product IDs are unique.
2. Required fields exist.
3. Prices and numeric attributes are plausible.
4. Category and attribute combinations are coherent.
5. Titles do not obviously contradict structured attributes.
6. Units and storage types are recognized.
7. Brand/category combinations are plausible.
8. Duplicates do not dominate.
9. Synthetic records are identified.
10. Embedding text contains no evaluation labels.

Prepare at least 30 representative products for real human review. Do not mark the review complete automatically.

## 8. Query understanding

Create a typed `QueryUnderstanding` model with optional fields:

```json
{
  "raw_query": "hp laptop 8gb 256 ssd under 40k",
  "category": "laptop",
  "brand": "HP",
  "ram_gb": 8,
  "storage_gb": 256,
  "storage_type": "SSD",
  "min_price": null,
  "max_price": 40000,
  "semantic_intent": null,
  "retrieval_strategy": null,
  "attributes": {}
}
```

Do not invent missing constraints. Treat explicit category, brand, model, RAM, storage, storage type, and price as potential hard constraints. Treat gaming, coding, student, lightweight, premium, comfort, travel, and office use as semantic intent unless reliable structured data supports filtering.

## 9. Deterministic parser

Implement deterministic parsing first. Support values such as `8gb`, `8 gb`, `16gb`, `256gb`, `512gb`, `1tb`, `40k`, `₹40000`, and `rs 40000`.

Normalize `40k` to `40000` and `1tb` to `1024gb`. Support RAM, memory, storage, SSD, NVMe, HDD, and common Hinglish phrases including `ke andar`, `ke under`, `wala`, `ka`, `sasta`, `coding ke liye`, and `gaming ke liye`.

Keep the parser small, typed, and testable. Do not create hundreds of fragile regular expressions.

## 10. Decision providers and Jev

Create a small provider interface conceptually shaped as:

```text
understand(query, deterministic_result) -> DecisionResult
```

Implementations may include:

- `DeterministicDecisionProvider`
- `JevDecisionProvider`
- Optional `LLMDecisionProvider`

The core must work without any remote provider. Isolate provider-specific API code and validate outputs through Pydantic.

Jev access is available, but integration occurs only after core evaluation is stable. Use Jev as a probability-backed bounded-decision layer for category, semantic intent, retrieval strategy, ambiguity, out-of-catalog detection, listing-category validation, attribute-candidate validation, policy-risk triage, and human-review routing.

Do not use Jev to invent attributes, replace reliable numeric parsing, make final legal conclusions, automatically ban sellers, create labels, or replace the reranker without a controlled experiment.

Keep endpoint, model, timeout, retry, enablement, and thresholds configurable. Store credentials only in environment variables. Log provider, pinned model version, latency, decision probabilities, fallback reason, and returned cost where available. Never log secrets.

## 11. Confidence gating

Initial thresholds are experimental. Final thresholds must be selected from human-reviewed data, calibration, coverage, abstention, and mistake cost.

High-confidence query decisions may be accepted. Medium-confidence decisions should avoid risky hard filters and use broader retrieval. Low-confidence decisions should fall back safely or request clarification.

For listings, low-risk high-confidence cases may be auto-accepted; uncertain cases must be warned or routed to human review. Confidence is not correctness.

## 12. Listing intelligence

Preserve raw seller input. Store normalized facts, inferred decisions, risk signals, and human outcomes separately.

Use deterministic normalization for casing, whitespace, known brand aliases, units, storage terms, and numeric formats. Do not automatically rewrite uncertain model names.

Extract exact attributes deterministically where possible. Use Jev only for bounded validation questions such as whether an extracted value refers to RAM or storage, whether an attribute is category-compatible, whether the listing is ambiguous, and whether review is required.

Optional risk signals include prohibited item, counterfeit indicator, misleading claim, suspicious review, policy mismatch, adult content, unsafe product, and unsupported medical claim. These are triage signals, not final legal conclusions.

Automated review outcomes may be `auto_accept`, `accept_with_warning`, or `manual_review`. Final rejection requires human review.

## 13. Retrieval and ranking

### Lexical

Use PostgreSQL Full Text Search for brands, model names, exact terms, and specifications. Create appropriate indexes and measure latency. This is V0.

### Dense

Use local embeddings and pgvector. Make the model configurable and document its name, revision, dimensions, normalization, catalog version, and generation timestamp. Regenerate embeddings and rerun evaluation after configuration changes.

### Hybrid

Combine lexical and dense retrieval using Reciprocal Rank Fusion. Initial configurable values: `lexical_k=50`, `dense_k=50`, `candidate_k=50`. Preserve source ranks and fused score. This contributes to V1.

### Filtering

Safely apply explicit category, brand, RAM, storage, storage type, and price constraints. Push filtering into SQL when safe and beneficial. Never use low-confidence model decisions as hard filters. This is V2.

### Reranking

Use a configurable lightweight cross-encoder. Support reranker on and off. Start with 50 fused candidates, rerank 20, and return 10. Benchmark no reranker, rerank 10, and rerank 20. This is V3.

## 14. Latency and evaluation

Do not promise an arbitrary universal latency. Measure `query_understanding_ms`, `jev_decision_ms`, `lexical_ms`, `vector_ms`, `filtering_ms`, `rrf_ms`, `reranking_ms`, `serialization_ms`, and `total_ms`. Report P50, P95, and P99.

Create `evals/` separate from application code. Implement Precision@K, Recall@K, NDCG@K, MRR, Constraint Satisfaction Rate, irrelevant-result rate, and zero-result rate.

Search versions:

```text
V0 = lexical
V1 = lexical + dense + RRF
V2 = V1 + structured filtering
V3 = V2 + reranking
V4 = V3 + confidence-gated Jev decisions
```

Create 20 human-reviewed search queries covering unconstrained, brand/category, specification, price, semantic-intent, and Hinglish queries. Generated labels are never authoritative.

For decisions, compare deterministic, optional general LLM, Jev, and deterministic plus confidence-gated Jev. Measure accuracy, precision, recall, F1, confusion matrix, schema validity, abstention, coverage, calibration, Brier score where appropriate, latency, and cost per 1,000 decisions.

For review routing, measure automation rate, review rate, unsafe auto-accept rate, and unnecessary review rate.

## 15. Annotation and experiment tracking

Build a small CLI or local interface showing query, parsed constraints, optional Jev decisions, candidate products, attributes, and retrieval sources. Allow labels `relevant`, `borderline`, and `not relevant`.

For listing review, show raw and normalized listing data, extracted attributes, risk signals, confidence, and proposed route. Save structured human labels.

Every experiment records ID, timestamp, Git commit, catalog/dataset/golden versions, provider and model, policy/prompt version, embedding model, reranker, candidate counts, thresholds, quality, latency, cost, notes, and environment. Generate reports from actual outputs.

## 16. FastAPI

Core endpoints:

```text
GET /health
GET /search?q=...
POST /search
```

Use typed requests/responses and OpenAPI. Search metadata should include search version, provider, reranker state, latency, result count, and safe applied filters.

Listing endpoints are deferred until their milestone:

```text
POST /listings/analyze
POST /listings/normalize
GET /reviews/pending
POST /reviews/{review_id}/decision
```

## 17. Intended repository structure

Create directories only when their milestone begins:

```text
ecommerce-search/
  docs/
    spec.md
    architecture.md
    data-quality.md
    evaluation.md
    decisions/
  src/
    api/
    config/
    models/
    ingestion/
    catalog/
    normalization/
    query_understanding/
    decision/
    search/
    review/
    observability/
  tests/
    unit/
    integration/
    contract/
  evals/
    goldens/
    metrics/
    runners/
    annotation/
    decisions/
    reports/
  data/
    raw/
    processed/
    labels/
  scripts/
  pyproject.toml
  uv.lock
  Dockerfile
  docker-compose.yml
  .env.example
  README.md
```

## 18. Testing, security, and observability

Test parsing, catalog validation, retrieval, RRF, filtering, reranking, APIs, provider failure, timeouts, malformed responses, low confidence, fallback, empty input, nonsense, contradictory constraints, unsupported categories, and prompt-like malicious seller text.

Treat seller content as untrusted data. It cannot override system instructions. Use environment variables, timeouts, limited retries, safe fallbacks, typed validation, and no secret logging.

Later observability should record request ID, search version, provider/model, stage timing, result count, filters, fallback reason, review routing, errors, and aggregate latency/error/fallback/confidence metrics.

## 19. Milestones

### Milestone 0 — Specification

Create only:

- `docs/spec.md`
- `docs/architecture.md`
- `docs/data-quality.md`
- ADR-001 PostgreSQL + pgvector
- ADR-002 Hybrid Retrieval with RRF
- ADR-003 DecisionProvider abstraction
- ADR-004 Local embedding strategy
- ADR-005 Jev as optional bounded-decision provider

Include goals, non-goals, requirements, architecture, data model, hard vs soft constraints, evaluation, quality, Jev boundaries, confidence gating, human review, latency, security, risks, and acceptance criteria. Do not write application code. Stop.

### Milestone 1 — Foundation

Implement uv, Python 3.12, FastAPI, PostgreSQL, pgvector, SQLAlchemy, Alembic, Docker Compose, configuration, and `GET /health`. Run tests, lint, and connectivity validation. Stop.

### Milestone 2 — Product catalog

Implement schema, migrations, ingestion, approved data, indexes, and automated quality checks. Prepare a 30-product human-review sample. Do not claim review completion until a human reviews it. Stop.

### Milestone 3 — Lexical baseline

Implement PostgreSQL FTS, search-vector generation, indexes, retrieval, and API integration. Measure correctness and P50/P95/P99. This is V0. Stop.

### Milestone 4 — Dense retrieval

Implement configurable local embeddings, generation, pgvector storage/indexes, and vector retrieval. Measure generation and query latency. Document the model decision. Stop.

### Milestone 5 — Hybrid search

Implement lexical plus dense retrieval with RRF. Compare lexical, dense, and hybrid. Preserve ranks. This is V1. Stop.

### Milestone 6 — Query understanding

Implement typed query understanding, deterministic parser, normalization, provider interface, and deterministic provider. Test English, short-form, numeric, and Hinglish queries. Do not implement Jev yet. Stop.

### Milestone 7 — Structured filtering

Implement safe filters for category, brand, RAM, storage, storage type, min price, and max price. Evaluate extraction, satisfaction, zero results, and contradictions. This is V2. Stop.

### Milestone 8 — Reranking

Implement configurable cross-encoder reranking with on/off modes. Benchmark no reranker, rerank 10, and rerank 20 using quality and latency. This is V3. Stop.

### Milestone 9 — Golden Dataset

Implement schema, annotation tool, evaluator, and 20 query candidates. Implement retrieval metrics. Pause for real human labeling. Stop.

### Milestone 10 — Core experiment report

Compare V0–V3 from actual runs. Include configuration, quality, latency, failure cases, limitations, and measured recommendation. Stop.

### Milestone 11 — Jev query provider

Prerequisites: stable V0–V3, human-reviewed Golden Dataset, credentials, and verified official API behavior. Implement typed client/provider, timeout, limited retry, safe fallback, configuration, and telemetry. Use bounded decisions only. Core must still work without Jev. Stop.

### Milestone 12 — Jev query experiment

Create human-reviewed decision data. Compare deterministic, optional LLM, Jev, and confidence-gated Jev. Measure classification, calibration, coverage, latency, cost, and end-to-end retrieval metrics. This is V4. Stop.

### Milestone 13 — Listing normalization

Implement raw and normalized models, deterministic unit and brand normalization, attribute candidates, compatibility checks, and contradiction detection. Preserve original values. Stop.

### Milestone 14 — Jev listing decisions

Implement bounded category, subcategory, attribute-validation, ambiguity, and review-routing decisions. Do not allow invented attributes. Evaluate on human-reviewed data. Stop.

### Milestone 15 — Policy-risk triage

Use explicit versioned criteria, human-reviewed examples, audit fields, and manual review. Do not implement automatic seller banning. Measure false positives and false negatives separately. Stop.

### Milestone 16 — Regression CI

Implement unit/integration CI, evaluation smoke tests, provider contract tests, and configurable thresholds. Routine CI must not require live Jev. Stop.

### Milestone 17 — Observability

Implement structured logging, request IDs, provider telemetry, stage timings, fallback/confidence/review metrics, and P50/P95/P99 reporting. Stop.

### Milestone 18 — Frontend

Build a simple search UI with result cards and optional debug metadata. Add reviewer UI only if listing workflow exists. Never expose hidden reasoning. Stop.

### Milestone 19 — AWS

Only after local stability, implement production image, ECR, ECS/Fargate, PostgreSQL with pgvector, Secrets Manager, CloudWatch, and Terraform. Do not introduce EKS. Require approval before chargeable actions. Stop.

## 20. Definition of done

Core is done only when PostgreSQL/pgvector, catalog quality, lexical/dense/hybrid retrieval, RRF, deterministic understanding, structured filtering, reranking, human-reviewed Golden Dataset, evaluation, FastAPI, tests, measured latency/quality, experiment report, and explanatory README all exist.

Jev is done only when official integration, typed validation, fallback, model/version logging, human evaluation, latency/cost measurement, calibration, justified thresholds, and deterministic-vs-Jev comparison exist.

Listing intelligence is done only when raw data is preserved, normalization and contradictions work, decisions are bounded, uncertainty reaches human review, and evaluation is human-reviewed.

## 21. Final principle

Prefer deterministic extraction over unnecessary inference, bounded decisions over unrestricted generation, confidence gating over blind automation, human-reviewed labels over generated ground truth, measured improvement over assumed performance, and simple architecture over unnecessary infrastructure.

## Start now

Inspect the repository and implement only Milestone 0. Create the requested specification, architecture, data-quality, and ADR documents. Do not write application code, call Jev, create frontend/AWS code, or generate fake results. Validate the documents, list files and commands, report unresolved questions, and stop.
