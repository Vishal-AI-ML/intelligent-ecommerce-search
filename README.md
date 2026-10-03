# Intelligent E-commerce Search & Decision Engine

A production-oriented e-commerce search and bounded-decision system.

Development is organized milestone by milestone. See
`docs/MASTER_PLAN.md` for the complete engineering plan.

## Current status

Milestone 1 (Foundation): typed settings, FastAPI app with `GET /health`,
PostgreSQL 16 + pgvector via Docker Compose, SQLAlchemy 2 (sync) with psycopg 3,
and Alembic.

Milestone 2 (Product catalog): **complete.** A product schema (migration `0002`), a
deterministic **synthetic** seed catalog of 240 products (80 laptops, 60 phones, 50 shoes,
50 headphones; not real marketplace data), atomic idempotent ingestion, automated quality
checks and reports, and a recorded human review of a 36-product sample (see "Product catalog"
below).

Milestone 3 (PostgreSQL lexical baseline, **V0**): **implemented.** A derived
`product_search_documents` table (migration `0003`, weighted `tsvector` + GIN index), document
building inside ingestion, an explicit `search reindex` backfill command, a lexical retrieval
service and `GET`/`POST /search` (see "Lexical search (Milestone 3)" below and
`docs/search-v0-baseline.md`).

Milestone 4 (dense semantic retrieval): **implemented.** Local embeddings with the pinned
`sentence-transformers/all-MiniLM-L6-v2` model (ADR-006), a `product_embeddings` pgvector table
(migration `0004`, `vector(384)`), explicit `embed` / `embed-status` commands and
`GET`/`POST /search/dense` (see "Dense search (Milestone 4)" below and
`docs/search-dense-m4.md`). A measured benchmark kept the exact scan: no persistent ANN index at
240 products. Hybrid retrieval (RRF) was added in Milestone 5.

Milestone 5 (hybrid retrieval, **V1**): **implemented.** `GET`/`POST /search/hybrid` fuses the
lexical and dense candidate lists with Reciprocal Rank Fusion and keeps both source ranks; no
schema change. `rrf_k = 100` is **provisional** (see "Hybrid search (Milestone 5)" below,
`docs/search-hybrid-m5.md` and ADR-002). `/search` remains V0 and `/search/dense` remains
available.

Milestone 6 (deterministic query understanding): **implemented.** A pure, typed, local parser
and the `DecisionProvider` interface with its deterministic provider (ADR-003). `/search/hybrid`
reports the parse in an additive, **informational** `query_understanding` block; it does not
change retrieval, ranking or filtering (`applied_filters` stays empty). See "Query understanding
(Milestone 6)" below and `docs/query-understanding-m6.md`.

## Prerequisites

- Python 3.12 (managed by `uv` via `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with Compose v2

## Setup

```bash
uv sync
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

**Before running Docker Compose or the app, edit `.env` and set `POSTGRES_PASSWORD` to a
non-empty, local-only password of your choosing.** `.env.example` deliberately leaves it
empty; Compose and the application both refuse to start with a missing or empty password.
Do not reuse a real password. `.env` is git-ignored and must never be committed.

Lexical-search limits are typed settings too: `SEARCH_LEXICAL_K=50` (maximum `top_k`),
`SEARCH_DEFAULT_TOP_K=10` and `SEARCH_MAX_QUERY_LENGTH=200`. They are forwarded by Compose to the
`api` service only (not to `db` or `migrate`), with these same defaults.

Database waits are bounded by settings in `.env.example` (`DB_CONNECT_TIMEOUT_SECONDS`,
`DB_POOL_TIMEOUT_SECONDS`, `DB_STATEMENT_TIMEOUT_MS`). The defaults are conservative
local-development values, not benchmarked figures or production SLOs. Compose forwards
them from `.env` (falling back to the same defaults) only to the containers whose code
uses them: `api` receives `DB_CONNECT_TIMEOUT_SECONDS`, `DB_POOL_TIMEOUT_SECONDS` and
`DB_STATEMENT_TIMEOUT_MS`; `migrate` receives only `DB_CONNECT_TIMEOUT_SECONDS`, because
Alembic uses no pool timeout or statement timeout. `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
are not forwarded by Compose; the containers use the built-in defaults for them.

## Run

Everything in containers (database, then migration, then API):

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

Startup order: `db` becomes healthy, the one-shot `migrate` service runs
`alembic upgrade head` and exits 0, then `api` starts. The API never runs
migrations itself.

**Compose, migration and a running API do not by themselves give you a populated search
index.** `migrate` creates the empty `product_search_documents` table (revision `0003`) and
nothing else. `GET /search` then returns no results until the catalog is ingested and the
documents exist: run `catalog ingest --database <db>` (which builds the documents) and/or
`search reindex --database <db>`, then check `search status --database <db>` (see "Lexical search
(Milestone 3)" below). The seed file is not part of the container image, so run these from the
host against the published database port.

Database only, with the API/tests on the host:

```bash
docker compose up -d db
uv run alembic upgrade head
uv run uvicorn ecommerce_search.api.app:create_app --factory --port 8000
```

### Warning: downgrading the initial migration is destructive

`alembic downgrade base` runs `DROP EXTENSION IF EXISTS vector`. This may require
extension ownership or superuser privileges, fails if any object depends on the extension
(for example vector columns or indexes in later milestones), and will also drop a `vector`
extension that already existed before this project created it. Do not run it against a
database you do not fully control.

Stop with `docker compose down` (add `-v` only if you want to delete the database volume).

## Product catalog (Milestone 2)

All catalog records are **synthetic** (project-authored, deterministic). Nothing is scraped or
downloaded. See `docs/data-quality.md` §2.5 for provenance.

```bash
uv run alembic upgrade head                      # creates the catalog tables (revision 0002)
uv run python scripts/generate_seed_catalog.py --check   # seed regenerates byte-for-byte

# commands that WRITE a database require an explicit --database (no default to POSTGRES_DB)
uv run python -m ecommerce_search.catalog ingest --database <scratch>
uv run python -m ecommerce_search.catalog review-import --database <scratch> ...   # humans only

uv run python -m ecommerce_search.catalog check --file data/seed/catalog_seed_v1.jsonl  # offline
uv run python -m ecommerce_search.catalog check --database <name>        # database audit
uv run python -m ecommerce_search.catalog review-sample                  # offline, blank sample
uv run python -m ecommerce_search.catalog review-status                  # offline
uv run python -m ecommerce_search.catalog review-verify                  # offline
```

* **Explicit targets.** `ingest` and `review-import` need `--database NAME` (argparse exits with
  code 2 otherwise) and print `target database: NAME` before writing. A DB audit
  (`check` without `--file`) also needs `--database`. Use a scratch database while developing;
  never ingest into the development database without deciding to.
* **Ingestion.** Input and provenance are validated before any engine or setting is touched.
  Writes are one transaction guarded by a PostgreSQL advisory lock per dataset id.
  Dataset versions are positive integers compared numerically. Re-running the same version and
  checksum changes nothing; the same version with different bytes is refused; **an older
  version than one already ingested is refused**; a reused version whose recorded provenance or
  transform/taxonomy/rules versions differ is refused (use a new version); a newer version
  upserts changed products and never deletes absent ones. An **empty catalog is an error**.
* **Errors.** Expected failures (bad files, wrong encoding, database errors) print one concise
  line and exit 1. Credentials, URLs and SQL parameters are never shown.
* **Review sample.** `review-sample` writes a blank CSV, a manifest and instructions under
  `data/processed/catalog_review/<batch>/` (git-ignored). It **refuses** if any of those files
  exist; `--force` replaces only a provably blank, untouched sample and **never** a CSV with any
  reviewer input. The manifest binds every sampled product's content hash and a hash of every
  non-review CSV column, so only `verdict`, `issue_fields` and `notes` may be edited.
* **Safe CSV editing.** Back the CSV up before editing. Edit only the three reviewer columns.
  Save as **CSV UTF-8**. Never put an email address, personal information, secrets or machine
  paths in `issue_fields` or `notes`, and use a short handle or pseudonym (never an email) as the
  reviewer name.
* **Evidence.** `review-import --reviewer <handle> --confirm-human-review` first checks that
  the supplied manifest is **exactly** the sample the committed seed and the current sampling
  code generate (a hand-crafted manifest is refused even if its hash is self-consistent), then
  records the review in the append-only `catalog_reviews` table and writes
  `data/labels/catalog_review_<dataset>_v<version>_<batch>.json`. That file **embeds the exact
  reviewed manifest** (seed, selection method/version, quotas, rules/taxonomy/transform
  versions, and each product's content and row hash) plus the outcomes and a canonical
  self-hash, so it is self-contained. It is committed only after a real human review.
  `review-verify` re-checks it offline against the committed seed, and `review-status` reports
  `PENDING` (no evidence), `RECORDED 36/36` or `ERROR`. Historical validity does **not** depend
  on today's rules, selection wording, quotas or seed: later sampling changes never invalidate
  old evidence (comparison with current rules is the informational
  `matches_current_sampling_rules` line). A changed seed checksum or product content, or any
  edit to the evidence, still fails. Git history plus a reviewer handle are an **audit trail
  and integrity check, not cryptographic proof** of a human's identity.
* **Recorded review (Milestone 2).** The required representative human review is
  **recorded: 36/36 products**, reviewer handle `Vishal`, all 36 verdicts `accept` (0
  `needs_correction`, 0 `unsure`), covering 10 laptops, 10 phones, 8 shoes and 8 headphones.
  Evidence: `data/labels/catalog_review_synthetic-seed_v1_c90b5cd0a3a7.json`
  (file SHA-256 `302d725dcf399679977f55a3da98976aa499c003a5236a1772eae42b368e9265`; this is the hash of the file as written with LF line endings, and
  verification itself uses the line-ending-independent canonical `records_sha256` inside the
  file). `review-verify` and `review-status` succeed offline, without a database. The review is
  bound to the exact dataset checksum, the embedded reviewed manifest and every product's
  content hash. It is an integrity/audit artifact; Git history supplies attribution, and it is
  **not cryptographic proof of the reviewer's identity**.
* **What the review does and does not show.** The catalog is synthetic and internally
  coherent. The review does not show real-market availability, pricing or product existence.
  The seed produced zero automated warnings, so the actual sample was stratified by category
  and brand, not by warning types (warnings-first selection is implemented and unit-tested
  with injected findings).

**Warning: downgrading revision `0002` drops all catalog tables and their data.**

## Lexical search (Milestone 3)

V0 is PostgreSQL full-text search over a **derived** index. Nothing about it is tuned or
measured against relevance labels: there is no Golden Dataset yet, so no quality metric is
claimed. Details, tokenization observations and measured latency are in
`docs/search-v0-baseline.md`.

Required sequence for a usable index (each step is explicit; none runs automatically):

```bash
uv run alembic upgrade head                                      # 1. revision 0003 creates the (empty) table
uv run python -m ecommerce_search.catalog ingest --database <db>  # 2. ingest; also builds the documents
uv run python -m ecommerce_search.search reindex --database <db>  #    (or) heal missing/stale documents
uv run python -m ecommerce_search.search status  --database <db>  # 3. verify: expect state=CURRENT
uv run python scripts/benchmark_lexical.py                        # benchmark (scratch databases only)
```

Run `search status` (or `catalog check --database <db>`, check 10 and its `search_index`
section) **before relying on result completeness**. Running the application or the commands
against a database that is not at revision `0003` fails with a sanitized message telling you to
run `uv run alembic upgrade head`; nothing is written. The API itself returns the fixed
`503 {"detail": "search unavailable"}` and logs the same hint (never SQL or credentials).

* **Documents.** One row per product (`product_id`, `document_version`,
  `source_content_sha256`, weighted `search_vector`, `built_at`), built from validated catalog
  fields only: weight A title and brand, B explicit category/subcategory singular and plural
  forms plus model-identifier variants, C canonical technical attributes (`8gb ram`, `256gb`,
  `ssd`, `nvme`, `anc`, ...), D description. Raw seller lines, dataset/provenance data, review
  outcomes, seller ids, prices, availability and ratings are never indexed.
* **Synchronization.** Ingestion writes documents in the same transaction as the catalog rows
  (a failure rolls back both) and leaves current documents, and their `built_at`, untouched.
  `reindex` builds missing and stale documents in one transaction and requires an explicit
  `--database`; it prints only the database name. **A missing or stale index may produce
  incomplete results until `reindex` succeeds.** Search itself never claims completeness; check
  with `search status` or `catalog check --database <db>` (check 10).
* **Search.** `plainto_tsquery('simple', q)`: every term must match (strict AND), no stemming,
  no stopwords, no synonyms. `ts_rank` with initial, untuned weights D=0.1, C=0.2, B=0.4, A=1.0,
  ordered by score then `product_id`. The score is query-relative and not a probability.
  Queries are not interpreted: `8gb`, `hp` and `40k` are plain words. Filler and Hinglish words
  (`ke liye`, `sasta`) are ordinary required terms, so they can empty the result (a V0
  limitation; Milestone 6 query understanding is informational on `/search/hybrid` only and
  does not change `/search`).

```bash
curl "http://127.0.0.1:8000/search?q=hp+laptop&top_k=5"
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json"      -d '{"query": "wireless headphones", "top_k": 5}'
```

`top_k` defaults to 10 and must be between 1 and `SEARCH_LEXICAL_K`; blank, over-long or unsafe
queries are `422`. Unsafe means surrogate code points, control characters (NUL included) and
invisible format characters such as zero-width space, bidirectional overrides or a byte-order
mark; ordinary whitespace is collapsed, and letters of every script (including ZWNJ/ZWJ
joiners used in Persian and Indic text) are accepted. Error text never repeats the input; punctuation-only and unknown queries are `200` with no results;
a database failure is a fixed `503 {"detail": "search unavailable"}`. `result_count` is the number
of results in the response, not the number of database matches. `/health` is unchanged.

**Warning: downgrading revision `0003` drops the derived search table** (rebuild it with
`reindex`); downgrading `0002` still drops all catalog data.

## Dense search (Milestone 4)

Dense-only semantic retrieval (`search_version = "dense_only"`; hybrid V1 is Milestone 5) over a
**derived** embedding table. No relevance or quality claim is made: there is no Golden Dataset
yet, and the catalog is synthetic. Design, freshness rules, API behaviour, the benchmark and the
Docker verification are in `docs/search-dense-m4.md`; the model decision is ADR-006.

* **Model.** `sentence-transformers/all-MiniLM-L6-v2` at immutable revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (384 dimensions, normalized, maximum 256 tokens,
  CPU). It is pinned in code; settings can only select a reviewed entry.
* **Installing torch.** `torch` comes from the official CPU-only wheel index on every platform
  (`[tool.uv.sources]` in `pyproject.toml`), so the environment is large (about 0.9 GB) but has no
  CUDA packages.
* **Network access.** Among the application commands, only `model-fetch` needs the network: it
  downloads the pinned model snapshot. Once the verified snapshot exists, model loading,
  embedding generation and dense serving run locally and offline. Environment setup (`uv sync`)
  may download the locked Python packages, and a fresh Docker image build may pull its tagged
  base images and download the locked packages from the configured indexes. Model weights are
  never downloaded during the image build or baked into the image.

Required sequence (each step is explicit; none runs automatically):

```bash
# once, the only application command that needs the network: weights go to the git-ignored models/ directory
uv run python -m ecommerce_search.search model-fetch --model-id sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
uv run alembic upgrade head                                          # revision 0004 (schema only)
uv run python -m ecommerce_search.catalog ingest --database <db>     # catalog + lexical documents
uv run python -m ecommerce_search.search embed --database <db>       # all missing/stale embeddings
uv run python -m ecommerce_search.search embed-status --database <db> --require-current
```

* **Generation.** `embed` plans, encodes (refusing any text over the 256-token limit; nothing is
  truncated) and writes in one transaction; any failure writes nothing, current rows are left
  untouched, `--all` re-embeds everything. Ingestion never loads the model.
* **Freshness.** Embeddings are content-addressed: a row is stale when the product's content, the
  model, its revision, the embedding configuration or the embedding-text version changes. Stale
  and missing rows are **excluded** from dense results until `embed` heals them, so results may be
  incomplete; `embed-status` (or `--verify-vectors` to re-encode and compare) and
  `catalog check --database` (check `dense_embedding_consistency` and the `dense_index` section)
  report the state. Dataset identity is reported by `catalog check --database`.
* **Search.** Exact cosine distance in pgvector, ordered by `dense_score` (cosine similarity,
  not a probability) then `product_id`. There is no similarity threshold, so every query returns
  up to `top_k` nearest products. `top_k` is 1..`SEARCH_DENSE_K` (default 50); the same text
  validation as `/search` applies; a query without any letter or digit returns no results.
* **Missing model.** If the snapshot is absent or cannot load, `/search/dense` returns the fixed
  `503 {"detail": "search unavailable"}` (the log names only a reason code such as
  `snapshot_missing`); `/search` and `/health` keep working.

```bash
curl "http://127.0.0.1:8000/search/dense?q=noise+cancelling+headphones&top_k=5"
curl -X POST http://127.0.0.1:8000/search/dense -H "Content-Type: application/json" -d '{"query": "running shoes", "top_k": 5}'
```

* **Docker.** Model weights are never copied into the image (`.dockerignore` excludes
  `models/`). Compose mounts the repository `models/` directory **read-only** into the `api`
  service only (`./models:/models:ro`, `EMBEDDING_MODELS_DIR=/models`); `db` and `migrate` get
  neither the mount nor embedding settings. Without a fetched snapshot, dense search returns the
  fixed 503 inside the container while lexical search and `/health` still work. The image is
  about 2 GB because of torch (CPU).
* **Limitations.** 240 synthetic products; English-only model (Hinglish is not understood by
  retrieval; Milestone 6 parses it for an informational report only); the first dense request
  in a process pays the model load (several seconds). The exact-scan decision applies only at the current catalog size.

**Warning: downgrading revision `0004` drops the derived embedding table** (regenerate it with
`embed`). `alembic downgrade base` still drops the pgvector extension and everything before it.

## Hybrid search (Milestone 5)

V1 hybrid retrieval (`search_version = "v1_hybrid"`): up to 50 lexical candidates (as
`/search`) and up to 50 dense candidates (as `/search/dense`) are read from one read-only
database snapshot and fused with Reciprocal Rank Fusion (equal weights, 1-based positional
ranks, ties by `product_id`). The fused list keeps at most `SEARCH_CANDIDATE_K` (50) candidates
and the response returns the first `top_k`. **No relevance or quality claim is made**: there is
no Golden Dataset yet and the catalog is synthetic. Design, API contract, the provisional
`rrf_k` selection, the lexical/dense/hybrid comparison and per-run latency are in
`docs/search-hybrid-m5.md`; the decision record is
[ADR-002](docs/decisions/ADR-002-hybrid-retrieval-rrf.md).

Prerequisites: no new migration or command. Hybrid search needs exactly what the two earlier
endpoints need: the database at revision `0004`, the catalog ingested with current lexical
documents, the fetched MiniLM snapshot and current embeddings (see the required sequences in
"Lexical search (Milestone 3)" and "Dense search (Milestone 4)" above). Check with
`search status --database <db>` and `embed-status --database <db> --require-current`.

```bash
curl "http://127.0.0.1:8000/search/hybrid?q=noise+cancelling+headphones&top_k=5"
curl -X POST http://127.0.0.1:8000/search/hybrid -H "Content-Type: application/json" -d '{"query": "apple phone", "top_k": 5}'
```

* **Metadata.** Every response reports a `fusion` block (`method: "rrf"`, `rrf_k`,
  `rrf_k_status: "provisional"`, `lexical_k`, `dense_k`, `candidate_k`), the source and fused
  counts (`lexical_hit_count`, `dense_hit_count`, `overlap_count`, `fused_count`,
  `candidate_count`) and per result `rrf_score`, `lexical_rank`/`lexical_score` and
  `dense_rank`/`dense_score` (null when the product is absent from that source). `latency_ms`
  adds `rrf_ms` to the lexical and dense stage timings.
* **Provisional `rrf_k`.** `SEARCH_RRF_K` defaults to 100. Every predeclared candidate value
  tied on a deterministic proxy, and the predeclared rule took the largest; this is not
  evidence that 100 is better. Milestone 10 re-decides it on the Golden Dataset.
* **Validation and failures.** Text validation as `/search`; `top_k` is 1..`SEARCH_CANDIDATE_K`
  (default 10). A query without any letter or digit returns no results without running the
  model or the database. If the model snapshot is missing or cannot load, or the database
  fails, the endpoint returns the fixed `503 {"detail": "search unavailable"}`: it never falls
  back silently to lexical-only results. Missing or stale embeddings shorten the dense list, so
  results may be incomplete.
* **Limitations.** The dense side has no similarity threshold, so hybrid search returns results
  even for nonsense queries; the model is English-only; scores are uncalibrated and not
  probabilities.

## Query understanding (Milestone 6)

A deterministic parser (`PARSER_VERSION = "qu-1"`, `LEXICON_VERSION = "1"`) reads the normalized
query into a typed `QueryUnderstanding`: category, brand, RAM, storage capacity and type, NVMe
interface, price bounds, `sasta` price preference and one semantic intent, each backed by
matched terms with exact source spans, plus closed-reason ambiguities, conflicts and unresolved
spans. It runs locally with no model, database, network or setting, through the
`DecisionProvider` interface whose only implementation is the deterministic provider
([ADR-003](docs/decisions/ADR-003-decision-provider.md)). Rules, examples, unsupported forms and
evidence are in [`docs/query-understanding-m6.md`](docs/query-understanding-m6.md).

* **Informational only.** `GET`/`POST /search/hybrid` add a `query_understanding` block
  (`usage: "informational"`, `provider`, `provider_version`, `understanding`) and
  `latency_ms.query_understanding_ms`. The parse never reaches retrieval: lexical and dense
  queries, RRF, ranks, scores and `rrf_k` are exactly as in Milestone 5, and `applied_filters`
  stays `[]`. Whether parsed constraints become filters is decided in Milestone 7.
* **Failure.** If query understanding fails, `/search/hybrid` returns the fixed
  `503 {"detail": "search unavailable"}` before any model or database work and logs only
  `query_understanding_failed`; it never returns results without the block.
* `/search` and `/search/dense` are unchanged. No relevance or extraction-accuracy claim is
  made: the rules are developer-authored and there is no Golden Dataset yet.

## `GET /health`

Readiness-oriented. Returns `200` with `status: "ok"` when PostgreSQL is
reachable and the pgvector extension is installed, and `503` with
`status: "unavailable"` otherwise. The body is a typed `HealthResponse`; failure
details are fixed generic strings (no connection strings, credentials or
stack traces).

## Validate

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not integration and not real_model"   # unit + API tests: no Docker, .env or model
docker compose up -d db
uv run alembic upgrade head
uv run pytest -m "integration and not real_model"       # needs the compose database (scratch DBs only)
uv run pytest -m real_model    # offline smoke tests with the fetched MiniLM snapshot (and the database)
docker compose config -q
```

Unit tests are hermetic: they never read `.env` or ambient `POSTGRES_*` settings, and a
regression test runs the whole unit suite with a blank password.
Integration tests fail (they are not skipped) if the database is not running.
They create and drop throwaway databases and never touch the development database. The
`real_model` tests also fail (they are not skipped) if the MiniLM snapshot is missing; they never
use the network. Every other test uses a deterministic fake embedder.

## Versions

PostgreSQL 16 with pgvector 0.8.6 (`pgvector/pgvector:0.8.6-pg16`). Python
dependencies are pinned in `uv.lock`.

## Security notes (local development only)

- The Compose database user is a **superuser** so the migration can run
  `CREATE EXTENSION vector`. This is not the intended production security model.
- Postgres and the API are published on `127.0.0.1` only.
- Use a local-only password; never commit `.env`.
- The host default is `127.0.0.1`, not `localhost`: on Windows `localhost` can try
  IPv6 first and add ~5 s to every connection while Docker publishes IPv4 only.
