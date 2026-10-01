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
below). There is no search, embedding or model code yet.

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
uv run pytest -m "not integration"     # unit + API tests, no Docker, no .env, no password
docker compose up -d db
uv run alembic upgrade head
uv run pytest -m integration           # needs the compose database running
docker compose config -q
```

Unit tests are hermetic: they never read `.env` or ambient `POSTGRES_*` settings, and a
regression test runs the whole unit suite with a blank password.
Integration tests fail (they are not skipped) if the database is not running.
They create and drop throwaway databases and never touch the development database.

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
