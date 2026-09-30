# Intelligent E-commerce Search & Decision Engine

A production-oriented e-commerce search and bounded-decision system.

Development is organized milestone by milestone. See
`docs/MASTER_PLAN.md` for the complete engineering plan.

## Current status

Milestone 1 (Foundation) only: typed settings, FastAPI app with `GET /health`,
PostgreSQL 16 + pgvector via Docker Compose, SQLAlchemy 2 (sync) with psycopg 3,
and Alembic (one migration that enables the `vector` extension). There is no
catalog, search or model code yet.

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

## `GET /health`

Readiness-oriented. Returns `200` with `status: "ok"` when PostgreSQL is
reachable and the pgvector extension is installed, and `503` with
`status: "unavailable"` otherwise. The body is a typed `HealthResponse`; failure
details are fixed generic strings (no connection strings, credentials or
stack traces).

## Validate

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not integration"     # unit + API tests, no Docker needed
docker compose up -d db
uv run alembic upgrade head
uv run pytest -m integration           # needs the compose database running
docker compose config -q
```

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
