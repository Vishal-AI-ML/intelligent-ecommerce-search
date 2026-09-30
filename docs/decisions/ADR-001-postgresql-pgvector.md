# ADR-001: PostgreSQL + pgvector as the search datastore

## Status

Accepted

## Date

2026-09-30

## Context

The system needs a catalog store, structured filtering, lexical full-text
search and dense vector search. The project values simple, measurable
architecture and prohibits adding search infrastructure without a measured
need.

**Provisional assumption:** the expected catalog is small enough to run and
evaluate locally on a single PostgreSQL instance. No catalog size has been
measured or chosen yet. The dataset is selected in Milestone 2
(`docs/data-quality.md` §2), and this assumption is confirmed or revised from
the actual catalog size and measured latency (Milestones 3 and 4).

## Decision

Use a single PostgreSQL database with:

- relational tables for the catalog, with typed columns for frequently
  filtered fields and validated JSONB for long-tail attributes;
- PostgreSQL Full Text Search for lexical retrieval;
- the pgvector extension for dense vectors and vector similarity search;
- SQLAlchemy for data access and Alembic for migrations.

Structured filters are pushed into SQL when safe and beneficial.

## Alternatives considered

- **Elasticsearch / OpenSearch:** strong lexical search, but adds a second
  system to operate and keep in sync. Rejected unless measurements show
  PostgreSQL FTS is insufficient.
- **Dedicated vector databases (Qdrant, Pinecone):** add another store and,
  for hosted options, a remote dependency. Rejected unless measurements show
  pgvector is insufficient.
- **Multiple databases for one workload:** rejected. It creates consistency
  and operational burden with no measured benefit.

## Consequences

### Positive

- One transactional store for catalog, lexical and vector data, which
  simplifies consistency, filtering and local development.

### Negative

- Lexical ranking quality depends on PostgreSQL FTS configuration. It must be
  measured (V0), not assumed.
- Vector index choice and parameters are not decided here. They are decided in
  Milestone 4 from measurements.
- Deployment requires a PostgreSQL instance with pgvector available (local
  Docker in Milestone 1, AWS PostgreSQL with pgvector in Milestone 19).

## Revisit when

- Measured quality or P95/P99 latency at the actual catalog size is
  insufficient and cannot be fixed by indexing, query or configuration
  changes.
- The actual catalog size differs materially from the provisional local-scale
  assumption above.

## References

- PostgreSQL Full Text Search and the pgvector extension are named in
  `docs/MASTER_PLAN.md`. No external benchmark or version has been verified
  for this ADR. Versions are pinned in Milestone 1.
