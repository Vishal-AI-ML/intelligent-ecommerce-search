# First Claude Code Prompts

## Prompt 1 — Plan Milestone 0

```text
Read CLAUDE.md and docs/MASTER_PLAN.md completely.

Inspect the current repository.

We are starting only Milestone 0 — Specification.

Do not modify any files yet.

First produce an implementation plan containing:

1. Files you intend to create or modify.
2. Required sections in each document.
3. Architectural decisions that need to be documented.
4. Assumptions and unresolved questions.
5. Validation commands you will run.
6. Explicit confirmation of what will not be implemented.

Do not write application code.
Do not create placeholder application directories.
Do not begin Milestone 1.
Stop after presenting the plan.
```

## Prompt 2 — Implement Milestone 0

Send this only after reviewing and approving the plan.

```text
The Milestone 0 plan is approved.

Implement only Milestone 0 according to CLAUDE.md and
docs/MASTER_PLAN.md.

Create only:

- docs/spec.md
- docs/architecture.md
- docs/data-quality.md
- ADR-001 PostgreSQL + pgvector
- ADR-002 Hybrid Retrieval with RRF
- ADR-003 DecisionProvider abstraction
- ADR-004 Local embedding strategy
- ADR-005 Jev as an optional bounded-decision provider

Do not write application code.
Do not create future milestone directories.
Do not call Jev.
Do not generate evaluation data or metrics.

After implementation:

1. Validate the documentation.
2. List every created or modified file.
3. Report every command executed and its result.
4. Report unresolved assumptions.
5. Stop.
```

## Prompt 3 — Review Milestone 0

```text
Review the current Milestone 0 implementation as a strict senior
search, backend, and applied-ML engineer.

Do not modify files yet.

Check:

- compliance with CLAUDE.md
- compliance with docs/MASTER_PLAN.md
- missing requirements
- architectural contradictions
- unnecessary complexity
- premature future work
- unsupported or fabricated claims
- missing acceptance criteria

Classify findings as:

- Blocking
- Important
- Optional

Include exact file paths and section names.
Stop after the review.
```

## Prompt 4 — Fix review findings

```text
Fix only the Blocking and Important findings from the previous review.

Do not begin Milestone 1.
Validate the documents again, report exact changes and commands,
and stop.
```

## Prompt 5 — Plan Milestone 1

Use this in a fresh Claude Code session after committing Milestone 0.

```text
Read CLAUDE.md, docs/MASTER_PLAN.md, and all documents created in
Milestone 0.

Inspect the complete repository.

Prepare a plan for only Milestone 1 — Foundation.
Do not modify files yet.

The plan must cover:

- Python 3.12 and uv
- FastAPI application structure
- configuration management
- PostgreSQL and pgvector
- SQLAlchemy
- Alembic
- Docker Compose
- GET /health
- pytest
- Ruff
- validation commands

Do not implement the catalog, ingestion, search, embeddings, query
parsing, Jev, frontend, or AWS.
Stop after presenting the plan.
```
