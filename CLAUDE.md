# Intelligent E-commerce Search & Decision Engine

## Primary specification

Read `docs/MASTER_PLAN.md` completely before planning or implementing any milestone.

## Absolute execution rule

Implement exactly one milestone at a time.

For every milestone:

1. Inspect the current repository.
2. Read the relevant specification and architecture documents.
3. Prepare a plan before editing.
4. Implement only the requested milestone.
5. Write or update relevant tests.
6. Run the relevant tests, Ruff, and validation commands.
7. Fix failures caused by the milestone.
8. Update relevant documentation.
9. Report every file changed.
10. Report every command executed and its actual result.
11. Report limitations and unresolved questions honestly.
12. Stop.

Do not automatically start the next milestone. Do not create placeholder code for future milestones.

## Locked technology stack

- Python 3.12
- uv
- FastAPI
- PostgreSQL
- pgvector
- PostgreSQL Full Text Search
- SQLAlchemy
- Alembic
- Pydantic
- pytest
- Ruff
- Local sentence-transformers-compatible embeddings
- Configurable local cross-encoder reranker

## Critical restrictions

- Never fabricate tests, metrics, benchmarks, dataset provenance, pricing, or integrations.
- Never claim a command passed unless it was executed successfully.
- Never create fake human-reviewed labels.
- Never hard-code credentials or commit `.env` files.
- Do not add Elasticsearch, OpenSearch, Qdrant, Pinecone, Kafka, Airflow, or Kubernetes without a measured requirement.
- Do not use an LLM where deterministic logic is sufficient.
- The core search engine must work without Jev or another remote model.
- Do not turn low-confidence model decisions into hard filters.
- Do not automatically reject listings or ban sellers based only on model output.
- Do not build the frontend or AWS infrastructure before their milestones.
- Preserve raw seller data separately from normalized or inferred values.
- Keep provider-specific code isolated behind typed interfaces.

## Git and scope discipline

- Review `git diff` before every commit.
- Prefer small, milestone-scoped commits.
- Do not alter unrelated files.
- Do not silently change architecture or scope.
- Ask before destructive database, Git, cloud, or billing operations.
