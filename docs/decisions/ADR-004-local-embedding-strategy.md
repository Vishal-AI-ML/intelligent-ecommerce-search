# ADR-004: Local embedding strategy

## Status

Accepted (strategy). Model selection deferred to Milestone 4.

## Date

2026-09-30

## Context

Dense retrieval needs text embeddings for catalog products and queries. A
paid embedding API must not be a core dependency. Embeddings must be
reproducible and tied to the exact catalog version they were generated from,
so evaluation results stay interpretable.

## Decision

- Use a configurable, local, sentence-transformers-compatible embedding
  model. Inference runs on the local machine. No paid remote API is required.
- Store vectors in pgvector (ADR-001).
- Record embedding metadata with every generated set: model name, model
  revision, dimensions, normalization, catalog version and generation
  timestamp.
- Generate the same kind of representation for queries at search time, with
  the same model and normalization.
- Embedding input text is built only from approved catalog fields and never
  includes evaluation labels.
- Any change to model, revision, normalization, input-text construction or
  catalog version requires regenerating embeddings and rerunning evaluation.

The specific model, its dimensions, the similarity metric and the vector index
type are **not chosen in this ADR**. They are selected in Milestone 4 from
measured retrieval quality, generation time and query latency on the actual
catalog, and documented there.

### Model directory layout

- Root-level `/models/` (at the repository root) is for downloaded model
  weights and caches. These are not source code and are git-ignored.
- `src/models/` (if it exists) is for source code, such as Pydantic or
  SQLAlchemy model classes. It is source code and must be tracked.
- The former `.gitignore` entry `models/` was unanchored and would also match
  `src/models/` at any depth. **Done in Milestone 1:** it is now `/models/`.
  (The source package is `src/ecommerce_search/models/` if it is ever created.)

## Selection criteria (Milestone 4)

- Retrieval quality on the evaluation data available at that time. Before
  Milestone 9 this is provisional, non-authoritative smoke evaluation only.
- Query-embedding latency and catalog generation time on the target hardware.
- Model size and memory footprint on the target hardware.
- License that permits this use.
- Handling of the query mix, including short queries and Hinglish.

## Alternatives considered

- **Paid remote embedding API:** makes the core depend on network, cost and a
  third party. Rejected as a core dependency.
- **No dense retrieval (lexical only):** kept as the V0 baseline. Weak on
  semantic intent.

## Consequences

### Positive

- Offline, reproducible embeddings, and no per-request embedding cost.
- Model upgrades are explicit, versioned events with re-evaluation.

### Negative

- Model download and local compute are needed. Model weights live under
  root-level `/models/` and are git-ignored.
- A Milestone 4 model choice made before Milestone 9 rests on provisional
  evaluation and may need revisiting.

## Revisit when

- Authoritative evaluation on the human-reviewed Golden Dataset (Milestone 10)
  disagrees with the Milestone 4 model choice.
- Local hardware cannot meet measured latency or memory needs.

## References

- `docs/MASTER_PLAN.md` (local embedding requirement). No specific model,
  benchmark or license has been verified for this ADR. Those are recorded in
  Milestone 4.
