# ADR-003: DecisionProvider abstraction

## Status

Accepted

## Date

2026-09-30

## Context

Query understanding and listing validation involve bounded decisions such as
category, semantic intent, retrieval strategy, ambiguity and review routing.
Deterministic logic handles many of these reliably. Some may later benefit
from a remote model. The core must work without any remote provider, and
provider-specific code must stay isolated behind typed interfaces.

## Decision

Define a small provider interface, conceptually:

```text
understand(query, deterministic_result) -> DecisionResult
```

- `DeterministicDecisionProvider` is the default implementation. It is local
  and needs no network or credentials.
- `JevDecisionProvider` (implemented in Milestone 11, ADR-005) and an optional
  `LLMDecisionProvider` (experimental comparison only) may be added later.
- `DecisionResult` is a Pydantic model. Each decision belongs to a fixed
  decision type with a closed label set and carries provider name, model
  version (where applicable), latency and fallback reason, together with the
  two distinct values below.
- Providers propose. A separate confidence gate decides whether a decision is
  accepted, used only softly, or discarded in favor of the deterministic
  result.
- Any provider failure (disabled, timeout, transport error, schema-invalid
  output, out-of-set label) produces the deterministic result with a recorded
  fallback reason.
- Providers receive the deterministic result and must not replace reliable
  deterministic numeric parsing (RAM, storage, price).

### Raw provider probability versus gate confidence

- **Raw provider probability** is provider output, stored exactly as returned.
  Its meaning is provider-specific and is not assumed to be a calibrated
  probability of correctness.
- **Gate confidence** is the value the confidence policy consumes. It is
  derived from the raw provider probability only through a mapping whose
  meaning and calibration have been verified (Milestone 12 for Jev).
  Deterministic providers set gate confidence by their own documented rules.
- The two are stored and logged as separate fields. Neither overwrites the
  other.

### Hard-filter rule

- Explicit deterministic extraction may become a hard filter when parsed with
  high reliability.
- A model-derived decision may become a hard filter only after a confidence
  threshold has been justified from human-reviewed data in Milestone 12.
- Before that validation, model-derived decisions may influence ranking or
  retrieval routing but not hard filtering.

The interface is created in Milestone 6 with only the deterministic provider.
It is kept minimal and not extended ahead of need.

### Implementation status (Milestone 6)

Implemented in `src/ecommerce_search/decision/provider.py`:

- `DecisionProvider` is a typed Protocol:
  `understand(query: str, deterministic_result: QueryUnderstanding) -> DecisionResult`.
- `DeterministicDecisionProvider` is the only implementation. It is local, needs no network,
  credentials, model or setting, and returns the deterministic parse unchanged.
- `DecisionResult` is a frozen Pydantic model that forbids extra fields and carries only
  `provider` (`deterministic`), `provider_version` (`qu-1`) and `understanding`. No raw
  probability, gate confidence, latency, fallback-reason or decision-type fields exist yet; they
  are added with the first provider that produces them (Milestone 11), not as placeholders.
- `understand_normalized_query` parses once, calls the provider once and checks that the result
  is a `DecisionResult` for the same normalized query; otherwise it raises
  `QueryUnderstandingFailed` (fixed message, no query text).
- Failure boundary: the deterministic provider is itself the fallback this ADR describes, so in
  M6 there is nothing further to fall back to. `/search/hybrid` turns any failure into the fixed
  `503 {"detail": "search unavailable"}`, logs only `query_understanding_failed` and runs no
  model or database work; it never returns results without the parse.
- The decision is informational only (`docs/query-understanding-m6.md`): it does not change
  retrieval, ranking or filtering.

The Jev provider (Milestone 11, ADR-005), its fallback semantics and the confidence gate remain
deferred.

## Alternatives considered

- **Calling a model directly inside the parser or search code:** couples core
  search to a remote provider and blocks offline operation. Rejected.
- **Model-first understanding with deterministic fallback:** puts inference
  ahead of reliable extraction and contradicts the principles. Rejected.

## Consequences

### Positive

- The core search path is testable offline and in CI without credentials.
- Providers can be compared on identical inputs (deterministic, LLM, Jev,
  gated Jev).
- Provider-specific request/response code lives only in its provider module.

### Negative

- Adds a thin layer of indirection, justified by the comparison and fallback
  requirements.
- Keeping raw probability and gate confidence separate adds fields and a
  mapping step to maintain.

## Revisit when

- A second real provider shows the interface is too narrow or too broad.
- Milestone 12 shows the gate-confidence mapping needs a different shape.

## References

- `docs/MASTER_PLAN.md` (decision-provider and confidence-gating sections).
- ADR-005 for the Jev-specific boundary. No external reference is cited.
