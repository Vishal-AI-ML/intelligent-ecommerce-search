# ADR-005: Jev as an optional bounded-decision provider

## Status

Accepted (architectural boundary only).

Provider implementation: not started; API behaviour unverified.

## Date

2026-09-30

## Context

Jev access is reported as available. `docs/MASTER_PLAN.md` describes Jev as a
probability-backed bounded-decision layer. The core search engine must work
without Jev or any other remote model, and model decisions must not become
hard filters or final judgments without confidence gating and human review.

This ADR records only the role and boundary defined in the master plan. It
does not document Jev API endpoints, model names, pricing, the meaning of
Jev's returned probabilities, or implementation details. Those are unverified.
API verification is deferred to Milestone 11, and evaluation of the
probabilities' meaning and calibration to Milestone 12.

## Decision

Jev may be integrated only as an optional `JevDecisionProvider` behind the
`DecisionProvider` interface (ADR-003), and only after its prerequisites are
met.

### Milestone order

- **Milestone 11** implements the optional provider: typed client, timeout,
  limited retry, safe fallback, configuration and telemetry.
- **Milestone 12** evaluates it against human-reviewed decision data.
- Jev is enabled in the default/V4 configuration **only if** Milestone 12
  demonstrates a measured benefit and acceptable safety, latency and cost.
  Otherwise the default stays deterministic and Jev remains an optional,
  disabled provider.

### Prerequisites (Milestone 11)

- Stable V0–V3.
- A human-reviewed Golden Dataset.
- Credentials available through environment variables.
- Official API behavior verified.

### Permitted bounded decisions

- Query category
- Semantic intent
- Retrieval strategy
- Ambiguity
- Out-of-catalog detection
- Listing-category validation
- Attribute-candidate validation
- Policy-risk triage
- Human-review routing

### Prohibited uses

Jev must not be used to:

- invent attributes;
- replace reliable deterministic numeric parsing;
- make final legal conclusions;
- automatically ban sellers or reject listings;
- create evaluation or review labels;
- replace the reranker without a controlled experiment.

### Boundary rules

- Jev outputs are validated through Pydantic into bounded, closed-set
  decisions. Anything else is discarded, with a recorded fallback reason.
- Every Jev decision passes the confidence gate.
- Hard-filter rule: before Milestone 12 validates a confidence threshold from
  human-reviewed data, Jev decisions may influence ranking or retrieval
  routing but never hard filtering. Low-confidence decisions never become
  hard filters at any time.
- **Raw probability is not gate confidence.** A Jev raw probability is
  provider output. It must not be used as gate confidence until its meaning
  and calibration are verified in Milestone 12 (see ADR-003). Raw
  probabilities and gate-confidence values are logged separately.
- Any failure falls back to the deterministic provider. The search result is
  still returned.
- Configurable: endpoint, model, timeout, retry, enablement and thresholds.
  Nothing is hard-coded.
- Credentials only in environment variables. Secrets are never logged.
- Log: provider, pinned model version, latency, raw provider probabilities and
  gate-confidence values (as separate fields), fallback reason and returned
  cost where available.
- Seller and query text sent to Jev is treated as untrusted data.
- Routine CI never requires live Jev access.

## Alternatives considered

- **Jev in the core path as a required dependency:** violates offline
  operation and the fallback requirements. Rejected.
- **No model-based decisions at all:** remains the default behavior. Jev is
  enabled by default only if the Milestone 12 experiment shows a measured
  benefit.

## Consequences

### Positive

- The core stays fully functional and testable without Jev.
- Jev's value is decided by measurement, not assumed.

### Negative

- Jev's value must be demonstrated in Milestone 12 against deterministic,
  optional LLM and gated-Jev baselines, using human-reviewed decision data,
  calibration, coverage, latency and cost. This is extra work that may
  conclude Jev is not worth enabling.
- Until Milestone 11, all Jev-specific behavior is an open question. Until
  Milestone 12, probabilities cannot be used as gate confidence.

## Revisit when

- Milestone 11 verification of the official API contradicts an assumption in
  this ADR.
- Milestone 12 shows no measured benefit, or unacceptable safety, latency or
  cost.

## Open questions

Deferred to Milestone 11 (API verification):

- Official API contract, authentication and error behavior.
- Available models and version pinning.
- Pricing and cost reporting.

Deferred to Milestone 12 (evaluation):

- Semantics and calibration of returned probabilities.

## References

No provider-specific reference has been verified. API verification is
deferred to Milestone 11. `docs/MASTER_PLAN.md` is the only source for the
role and boundaries recorded here.
