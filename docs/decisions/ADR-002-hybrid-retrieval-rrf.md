# ADR-002: Hybrid retrieval with Reciprocal Rank Fusion

## Status

Accepted

## Date

2026-09-30

## Context

Lexical search handles brands, model names, exact terms and specifications
well, but misses paraphrase and intent. Dense retrieval handles semantic
intent, but can return similar products that violate explicit constraints.
Lexical and dense scores are on different, uncalibrated scales, so combining
them directly is fragile.

## Decision

Retrieve candidates independently from lexical (PostgreSQL FTS) and dense
(pgvector) retrieval, then fuse them with Reciprocal Rank Fusion, which uses
ranks only:

```text
rrf_score(d) = sum over sources s where d appears of 1 / (rrf_k + rank_s(d))
```

Here `rank_s(d)` is the 1-based rank of document `d` in source `s`.

Configuration (all configurable):

| Parameter | Initial value |
|---|---|
| `lexical_k` | 50 |
| `dense_k` | 50 |
| `candidate_k` | 50 |
| `rrf_k` | **Unspecified.** Chosen from measured evaluation |

No default for `rrf_k` is assumed in this ADR, and the RRF paper (see
References) is cited only for the algorithm and formula, not as
justification for any value.

### `rrf_k` selection order

Human-reviewed labels do not exist until Milestone 9, so `rrf_k` is chosen in
two stages:

1. **Provisional (Milestone 5).** Milestone 5 selects a provisional `rrf_k`
   using provisional, non-authoritative smoke queries. The value is recorded
   in configuration and explicitly marked **provisional**. It is not chosen in
   this ADR.
2. **Authoritative (Milestone 10).** Milestone 10 re-runs V0 through V3 on the
   human-reviewed Golden Dataset (Milestone 9). Any change to `rrf_k` and the
   tuning-versus-reporting rules follow `docs/spec.md` §9.

Reports must distinguish provisional tuning from authoritative evaluation.

Each fused result preserves its lexical rank, dense rank (either may be
absent) and fused score for debugging and evaluation. Safe structured filters
(V2) and optional reranking (V3) operate on top of this.

## Alternatives considered

- **Lexical only (V0) or dense only:** kept as measured baselines, not as the
  target.
- **Weighted linear combination of normalized scores:** needs score
  normalization and weight tuning across uncalibrated scales. Not chosen
  initially. It may be compared later if measurement justifies it.
- **Reranker only over a single retriever:** loses candidates that only one
  retriever finds.

## Consequences

### Positive

- Robust to score-scale differences. Simple to implement and test.
- The comparison of lexical, dense and hybrid retrieval is a Milestone 5
  deliverable.

### Negative

- Ignores score magnitudes, so a very strong single-source match is not
  boosted beyond its rank.
- Fusion quality depends on `rrf_k` and the `k` values. These must be tuned
  from measurement, and early tuning is provisional until Milestone 10.

## Revisit when

- Measured evaluation shows that another fusion method improves quality on
  the human-reviewed Golden Dataset at acceptable latency.
- Authoritative evaluation in Milestone 10 disagrees with the provisional
  `rrf_k`.

## Milestone 5 executed decision (2026-10-02)

This section records what Milestone 5 did. The design history above is unchanged.

- **Runtime value:** `rrf_k = 100` (`SEARCH_RRF_K`), reported in every hybrid response as
  `fusion.rrf_k_status = "provisional"`. Source depths `lexical_k = dense_k = 50` and
  `candidate_k = 50` keep their initial values and were not tuned.
- **How it was chosen.** Phase S, experiment `hybrid-m5-s-20261002T130708Z-2c997cb9` (harness
  commit `c59475cbf9a590fb5a1757b5fc82d624b346f22b`, pinned production commit
  `f7cb2757e669cb58b281c9ac3d8a37b6da7e1a7a`), fused the same lexical and dense lists for the
  predeclared grid `[1, 5, 10, 20, 40, 60, 100]` over 17 frozen queries (15 checked, 2
  diagnostic) in two identical runs. Every grid value scored 1.0 on the provisional
  deterministic attribute-consistency proxy, so the predeclared rule (exact tie: largest tied
  `rrf_k`) selected 100. The value was adopted in commit
  `86f7b68f39155a002390bffc110036ad75dbb326`.
- **What it is not.** The proxy was saturated. The result is **not evidence that 100 is better**
  than any other grid value; it is a tie-break. No relevance or quality claim follows from it.
- **Phase CL does not re-select it.** The comparison experiment
  `hybrid-m5-cl-20261002T155427Z-8636430a` (harness commit
  `691600937ad3cbe6abb47ae6839683726d72fd81`, runtime commit
  `86f7b68f39155a002390bffc110036ad75dbb326`) ran only at `rrf_k = 100` and is a description of
  that configuration, not a second selection.
- **Re-decision.** Milestone 10 re-decides `rrf_k` (and may revisit the source depths and
  `candidate_k`) on the human-reviewed Golden Dataset, following `docs/spec.md` §9.3.
- **Artifacts** (git-ignored, not committed; SHA-256):

  | Experiment | `.json` | `.md` | `.samples.jsonl` |
  |---|---|---|---|
  | Phase S | `5dd84db54496ebb008d02e48d6a3631318be52ef13d91ad25f10791b27e2cea1` | `5a1e53b7cca5075ae698507fddde5f8465b22675db18ebdc60c75dcf213de345` | `733f11e3a646c18daad0fb0d3718254087aa4490baec6d739054e1a4c42dd8a4` |
  | Phase CL | `9a0626dd6564df527235076c05173dc01ffe608b519b4a5e70b301ee656f4d3e` | `008f2811300d92219215e59749e5962c1d76b6c73a3b04c7aa314d915ecb7653` | `c313fbe80887544471ba0f48b7c7cf11c963781fab0035ddebba0cabf66253de` |

Details, protocol and measurements: `docs/search-hybrid-m5.md`.

## References

- Cormack, Clarke and Büttcher (2009), "Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods", SIGIR. Cited only for the
  algorithm and formula, not as justification for a default `rrf_k`.
