# Data Quality and Dataset Strategy

- Status: Milestone 0 design (no dataset selected, no data created)
- Date: 2026-09-30
- Related: `docs/spec.md`, `docs/architecture.md`

## 1. Purpose

Search metrics are only as trustworthy as the catalog and labels behind them.
This document defines how the catalog dataset is chosen and recorded, which
automated checks must pass before evaluation, and what humans must review.

## 2. Dataset selection (undecided)

**The initial dataset is deliberately not selected in Milestone 0.** It will
be chosen and recorded in Milestone 2, before ingestion.

### 2.1 Allowed options

1. A reproducible public dataset.
2. A clearly labeled synthetic/seed catalog.
3. An allowed combination of the two, with every record's origin identifiable.

Synthetic data is never described as real marketplace data.

### 2.2 Selection criteria

- Covers the initial categories: laptops, phones, shoes, headphones.
- Contains, or can deterministically derive, the fields required by the
  schema (`docs/spec.md` FR-CAT-1/2) without inventing values.
- Supports the query types in the Golden Dataset: unconstrained,
  brand/category, specification, price, semantic intent and Hinglish.
- Has plausible prices in a single currency, or a documented currency field.
- Is small enough to ingest, embed and evaluate locally.
- Is reproducible: a fixed version, snapshot or checksum can be recorded.

### 2.3 Licensing and provenance requirements

A dataset may be used only if:

- its license or terms of use permit this use and are recorded verbatim or
  linked;
- its source, version or snapshot date, and retrieval method are recorded;
- any personal data is absent or removed before use;
- any transformation from source to catalog is scripted and repeatable;
- synthetic generation, if used, is described (method, seed or generator
  version) and every synthetic record is flagged.

If licensing cannot be confirmed, the dataset is not used.

### 2.4 Provenance record (per dataset version)

| Field | Meaning |
|---|---|
| `dataset_id` / `dataset_version` | Stable identifier and version |
| `source` | Origin (URL, publisher, or "synthetic") |
| `license` | License name and reference |
| `retrieved_at` | Retrieval or generation date |
| `checksum` | Hash of the raw input |
| `transform_version` | Version of the ingestion/normalization code |
| `synthetic` | Whether records are synthetic, per record |
| `notes` | Known gaps and caveats |

## 3. Automated quality checks

These run in Milestone 2 ingestion and must be evaluated before any search
evaluation. Each finding is reported with the product ID and the check name.

| # | Check | Intent |
|---|---|---|
| 1 | Product IDs are unique | No duplicate primary keys |
| 2 | Required fields exist | Core fields present and non-empty where required |
| 3 | Prices and numeric attributes are plausible | Non-negative prices. Numeric attributes within ranges defined per category in Milestone 2 |
| 4 | Category/attribute combinations are coherent | No attributes that do not belong to the category, such as `ram_gb` on shoes |
| 5 | Titles do not obviously contradict structured attributes | For example a title saying "16GB" when `ram_gb=8` |
| 6 | Units and storage types are recognized | Only known units and `storage_type` values. The value set depends on the storage taxonomy decision (NVMe is an interface, normally on SSDs. Milestone 6 or 7 decides how it is represented, see `docs/spec.md` §14, item 9) |
| 7 | Brand/category combinations are plausible | Against the brand dictionary |
| 8 | Duplicates do not dominate | Near-duplicate share reported. The acceptable level is set in Milestone 2 from the actual data |
| 9 | Synthetic records are identified | Every record has a synthetic flag |
| 10 | Embedding text contains no evaluation labels | Search and embedding text is built only from catalog fields |

Outcome levels:

- **Error:** blocks ingestion or evaluation.
- **Warning:** recorded and reported, record retained.

**Provisional:** which checks are errors and which are warnings is not
decided here. The examples that follow are proposals only: checks 1, 2, 9 and
10 as errors, and borderline plausibility or a suspected title contradiction
as warnings. The severity of every check is finalized in Milestone 2.

Ranges, tolerances and duplicate thresholds depend on the dataset, are
provisional, and are chosen in Milestone 2 from the selected dataset and
documented there. None are assumed here.

## 4. Raw data preservation

- Raw input is stored exactly as received, separately from normalized
  values.
- Normalization is deterministic, versioned and repeatable from raw.
- Inferred values (model decisions) are stored separately, with provider,
  model version and confidence.
- Nothing overwrites raw seller data.

## 5. Human review

- At least 30 representative products are prepared for **real human
  review** in Milestone 2, covering every category and a range of quality
  findings.
- Review completion is recorded only by a human. It is never marked complete
  automatically, and no reviewed labels are generated.
- Golden Dataset relevance labels (`relevant`, `borderline`,
  `not relevant`) and decision-evaluation labels are human-produced.
  Model-generated labels may be used only as unverified suggestions and are
  never authoritative.
- Search annotation is done in the Milestone 9 annotation tool, whose
  requirements are in `docs/spec.md` §6.7. Gold constraints used for
  Constraint Satisfaction Rate are also human-annotated
  (`docs/spec.md` §9.2).
- Until the Golden Dataset exists, any queries used in Milestones 3 to 8 are
  provisional smoke queries. They carry no authoritative labels
  (`docs/spec.md` §9.1).

## 6. Evaluation leakage controls

- Golden queries and labels live under `evals/`, separate from catalog data.
- Embedding text and FTS text never include labels or query-specific hints.
- Tuning versus reporting leakage: a configuration tuned on the queries used
  for final reporting gives optimistic metrics. Authoritative reports must use
  a recorded tuning/reporting split, use cross-validation, or explicitly
  disclose that the configuration was tuned on the reporting set. The
  preferred approach is to freeze the primary configuration before final
  reporting (`docs/spec.md` §9.3).
- Candidate-pool bias in pooled Recall@K is disclosed in reports
  (`docs/spec.md` §9.2).

## 7. Versioning

Catalog version, dataset version, golden-set version and normalization-rule
version are recorded in every experiment. Changing any of them requires
regenerating dependent artifacts (such as embeddings) and rerunning
evaluation.

## 8. Acceptance criteria (for Milestone 2 onward)

- [ ] Dataset selected with a complete provenance record and confirmed license.
- [ ] All ten automated checks implemented and tested.
- [ ] Quality report generated from the actual ingested data.
- [ ] Synthetic records flagged. No synthetic data described as real.
- [ ] 30-product sample prepared, with review status pending until a human
  completes it.
