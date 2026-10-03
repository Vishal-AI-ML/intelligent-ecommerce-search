# Milestone 6: deterministic query understanding (informational)

- Status: Milestone 6 implemented. `PARSER_VERSION = "qu-1"`, `LEXICON_VERSION = "1"`, decision
  provider `deterministic` (`provider_version = "qu-1"`).
- Related: ADR-003 (decision-provider interface), `docs/spec.md` §6.2, §6.5, §7, §7.1 and §14
  (Q5, Q6, Q9), `docs/architecture.md` §3.1, §5.4 and §6, `docs/search-hybrid-m5.md` (the
  unchanged retrieval this milestone reports on).
- **No relevance or extraction-accuracy claim is made.** The rule tables below are
  developer-authored and checked by developer-written tests. There is no Golden Dataset and no
  human-labelled extraction set yet (Milestones 9 and 10). No benchmark or quality metric was
  run for this milestone.

## 1. Status and boundary

* M6 adds a pure, typed, deterministic parser (`src/ecommerce_search/query_understanding/`), the
  `DecisionProvider` interface and its only implementation, `DeterministicDecisionProvider`
  (`src/ecommerce_search/decision/`).
* **Parsing is informational.** `GET`/`POST /search/hybrid` report the parse in an additive
  `query_understanding` block. Nothing in it reaches retrieval: the lexical and dense queries,
  the SQL, the source lists, RRF, `rrf_k`, the fused ranks and scores are exactly the M5
  behaviour (`docs/search-hybrid-m5.md`). `applied_filters` is still always `[]`.
* `/search` (V0) and `/search/dense` are unchanged and return no `query_understanding` field.
* No database, migration, model, network, dependency or setting was added or changed. The
  migration head is still `0004`.
* Milestone 7 decides how (and whether) parsed constraints become filters. Until then a parsed
  value is a report, not a constraint.
* The only provider is local and deterministic. Jev or any other remote provider, confidence
  values, the confidence gate and fallback semantics are Milestone 11 and later (ADR-003,
  ADR-005); M6 adds no placeholder for them.

## 2. Request flow (`/search/hybrid`, shared GET/POST path)

Code: `api/hybrid.py` `_hybrid_search`.

1. Start the total timer.
2. `normalize_query(raw, SEARCH_MAX_QUERY_LENGTH)`: the **only** normalization call. Its `422`
   behaviour is unchanged.
3. Validate `top_k` (`422`, unchanged).
4. `understand_normalized_query(query, provider)`: the parser runs once and the provider runs
   once, then the result is validated (it must be a `DecisionResult` whose
   `understanding.raw_query` equals the normalized query). The time of this step (parser,
   provider and validation) is `latency_ms.query_understanding_ms`.
5. `has_searchable_text(query)`: the no-searchable-text check runs **after** step 4, so a
   punctuation-only query such as `!!!` still receives the informational block and a non-null
   `query_understanding_ms` (and, as in M5, no results and no model or database work).
6. Encode the query, `read_sources`, `fuse_rrf` and build the results exactly as in M5. Every
   retrieval call receives only the normalized query string from step 2; the decision is used
   only to build the response block.

**Failure boundary.** Any exception from step 4 (a parser error, a provider error or an invalid
provider result) returns the existing fixed `503 {"detail": "search unavailable"}` and logs one
fixed line, `hybrid search unavailable: query_understanding_failed`, with no query text,
exception message, exception type or traceback. On this path no model is loaded, nothing is
encoded, no statement runs and no database connection is checked out (the session dependency is
constructed but never used). There is **no hidden fallback**: the endpoint never returns M5
results without the block. The deterministic provider is itself the fallback described in
ADR-003, so in M6 there is nothing further to fall back to.

The provider is created once per app (`create_app` sets `app.state.decision_provider`) and
injected by `api/dependencies.get_decision_provider`; tests override it. No setting selects it.

## 3. Typed contract

Code: `query_understanding/models.py` and `decision/provider.py`.

**Library functions**

* `parse_normalized_query(normalized_query: str) -> QueryUnderstanding`. It accepts the output
  of `normalize_query` and never normalizes or enforces the API length limit itself (direct
  library callers bound their own input). Precondition: the input is non-empty and equal to
  `" ".join(normalized_query.split())`; otherwise it raises `ValueError` with a fixed message
  that does not echo the input.
* `DecisionProvider` (a `runtime_checkable` Protocol):
  `understand(query: str, deterministic_result: QueryUnderstanding) -> DecisionResult`.
* `DeterministicDecisionProvider.understand` returns
  `DecisionResult(understanding=deterministic_result)` unchanged.
* `understand_normalized_query(normalized_query, provider) -> DecisionResult` parses once, calls
  the provider once and validates the result; an invalid result raises
  `QueryUnderstandingFailed` (fixed message `query understanding failed`, no query text).

**Models.** `SourceSpan`, `MatchedTerm`, `Ambiguity`, `Conflict`, `QueryAttributes`,
`QueryUnderstanding` and `DecisionResult` are Pydantic models with `frozen=True` and
`extra="forbid"`; every collection is a tuple. The API wrapper `QueryUnderstandingBlock` forbids
extra fields (it is not frozen).

| Model | Fields |
|---|---|
| `SourceSpan` | `start`, `end`, `text`; validated `0 <= start < end` |
| `MatchedTerm` | `rule: MatchRule`, `field: UnderstandingField`, `value: str` (canonical), `span` |
| `Ambiguity` | `reason: AmbiguityReason`, `span`, `value: str \| None` |
| `Conflict` | `field: UnderstandingField`, `reason: ConflictReason`, `candidates: tuple[MatchedTerm, ...]` (at least one) |
| `QueryAttributes` | `storage_interface: "NVME" \| None`, `price_preference: "low" \| None` |
| `QueryUnderstanding` | `raw_query` (non-empty), `category`, `brand`, `ram_gb` (1–4096), `storage_gb` (1–65536), `storage_type`, `min_price`, `max_price`, `semantic_intent`, `retrieval_strategy` (always `null`), `attributes`, `parser_version` (`"qu-1"`), `lexicon_version` (`"1"`), `matched_terms`, `ambiguities`, `conflicts`, `unresolved: tuple[SourceSpan, ...]` |
| `DecisionResult` | `provider` (`"deterministic"`), `provider_version` (`"qu-1"`), `understanding` |
| `QueryUnderstandingBlock` (API) | `usage` (`"informational"`), `provider`, `provider_version`, `understanding` |

`category` is the catalog `Category` (`laptop`, `phone`, `shoes`, `headphones`), `brand` the
canonical `BRAND_LOOKUP` spelling, `storage_type` the catalog `StorageType` (`SSD`, `HDD`).
`min_price`/`max_price` are `Decimal` with `0 < v <= 9999999999.99`, at most 12 digits and 2
decimal places, no NaN or infinity, stored quantized to `0.01`; JSON carries them as strings
(`"40000.00"`).

**Closed enumerations**

| Enum | Values |
|---|---|
| `UnderstandingField` | `category`, `brand`, `ram_gb`, `storage_gb`, `storage_type`, `storage_interface`, `min_price`, `max_price`, `semantic_intent`, `price_preference` |
| `MatchRule` | `category_term`, `brand_term`, `intent_term`, `storage_type_term`, `storage_interface_term`, `storage_type_implied_by_interface`, `price_preference_term`, `ram_capacity`, `ram_capacity_paired`, `storage_capacity`, `storage_capacity_implicit_gb`, `price_upper_bound`, `price_lower_bound` |
| `AmbiguityReason` | `bare_capacity`, `ambiguous_memory_capacity`, `capacity_multiple_roles`, `price_without_bound`, `unsupported_grouped_number`, `unsupported_range`, `unsupported_numeric_compound`, `out_of_range_value` |
| `ConflictReason` | `repeated_different_values`, `min_price_exceeds_max_price`, `dependent_on_conflicted_storage_type` |
| `Intent` | `gaming`, `coding`, `student`, `lightweight`, `premium`, `comfort`, `travel`, `office` |

**Spans.** `start` and `end` are zero-based Unicode **code-point** offsets into `raw_query`
(the normalized query) with an exclusive end, and `span.text == raw_query[start:end]` is
enforced by a model validator. A multi-token construction has one covering span, gaps included.
`unresolved` holds one span per token that no construction owns, in source order.

**Ordering.** Matched terms and conflict candidates are sorted by
`(span.start, span.end, field, rule)`, ambiguities by `(span.start, span.end, reason)`,
conflicts by `(field, reason)`; `unresolved` is in source order. The same input and versions
give byte-identical `model_dump_json()`.

**Canonical evidence strings.** Capacities are the base-10 GB integer (`"256"`, `"1024"`);
prices are `format(v.quantize(Decimal("0.01")), "f")` (`"40000.00"`); enumerations are the enum
value (`"laptop"`, `"HP"`, `"SSD"`, `"NVME"`, `"low"`, `"coding"`). Never an exponent, sign,
NaN, infinity or locale grouping.

**Evidence rules.** Every non-null field has a matched term with the same canonical value. A
conflicted field is null and its candidates appear only in `Conflict.candidates`. Every token is
covered by evidence (a matched term, an ambiguity or a conflict candidate) or by exactly one
`unresolved` span, never both.

**Versioning.** `PARSER_VERSION` (`parser.py`) and `LEXICON_VERSION` (`lexicon.py`) are
literal-typed in the models. Any later change to a rule, a lexicon entry, a role or a gap set
must bump the relevant version.

## 4. Rules (`qu-1`, lexicon `1`)

### 4.1 Tokens

The tokenizer is one hand-written left-to-right pass (no regular expressions). Folding is 1:1
only: ASCII and fullwidth digits and Latin letters (lowercased), `.`/`．`, and `₹` (always its
own token). Any other letter, mark or number character (Devanagari, accented letters, `ß`, `½`,
`²` and so on), ZWNJ and ZWJ is kept as OTHER. Everything else is a separator. A run of
non-separators is a NUMBER (digits, at most one internal dot), a WORD (letters) or a NUMBER
glued to a WORD (`8gb`, `40k`); every other run (`i5`, `rtx3050`, `4k60hz`, `1.2.3`, any run
containing OTHER) is one UNRESOLVED token. A NUMBER must match `[0-9]{1,10}(\.[0-9]{1,2})?`
before any conversion; otherwise its whole run is UNRESOLVED and is never converted.

Adjacency is checked per construction against exact allowed gaps; separators are never ignored
globally:

| Gap set | Allowed gaps | Used by |
|---|---|---|
| `GLUE` | `""`, `" "` | NUMBER + unit (`gb`, `tb`, `k`); `₹` + NUMBER |
| `SPACE` | `" "` | phrases, bound markers, connectives, bare integer + storage term |
| `LINK` | `" "`, `"-"`, `","`, `", "` | capacity ↔ capacity keyword, keyword ↔ keyword (`8gb-ram`) |
| `CUR` | `" "`, `"."`, `". "` | `rs`/`inr` + NUMBER (`rs.40000`) |
| exact key | the gap inside the dictionary key | brand aliases (`hewlett-packard`, `one plus`) |

A WORD glued to a NUMBER (`8ssd`, `16gaming`, `₹40kg`) is never a lexicon atom: it is usable
only as the unit of a capacity (`gb`, `tb`) or price (`k`), and a run whose word is not such a
unit binds nothing.

### 4.2 Stages

Each stage is one left-to-right pass, in this order:

| Stage | Accepted forms | Result |
|---|---|---|
| U1 unsupported numeric complex | Two or more numeric expressions `[₹ \| rs \| inr] NUMBER [glued unit]` joined by a comma after a bare number (grouped), a dash or `~` (range), `to` or `between … and` (range), `/` or any other non-space separator (compound), or a plain space before a bare 3-digit number (compound); an immediately preceding bound marker or `between`/`from` is absorbed | One ambiguity `unsupported_grouped_number`, `unsupported_range` or `unsupported_numeric_compound` over the whole complex; no value is set from any part of it |
| S1 phrases and atoms | Categories `laptop(s)`, `phone(s)`, `smartphone(s)`, `mobile(s)`, `shoe(s)`, `headphone(s)`, `earphone(s)`, `earbuds`; brands from `BRAND_LOOKUP` (whole token or exact alias); intents (bare, `X ke liye`, `for X`); `ssd`, `hdd`; `nvme`; `sasta` | `category_term`, `brand_term`, `intent_term`, `storage_type_term`; `nvme` gives `storage_interface_term` (NVME) plus `storage_type_implied_by_interface` (SSD); `sasta` gives `price_preference_term` (`low`) |
| S2 capacity value | Integer NUMBER + `gb`/`tb` (`GLUE`) | GB value (`tb` × 1024) |
| S3 capacity roles | Keyword chains after the capacity (`ram`, `memory`, `storage`, `ssd`, `hdd`, `nvme`) and before it (`ram`, `memory`, `storage`), joined by `LINK` | Only RAM → `ram_capacity`; only STORAGE (`storage`/`ssd`/`hdd`/`nvme`) → `storage_capacity`; only `memory` → `ambiguous_memory_capacity`; two or more roles → `capacity_multiple_roles` (no capacity set) |
| S3b implicit GB | Bare integer + `SPACE` + a chain of storage keywords only | `storage_capacity_implicit_gb` |
| S4 pairing (R6) | Exactly one unclaimed capacity, one unconflicted storage value, no RAM candidate, no memory or multiple-roles ambiguity, and the leftover smaller than the storage value | `ram_capacity_paired` |
| S4b leftover | Any capacity still unclaimed | `bare_capacity` |
| S5 prices | `P` = NUMBER + `k` (`GLUE`, × 1000), `₹` + NUMBER[`k`] (`GLUE`), `rs`/`inr` + NUMBER[`k`] (`CUR`). Upper: `under`, `below`, `upto`, `up to`, `within` before `P`, or `ke andar`, `ke under` after `P`. Lower: `above`, `over`, `from` before `P` | `price_upper_bound` → `max_price`; `price_lower_bound` → `min_price` (both inclusive); `P` without a marker → `price_without_bound`; `from P ke andar` binds both bounds to the same value |
| S6 connectives | `wala` or `ka` (`SPACE`) right after a token owned by a construction | Extends every evidence span ending at that token; otherwise unresolved |
| S7 resolution | All candidates for a field | Equal values → field set; different values → field null, conflict `repeated_different_values`; `min_price > max_price` → both null, conflict `min_price_exceeds_max_price`; a conflicted `storage_type` with an NVME interface → interface null, conflict `dependent_on_conflicted_storage_type` |

Numeric bounds are the catalog bounds (`docs/data-quality.md` §3.1): price
`0 < v <= 9999999999.99` with 2 decimals, RAM 1–4096 GB, storage 1–65536 GB. A value that
parses but falls outside its bound gives `out_of_range_value` (with the canonical value) and
never a partial binding; its marker or keyword stays unresolved. Numbers are converted only with
`int` and `Decimal` on validated ASCII text.

### 4.3 Decisions recorded for spec §14

* **Q5 (price-bound inclusivity):** bounds are inclusive. `under 50k` reports
  `max_price = 50000.00` meaning `price <= 50000`; `above 30k` reports `min_price = 30000.00`
  meaning `price >= 30000`. These are parsing semantics only; M7 decides filtering.
* **Q6 (implicit storage units):** a bare integer followed by a chain of storage keywords only
  (`256 ssd`) is read as GB (`storage_capacity_implicit_gb`). This is a convention, not an
  inferred fact. A chain containing `ram` or `memory` leaves the number unresolved.
* **Q9 (storage taxonomy, parsing side):** `storage_type` (`SSD`/`HDD`) and
  `attributes.storage_interface` (`NVME`) are separate fields, matching the M2 schema. `nvme`
  sets the interface and implies `storage_type = SSD`; `ssd` sets only `storage_type = SSD` and
  never an interface, so the parse of an `ssd` query names nothing that would exclude NVMe SSDs.
  How a parsed value matches products (filtering) remains M7.

### 4.4 Examples

Generated directly from the committed parser (`parse_normalized_query` on the normalized query)
at `f0c4d18`; spans are `start:end`.

| Query | Fields set | Evidence |
|---|---|---|
| `hp laptop 8gb ram 256gb ssd under 40k` | brand `HP`, category `laptop`, `ram_gb` 8, `storage_gb` 256, `storage_type` `SSD`, `max_price` `40000.00` | `brand_term` 0:2 `hp`; `category_term` 3:9 `laptop`; `ram_capacity` 10:17 `8gb ram`; `storage_capacity` 18:27 `256gb ssd`; `storage_type_term` 24:27 `ssd`; `price_upper_bound` 28:37 `under 40k` |
| `coding ke liye laptop` | `semantic_intent` `coding`, category `laptop` | `intent_term` 0:14 `coding ke liye`; `category_term` 15:21 `laptop` |
| `sasta phone` | `attributes.price_preference` `low`, category `phone` | `price_preference_term` 0:5 `sasta`; `category_term` 6:11 `phone`; no price bound |
| `ram 8gb ssd` | `storage_type` `SSD` only | ambiguity `capacity_multiple_roles` (value `8`) 0:11 `ram 8gb ssd`; `storage_type_term` 8:11 `ssd` |
| `₹40,000 laptop under 50k` | category `laptop`, `max_price` `50000.00` | ambiguity `unsupported_grouped_number` 0:7 `₹40,000`; `category_term` 8:14; `price_upper_bound` 15:24 `under 50k` |
| `from 20k ke andar` | `min_price` `20000.00`, `max_price` `20000.00` | `price_lower_bound` 0:8 `from 20k`; `price_upper_bound` 5:17 `20k ke andar` |
| `8 256 ssd` | `storage_type` `SSD` only | ambiguity `unsupported_numeric_compound` 0:5 `8 256`; `storage_type_term` 6:9 `ssd` |
| `8ssd` | none | unresolved 0:1 `8`, 1:4 `ssd` (glued word is not an atom) |
| `!!!` | none | no tokens, no evidence |

Further behaviour, from the same generator:

| Query | Result |
|---|---|
| `laptop 8gb 256gb ssd` | `ram_capacity_paired` 7:10 `8gb` (R6), `storage_gb` 256, SSD |
| `laptop 8gb 256 ssd` | `ram_capacity_paired` 8, `storage_capacity_implicit_gb` 11:18 `256 ssd` |
| `laptop 16gb` | ambiguity `bare_capacity` (16); no RAM is set |
| `memory 8gb` | ambiguity `ambiguous_memory_capacity` (8) |
| `nvme` | `storage_type` `SSD`, `attributes.storage_interface` `NVME` |
| `nvme hdd` | both null; conflicts `storage_type`/`repeated_different_values` and `storage_interface`/`dependent_on_conflicted_storage_type` |
| `hp dell laptop` | brand null, conflict `repeated_different_values` (`HP`, `Dell`); category `laptop` |
| `above 30k under 20k` | both prices null, two `min_price_exceeds_max_price` conflicts |
| `40k phone` | ambiguity `price_without_bound` (`40000.00`); no bound |
| `50k wala phone` | `price_without_bound` over 0:8 `50k wala` (connective extends it) |
| `under 40000` | both tokens unresolved: a bare number is never a price |
| `from 20k to 40k` | one `unsupported_range` ambiguity over the whole query |
| `10000000k` | `out_of_range_value` (`10000000000.00`) |
| `4097gb ram` | `out_of_range_value` (`4097`) on `4097gb`; `ram` unresolved |
| `ＨＰ ８ＧＢ ｒａｍ` | brand `HP`, `ram_gb` 8, spans over the fullwidth source text |

## 5. Unsupported or limited forms

Each of these sets no field (or no partial value) in `qu-1`:

* `lakh`, `crore`, `hazar`: unresolved words (the preceding bare number is unresolved too).
* Comma-grouped prices (`₹40,000`, `rs 40,000`, `under 40,000`): reported as
  `unsupported_grouped_number`, never as a value.
* Ranges (`20k-40k`, `from 20k to 40k`, `between 20k and 40k`): reported as `unsupported_range`,
  never as two bounds.
* Model names (`iphone`, `i5`, `rtx3050`): unresolved; there is no `iphone` → Apple inference.
* `anc`, colour, size: not in the lexicon.
* `cheap`, `sasti`, `saste`: unresolved; only `sasta` is in the lexicon.
* `rom`: unresolved; a capacity before it is a `bare_capacity`.
* Suffix currency (`40000rs`) and `rs-40000`: no price.
* `8gb ka ram`: `ka` extends the bare capacity; `ram` stays unresolved.
* Non-ASCII digits and letters (Devanagari `१६`, Hindi text, `ß`, `½`): unresolved, never
  converted.
* Fractional capacities (`1.5tb`) and `g`/`gig`: unresolved.
* Malformed glued runs (`8ssd`, `8gbssd`, `₹40kg`) do not partially bind.
* R6 pairing and implicit GB are documented conventions, not inferred facts.
* One semantic intent: two different intents conflict (`gaming laptop for coding`).

## 6. API

`GET`/`POST /search/hybrid` responses gain, additively:

* `query_understanding`: `{"usage": "informational", "provider": "deterministic",
  "provider_version": "qu-1", "understanding": { ...QueryUnderstanding... }}`;
* `latency_ms.query_understanding_ms`: a float, always present on a `200` response.

GET and POST share one code path and return the same block for the same query. Everything else
is unchanged from M5: `search_version = "v1_hybrid"`, `query`, `tsquery`, the `fusion` block,
the hit, overlap, fused and candidate counts, results, ranks and scores, and
`applied_filters = []`. The `422` bodies are unchanged and are produced before the provider
runs. The `503` body is the same fixed `{"detail": "search unavailable"}`; its OpenAPI
description now also names query-understanding failure. `/search` and `/search/dense` are
unchanged.

## 7. Security and privacy

* The parser and provider are pure: no I/O, network, database, model, settings, clock,
  randomness or logging. An import-isolation test confirms the pure packages load no database,
  web or model stack.
* The route logs one fixed reason code on failure (section 2), never query text or exception
  details.
* Input is bounded by the existing `SEARCH_MAX_QUERY_LENGTH` (200) before parsing. The tokenizer
  is a linear scanner with no regular-expression backtracking, and phrase lookahead is at most
  three tokens; tests assert structural bounds (token and evidence counts), not wall-clock time.
* Numbers are validated before conversion; no SQL is built from the parse.
* The query text (normalized) appears in the response body (`query`, `raw_query`, span texts),
  as the query already did in M5.
* The existing uvicorn access-log behaviour (it records GET query strings) is unchanged and
  deferred to Milestone 17.

## 8. Evidence

Commits (on `main`, not yet pushed when this report was written):

* `0c924f29682dbca2e5d73ab120b881fe63fccf79` `feat(m6): add deterministic query understanding
  and provider`: the pure packages and five unit test files (389 tests collected).
* `f0c4d182bc5937fb492228f60a57a145ef13d5c4` `feat(m6): expose informational query
  understanding on hybrid search`: the API block, provider injection and the API unit and
  hybrid integration tests.

Validation of the final production tree `f0c4d18` (M6 Session D): 1564 unit tests passed
(`not integration and not real_model`), 319 integration tests passed (throwaway scratch
databases), 5 `real_model` tests passed (offline, fetched MiniLM snapshot); `ruff check`,
`ruff format --check`, `uv lock --check` and `docker compose config -q` passed. The integration
tests compare two different stub providers and an independent RRF oracle and find identical
source lists, fused IDs, ranks and scores, and count zero connection checkouts on a provider
failure.

No benchmark was run, no relevance or extraction-accuracy metric exists, and no Golden Dataset
or human label was created. The M5 benchmark artifacts are unchanged.

## 9. Limitations

* Coverage is the closed v1 lexicon; everything else stays unresolved or ambiguous by design.
* The conservative U1 and capacity-role rules leave some readable capacities unbound
  (`8 256 ssd`, `ram 8gb ssd`).
* Hinglish and other phrasing are parsed for the report only; retrieval still treats them as
  plain text exactly as in M5, so `coding ke liye laptop` retrieves as before.
* Parsed values are not filters, boosts or ranking signals until a later approved milestone
  (M7 for filtering).
