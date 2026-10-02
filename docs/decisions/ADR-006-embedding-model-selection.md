# ADR-006: Embedding model selection (Milestone 4)

## Status

Accepted (provisional). Selected by a predeclared rule and an explicit user sign-off on
2026-10-01. **Provisional until evaluation on the human-reviewed Golden Dataset (Milestone 10).**
Refines ADR-004, which fixed the strategy and deferred the model choice to Milestone 4.

## Date

2026-10-01

## Decision

| Setting | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Immutable revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Dimension | 384 (planned column type `VECTOR(384)`) |
| Normalized embeddings | true |
| Query prefix / document prefix | none / none |
| Maximum sequence length | 256 tokens |
| Distance | cosine |
| Index | exact scan initially; no persistent ANN index |
| Embedding text | `EMBEDDING_TEXT_VERSION = "1"` (`src/ecommerce_search/embeddings/text.py`) |
| Configuration hash | `6d5cfc238e11c60053e8227b5206d295ef4f06b3fcfbd8d359c2bbb96efb743b` (`EmbeddingModelSpec.config_sha256()`) |

The specification is pinned in code as `ALL_MINILM_L6_V2` in
`src/ecommerce_search/embeddings/spec.py` (production registry). Changing any value requires a
new ADR, regenerated embeddings and re-evaluation.

## Candidates

All three come from their official Hugging Face repositories, at revisions resolved through the
official API on 2026-10-01. They were downloaded only into the git-ignored repository `models/`
directory with `python -m ecommerce_search.search model-fetch`, with a SHA-256 file manifest per
snapshot.

**Side effect, since fixed.** During those downloads, `hf-xet` also wrote three log files
(188 KB, logs only, no model data) under the machine's `HF_HOME`. The fetch command has since
been changed to confine every Hugging Face cache, token and log location to the models
directory, and to refuse to run if the libraries were already imported. Those logs were deleted
with the user's permission. The pre-existing user cache under the home directory was never read
or modified.

| Model | Revision | Model card | Licence declared in the pinned model card |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md | Apache-2.0 |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md | MIT |
| `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` | https://huggingface.co/intfloat/e5-small-v2/blob/ffb93f3bd4047442299a41ebb6fa998a38507c52/README.md | MIT |

**Licence evidence and its limits.** The licence of each model was read from the `license:`
field of the YAML front matter of the model card (`README.md`) downloaded at the pinned
revision. The README body agrees for bge (it names the MIT License). **None of the three
repositories contains a separate LICENSE file at these revisions.** This record therefore
confirms the model-card declarations; it is not an independent legal certification. Reference
licence texts: Apache-2.0 https://www.apache.org/licenses/LICENSE-2.0, MIT
https://opensource.org/license/mit.

Facts read from the snapshots (not assumed):

| Model | Dimension (pooling) | max_seq_length | Pooling | Normalize module | Query / document prefix (model card) | Snapshot bytes |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 256 | mean | yes | none / none | 91,578,415 |
| bge-small-en-v1.5 | 384 | 512 | CLS | yes | `Represent this sentence for searching relevant passages: ` / none | 134,505,940 |
| e5-small-v2 | 384 | 512 | mean | yes | `query: ` / `passage: ` | 134,478,697 |

`model.safetensors` SHA-256: MiniLM
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`; bge
`3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`; e5
`45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1`. No snapshot needs remote code:
there is no `auto_map` and no Python file. The download allow-list excludes `train_script.py`,
ONNX/OpenVINO/TF/`pytorch_model.bin` variants.

## Experiment

- **Experiment ID:** `embed-select-20261001T112530Z-a550be37`, made with
  `scripts/embedding_model_selection.py`. Artifacts are under `data/processed/embedding_selection/`
  (git-ignored, not committed): summary JSON, Markdown, and raw `samples.jsonl` with 21,609
  samples. `--verify` reported `OK`: the samples file hash matched and every summary recomputed
  exactly.
- **Source fingerprint:** base commit `2c41a36fe696b6e7fcb913a5aa58cf1792b8a77c`, **working tree
  dirty**, so the Milestone 4 code was uncommitted. Source tree SHA-256
  `a9c2241c939d7deec99cca9b99cd65418541f4ffe868df5ec809b0352af58ead`. A confirmation rerun from
  the committed tree is required before migration 0004 (see "Follow-up").
- **Protocol:**
  - 3 runs × 3 models, each in a fresh subprocess with outbound sockets disabled and no
    database;
  - the production provider (`SentenceTransformerEmbedder`) and embedding-text builder;
  - corpus: the committed 240-product synthetic seed, encoded 3× per run at batch size 32;
  - 12 queries, each with 1 cold encoding, 20 untimed warm-ups and 200 timed encodings:
    7,200 timed samples per model;
  - nearest-rank percentiles.
- **Environment:** Windows 11 (10.0.26300), Intel i5-1135G7 (4 cores, 8 threads, CPU only),
  15.7 GB RAM, Python 3.12.3, torch 2.14.1+cpu (4 threads), sentence-transformers 6.1.0,
  transformers 5.18.0.

### Hard gates (predeclared; all must pass)

| Gate | MiniLM | bge | e5 |
|---|---|---|---|
| Permissive licence declared in pinned model card | pass (Apache-2.0) | pass (MIT) | pass (MIT) |
| Immutable 40-hex revision, manifest intact | pass | pass | pass |
| Offline load from local snapshot, `trust_remote_code=False`, sockets blocked | pass | pass | pass |
| Zero truncated corpus documents (max tokens / limit) | pass (137 / 256) | pass (137 / 512) | pass (139 / 512) |
| Valid vectors (finite, 384-d, unit norm; enforced by the provider) | pass | pass | pass |
| Re-encode agreement, batch 1 vs 32 (min cosine ≥ 0.9999) | pass (1.0*) | pass (1.0*) | pass (1.0*) |

\* Rounded to 8 decimals.

### Measurements (per run; CPU)

| Model | Run | Warm query p50 / p95 / p99 / max (ms) | Cold first query (ms) | Load (ms) | Import torch+ST (ms) | Corpus encode, 240 docs (ms, 3 repeats) | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| MiniLM | 1 | 8.376 / 10.776 / 12.518 / 36.669 | 7.170 | 381.0 | 7246 | 3180 / 3331 / 3372 | 655.2 |
| MiniLM | 2 | 8.852 / 11.094 / 12.879 / 17.052 | 7.655 | 332.9 | 7790 | 3668 / 3700 / 3568 | 656.1 |
| MiniLM | 3 | 8.626 / 11.085 / 13.173 / 40.231 | 8.281 | 378.1 | 7694 | 4102 / 3906 / 3731 | 656.7 |
| bge | 1 | 20.347 / 30.504 / 40.725 / 93.027 | 18.683 | 411.9 | 7713 | 7228 / 7354 / 7601 | 698.5 |
| bge | 2 | 18.754 / 22.470 / 24.745 / 34.334 | 17.918 | 420.4 | 7612 | 7606 / 7349 / 7630 | 699.3 |
| bge | 3 | 18.807 / 23.641 / 30.280 / 90.865 | 16.995 | 411.6 | 7749 | 7354 / 7175 / 7547 | 698.5 |
| e5 | 1 | 16.686 / 20.506 / 24.213 / 392.524 | 17.209 | 376.2 | 7553 | 7202 / 7114 / 7358 | 702.1 |
| e5 | 2 | 16.675 / 20.489 / 23.241 / 40.473 | 18.111 | 387.0 | 7634 | 7154 / 7213 / 7390 | 700.6 |
| e5 | 3 | 16.279 / 19.822 / 22.026 / 35.526 | 16.469 | 402.2 | 8021 | 7514 / 7532 / 7911 | 702.9 |

**Throughput.** Corpus throughput was 58.5–75.5 documents/s for MiniLM and 30.3–33.7 for bge and
e5.

**Outlier.** The single 392.5 ms e5 sample is kept in all figures.

**Interpretation limits.**
- These figures cover query encoding only, measured in-process. They are not end-to-end API
  latency, which the dense benchmark measures later.
- The approximately 7–8 s torch/sentence-transformers import is a real cold-start cost, borne
  by the first dense request in a process.

### Predeclared rule outcome

All three models passed every gate. The rule ranks passing models by:
1. pooled warm query-encode p50 (median of per-run p50);
2. then peak RSS;
3. then snapshot size.

A difference not larger than the run-to-run spread is a tie and falls through to the next
criterion.

- MiniLM 8.626 ms, e5 16.675 ms, bge 18.807 ms.
- The gap between MiniLM and the runner-up (e5) is 8.049 ms, larger than the spread of 0.476 ms.
- **MiniLM wins on the first criterion.**

This was independently recomputed from the raw samples, without the experiment's own summary
code, with the same result. MiniLM is also the smallest candidate and had the lowest measured
peak memory.

## Qualitative probes (not a relevance evaluation)

There is no Golden Dataset yet, so no relevance or quality metric was computed. The catalog is
synthetic and templated. The table counts how many of each model's top-10 results (exact cosine
over the corpus) have a catalog attribute. These are attribute checks, not human relevance
labels.

| Probe | MiniLM | bge | e5 | Products in catalog with the attribute |
|---|---|---|---|---|
| `noise cancelling headphones`: ANC true | 10 | 10 | 10 | 20 |
| `running shoes`: subcategory running | 10 | 10 | 10 | 15 |
| `iphone`: Apple phone | 8 | 10 | 10 | 10 |
| `8gb laptop`: laptop with 8 GB RAM | 8 | 10 | 9 | 34 |
| `coding laptop`: "coding" in title/description | 4 | 4 | 6 | 27 |
| `premium phone`: phone with "premium" in description | 2 | 1 | 2 | 12 |
| `lightweight laptop`: "lightweight" in title/description | 0 | 0 | 0 | 5 |

### Known limitations

- **"lightweight" is weak for every model.** None ranked any of the 5 products whose text says
  "lightweight" in its top 10. The mean weight of the top 10 was 1.87 kg for MiniLM, 1.70 kg for
  bge and 1.80 kg for e5, against a laptop range of 1.20–2.30 kg.
- **"student"** never occurs in the catalog text, so that probe cannot be supported by the data.
- **Exact-attribute probes.** MiniLM placed fewer exact-attribute matches in its top 10 on
  `iphone` and `8gb laptop` than bge (8 versus 10). Cross-model overlap@10 for `8gb laptop` was
  0.1–0.3. This does not override the predeclared rule: exact attributes are handled by lexical
  retrieval (V0, and hybrid in Milestone 5) and later by structured filtering (Milestone 7).
- **No similarity threshold.** A nonsense query (`zzqxv`) still returns the 10 nearest
  products, for every model.
- **English only.** The models are English; Hinglish (`coding ke liye laptop`) is not supported
  by design. Query understanding is Milestone 6.

## User sign-off (2026-10-01, recorded from the user's approval message)

The user approved `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with dimension 384, normalized embeddings, no query
or document prefix, maximum sequence length 256, future column type `VECTOR(384)`, cosine
distance, and an exact scan with no persistent ANN index.

Stated reason: it wins the predeclared selection rule by a clear query-encoding latency margin,
is the smallest candidate, uses the least measured memory, and all 240 documents fit without
truncation. The qualitative probes are not a relevance evaluation. BGE's better results on some
exact-attribute probes do not override the rule; exact attributes will be handled by lexical
retrieval and later structured filtering.

Licence recording approved by the user: Apache-2.0 (MiniLM), MIT (bge) and MIT (e5), as
declared in the pinned model cards, with no separate LICENSE file at the pinned revisions.

## Consequences

- **Positive:** fastest measured query and corpus encoding of the three, smallest download,
  lowest memory, and a permissive licence.
- **Negative:**
  - 256-token input limit. The current corpus maximum is 137 tokens; `embed` refuses any text
    that would be truncated.
  - Weaker on some exact-attribute probes in the qualitative check.
- **Neutral:** all candidates are 384-dimensional, so a later switch to bge or e5 would keep
  `VECTOR(384)`, but still requires a new ADR and re-embedding.

## Confirmation run from the committed tree

The selection experiment above ran from an uncommitted working tree. It was rerun once,
unchanged, from the reviewed Phase A commit, and **independently reconfirmed MiniLM**.

- **Experiment ID:** `embed-select-20261001T115704Z-d4c414ad`.
- **Source:** base commit `2ddfedc9786c8368443af343ca8aa76795fba330`, **working tree clean**,
  no untracked source files. Source tree SHA-256
  `231f8bbc3eaf61fbaa5389bbf3638372dcc8ab9912ea65f7fcec3e4b15c23e34`.
- **Same protocol:** 21,609 raw samples. `--verify` reported `OK`.
- **Gates:** all three models passed every gate again, with zero truncation and re-encode cosine
  1.0 (rounded to 8 decimals).
- **Rule outcome:** recomputed independently from the raw samples. Per-run warm query-encode p50
  for MiniLM was 8.088 / 8.317 / 10.663 ms (median 8.317). For e5, the runner-up, it was 16.038 /
  18.684 / 16.176 ms (median 16.176), and for bge 17.736 / 18.084 / 21.006 ms (median 18.084).
  The gap of 7.859 ms exceeds the larger run-to-run spread of 2.645 ms (2.646 in the artifact,
  which rounds per-run values first), so **MiniLM wins on the
  first criterion again**.
- **Probes:** the top-10 product IDs for all 12 probe queries were identical to the first run
  for every model.
- **Run 3:** the third run was slower for every model (for example MiniLM corpus throughput of
  33–47 documents/s against about 70 in runs 1–2). The cause was not investigated. It widened
  the spreads but did not change the outcome.

These figures are taken from the recorded artifact. They do not replace the first experiment's
measurements above, which remain the selection evidence.

## Follow-up

1. Done: the selection experiment was rerun from the committed tree and confirmed MiniLM (see
   above).
2. Revisit at Milestone 10, when Golden Dataset evaluation is available, or if the catalog
   outgrows the 256-token limit.

## References

- ADR-004 (local embedding strategy), `docs/spec.md` §9.1 (provisional evaluation before
  Milestone 9).
- `scripts/embedding_model_selection.py`, `src/ecommerce_search/embeddings/`.
