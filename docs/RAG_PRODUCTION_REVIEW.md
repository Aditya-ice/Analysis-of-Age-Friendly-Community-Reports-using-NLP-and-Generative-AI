# ElderHelp production RAG review

Reviewed 2026-09-07, baseline commit 8ae26b8. This is a source-code review plus local deterministic probes, not a live Google quality evaluation or load test.

## Verdict

A real RAG prototype with credible architectural foundations, but not production-ready. Hybrid SQL retrieval, HNSW/GIN indexes, Google provider adapters, OCR integration, provenance fields, and mobile integration are implemented. Correctness, corpus lifecycle, and operational guarantees are not yet demonstrated. Prior passing build/test claims must not be interpreted as evidence of answer accuracy or scalability.

A senior technical interviewer would value the explicit retrieval implementation and reproducible builds, then ask for measured quality improvements, failure analysis, rollback behavior, and load/cost evidence. The current project cannot yet supply those results. Do not advertise guaranteed grounded answers or measured production scale.

## Deployment-blocking findings

1. **Citation existence is mistaken for factual support.** `backend/elderhelp/services/answering.py:63` only validates marker IDs. A local probe supplied evidence “The city will preserve affordable homes.” and the claim “The city guarantees every resident a free private airplane [S1].”; validation returned true. One valid marker can also accompany uncited claims. Add structured claims with source-span IDs, deterministic span validation, and semantic support checks calibrated against human labels. Verify every material claim, including numbers, dates, entities, and scope. A model judge is an additional fallible check, not a proof.

2. **Unverified text is already streamed.** `answering.py:151` emits generated deltas before checking citations. A local fake-generator probe confirmed an unsupported delta appears before the final insufficient-evidence response. First implement buffered generation → validation → emission; then consider sentence-level validated streaming if measured latency warrants it. Measure first verified answer text, not the cheap `start` event.

3. **Index reuse ignores the embedding and extraction configuration.** `services/ingestion.py:110` skips a report when PDF hash and existing chunk count match. Changing embedding model, prefixes, chunking, or extraction does not rebuild it. `IngestionRun` records model names, but chunks have no index-generation/run relationship. Introduce immutable report revisions and index generations keyed by file hash plus extraction/chunking/embedding configuration hashes; serve only a compatible active generation. Validate dimension, cardinality, finite values, and provider batch semantics.

4. **Approval revocation and metadata changes can be ignored.** The same early return precedes metadata/status updates. Removing a report from the manifest does not reconcile it out of the database. An existing local PDF is never refreshed from the publisher. Reconcile metadata independently of indexing, explicitly withdraw omitted/revoked items, and define refresh/version policy. Test revocation immediately excludes retrieval and catalog results. Decide duplicate-content aliases explicitly: the unique report SHA currently makes identical PDFs under different slugs fail.

5. **Evaluation does not measure the promised metrics.** `evaluations/run.py` tests title substrings and required words; it ignores `expected_pages`, does not inspect retrieved top-eight evidence, and does not measure citation precision, supported-claim rate, latency, or cost. The 40-case unit test checks counts/categories only. Existing categories do not explicitly ensure OCR and exact-name coverage. Build an evaluator with source-span gold labels and raw retrieval traces available only in an authorized evaluation context. Audit the existing page labels before calling them ground truth.

## Quality gaps

- **Follow-ups lose resolved meaning after retrieval:** planner queries use history, but reranking and generation use the original question without history. Return a structured plan with `standalone_question` and bounded subqueries; use the resolved question consistently while preserving original user intent.
- **Parent/child chunks are page windows, not document sections:** the heading detector checks a few lines and parent context is the page. Long parent pages are truncated from the beginning, so a retrieved child near the end can be absent from the generator prompt. Include the selected child unconditionally; construct bounded section context with offsets and page mappings.
- **Displayed citations may not contain the support:** generation uses parent content while the excerpt is the first 500 characters of child content. Cite the actual supporting span on the exact page, with explicit PDF-page versus printed-page labeling.
- **OCR quality is a character-density heuristic:** it cannot reliably detect scrambled multi-column reading order, repeated headers, or broken tables. Preserve layout blocks/tables and evaluate digital, scanned, mixed, rotated, and multi-column examples. Quarantine extraction failures instead of silently accepting unusable pages when OCR is unconfigured.
- **No evidence sufficiency calibration:** nearest neighbors are returned for unrelated questions. Reranker scores are discarded, and marker validation is the final gate. Preserve scores, calibrate answerability on held-out data, and handle partial evidence/contradictory reports explicitly.
- **Evidence diversity is coarse:** deduplication is by chunk ID, allowing overlapping chunks with repeated parent pages. With two reports the default cap permits at most six blocks. Optimize evidence coverage per question/subquery, merge overlaps, and enforce a prompt-token budget. Do not treat eight as a universally optimal number.
- **No measured ablation:** compare lexical-only, dense-only, hybrid, and hybrid+reranker with identical questions/corpus. Add rewriting and layout/chunking improvements only when they improve held-out results at an acceptable latency/cost.

## Reliability, security, and scale

- The stream holds its database session/transaction after retrieval while waiting on Google. Release database resources after materializing evidence and before generation. Explicitly budget DB connections against Cloud Run instance count and concurrency; the current settings are not a load-test result.
- Subqueries and dense/keyword queries execute sequentially. Batch query embeddings and use bounded concurrency where safe; never concurrently use the same SQLAlchemy session. Measure first, then optimize critical-path stages.
- No application-level overall deadline, stage budgets, distributed rate/cost limits, or admission control. Cloud Run's 60-second timeout does not provide these. Add bounded retries with jitter for transient failures, explicit cancellation, quota-aware concurrency, and per-client/installation and global budgets for an account-free public service. Do not distribute service-account keys to mobile clients.
- Reranker fallback silently swallows all errors. Record categorized fallback events and degraded-mode metrics, and test the quality impact.
- `/readyz` checks database/chunk existence and whether a project string is set, not model availability, vector compatibility, or approved active corpus readiness. Use cached capability checks and an explicit active-index fingerprint; avoid paid calls on every health probe.
- Request middleware duration ends around response creation, not stream completion. Log per-stage timing, first verified delta, completed/cancelled stream duration, and model/index/prompt versions. Join all Google usage to request IDs; current generation usage omits planner/embedding/ranking/OCR cost and lacks a request ID. Do not retain raw user text in ordinary telemetry.
- Documents and questions are concatenated with instructions in a single prompt. Treat retrieved text as untrusted data, use separate system instructions and structured evidence, and add indirect-injection tests. Delimiters alone do not solve prompt injection.
- Download validation lacks maximum size/page limits, pinned expected hashes, path containment, and redirect/network destination policy. This is an administrator-controlled manifest today, so it is not an exposed arbitrary-URL API, but the ingestion trust boundary still needs limits before expanding corpus administration.
- Ingestion is serial and stops on first failure. Introduce resumable per-report jobs, leases/idempotency keys, bounded batching, quarantine/dead-letter handling, and atomic promotion of a complete corpus generation. Keep the previous version available for rollback.
- CI provisions PostgreSQL and runs migrations, but current retrieval tests exercise Python fusion/diversification, not real hybrid SQL/filters/HNSW behavior. Add database integration tests and representative data before claiming scale.

## Recommended implementation order

### PR A — correctness and corpus lifecycle

Fix claim/span validation and pre-display gating, active index versioning, metadata/revocation reconciliation, and follow-up propagation. Add regression tests reproducing every failure above. Exit gate: unsupported synthetic claims are rejected before display; revoked reports disappear; model/config changes cannot reuse incompatible vectors.

### PR B — evaluation and retrieval evidence

Human-audit the 40 seed cases; expand to a proposed 150–300 cases covering names, numbers, tables, OCR, follow-ups, comparisons, conflicting dates, partial answers, injection, and unanswerable questions. Keep a held-out set separate from tuning. Store corpus/config fingerprints and per-stage metrics for reproducible runs.

Report evidence-unit Recall@8 (including all required sources for comparisons), nDCG@8, citation precision, supported-claim coverage, and false-answer/refusal rates by category. Distinguish citation validity from entailment. Compare ablations with uncertainty estimates; do not claim robustness from six refusal examples.

The original targets (Recall@8 >=90%, citation precision >=95%, supported-claim rate >=90%, correct refusal on every curated unanswerable case) are initial release gates, not achieved scores or universal standards. Require no material unsupported high-risk claims in reviewed cases. Define the acceptable error budget and user scope explicitly.

### PR C — extraction and retrieval improvements

Add layout-aware sections/tables, exact support spans, child-preserving parent context, overlap deduplication, and calibrated insufficiency. Benchmark filtered HNSW against exact search. Tune candidates, HNSW search parameters, diversity, and context budgets using evaluation results.

### PR D — operations and controlled staging

Add rate/cost budgets, deadlines, resource-pool limits, full-stream observability, retries/fallback metrics, and real PostgreSQL/provider contract tests. Load-test corpus tiers such as 10k/100k/1m chunks and concurrent streams such as 5/20/50; these are proposed test points, not supported capacity. Measure p50/p95 latency, errors, connection usage, memory, provider quotas, and cost per successful supported answer. Test ingestion concurrently with serving and restore/rollback procedures.

Use private staging for paid provider/evaluation tests before public exposure. Verify actual Google model availability and batch/prefix behavior in the selected region. Deploy no public service until quality gates and operating budgets are met.

## Architecture recommendation

Keep FastAPI, Google providers, and PostgreSQL/pgvector for the next milestone. Separate serving from asynchronous ingestion and add versioned index promotion. Add complexity only in response to measured bottlenecks. A graph database, agent loop, or dedicated vector service is not automatically an improvement for two report PDFs. Scale claims require corpus/load measurements; accuracy claims require source-labeled evaluation.

## Primary references

- pgvector documents HNSW recall/speed tradeoffs and filtered approximate-search behavior, including iterative scans: https://github.com/pgvector/pgvector#filtering
- Google evaluation service: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
- OWASP explains indirect prompt injection via external content: https://genai.owasp.org/llmrisk/llm01-prompt-injection/

These references inform recommendations. Findings above come from this repository and local probes. No paid model calls, ingestion, cloud deployment, or accuracy/load benchmark was run during this review.
