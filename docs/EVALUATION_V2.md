# Evidence evaluation and measured limitations

The evaluation harness is implemented. **The release quality gate is blocked**, with no live Google answers and no human-approved gold cases. The tracked reports in `evaluations/v2/results/quality-*.json` show missing review/output evidence and null quality rates; they are not passing scores.

## Audit and dataset

The original 40 questions remain in `evaluations/cases.jsonl` for historical reference. Their old `evaluations/run.py` checks report-title substrings and required words, ignores expected pages and retrieval order, and cannot measure citation support or claim correctness. Do not use its percentage as a release gate.

The new `evaluations/v2/cases.jsonl` has 120 candidate cases: 35 direct, 15 exact names/dates/numbers, 15 comparisons, 10 follow-ups, 10 layout/OCR, 20 unanswerable, 10 partial/conflicting-evidence probes, and five prompt-injection probes. All 80 development and 40 held-out cases belong to whole related-fact groups. Topic families and shared candidate span IDs stay in one split. Both splits contain every primary category. The deterministic split seed, extraction/chunking fingerprint and dataset hash are recorded in `dataset.json`.

Candidate passages were selected from local source extraction and **have not been certified as gold evidence**. Human reviewers must check answerability, required components, exact spans, numerical relationships, contradictions and permission to use uncertain extraction. The partial category currently emphasizes missing evidence; if reviewers find no actual contradictory statements in the two seeds, do not invent report disagreements to satisfy a label. Contradiction scenarios without corroboration in the seeds must be marked synthetic; they are not seed-corpus gold cases.

The layout audit found scanned commission rosters on NYC PDF pages 81–83 whose readable footer text previously suppressed OCR. Text-yield checks now use blocks intersecting the page body. NYC PDF pages 12–13 contain numerical infographics with uncertain reading order; they are retained in extraction provenance, excluded from evidence, and represented by explicit-limitation cases. Current extraction: NYC 92 pages/161 chunks; engagement report 16 pages/60 chunks. These counts are extraction results, not retrieval accuracy.

Only publication years are confirmed for the seed reports. Their manifest now records `publication_precision: year`; v2 returns `2017`/`2018`, without inventing January 1. The nullable v1 date is omitted when a full day is unknown. Migration `20260908_0003` gives existing metadata unknown precision until the authoritative manifest is reconciled.

## Human review and repeatable evaluation

Use the Python environment from the root README. These commands perform no model calls:

```sh
uv run python evaluations/v2/review_tools.py packet --split development
uv run python evaluations/v2/review_tools.py validate
```

Open `.local/evaluation-v2/development-review.html` and check each candidate against the original local PDF. Edit the case JSON: correct the question/labels, add any missing exact candidate spans, then add `gold_evidence_units` such as `[{"alternatives": [["SPAN_UUID_A", "SPAN_UUID_B"], ["SPAN_UUID_C"]]}]`. One alternative must be fully retrieved to recall the evidence unit. The referenced spans must be searchable and individually reviewed. Set `review.status` to `human_verified`, supply the human reviewer's name and an ISO-8601 `reviewed_at` with timezone. Do not have an AI attest to human review. Run validation and deliberately refresh the recorded dataset hash:

```sh
uv run python evaluations/v2/review_tools.py validate --refresh-metadata
```

Commit reviewed labels separately. The candidate generator refuses to overwrite reviewed labels. Hold-out passages/questions must not inform tuning; evaluate the held-out split after freezing code, models, prompts and configuration. A changed index fingerprint requires a new evaluation version and reviewed spans.

After a compatible corpus has been ingested/activated and the operator has confirmed a billing-disabled Google project and its actual quotas:

```sh
uv run python evaluations/v2/research.py live --split development --limit 3
uv run python evaluations/v2/research.py live --kind ablation --split development --limit 3
uv run python evaluations/v2/review_tools.py annotations
# A human fills the output-review JSON in .local/evaluation-v2/annotations.
uv run python evaluations/v2/research.py report --split development
```

Live runs use only reviewed curated cases, the same persisted quota counters and admission rules as the pilot, and a stable local session identity. Resume by repeating the command with the same output directory. Successful results and query embeddings are cached; quota exhaustion pauses work. Existing traces are bound to the full reviewed case hash; output annotations are bound to the trace hash. Stale or incomplete reviews block release. Separate output directories/versioned metadata are required when changing code/model/configuration; never rotate session identity to bypass limits. No model calls run in ordinary CI.

Ablations use the same resolved query plan for keyword-only, dense-only, hybrid and hybrid-plus-reranker search. A dense-only regression test prevents accidental keyword inclusion. Cached query vectors avoid spending embedding quota repeatedly. Outputs record evidence-unit hits, candidate counts, fallback use and latency for each configuration; these four-way live results are **pending**.

Output reviewers classify every displayed paragraph as factual or non-factual, judge each factual claim and each citation link, explicitly flag unsupported dates/quantities/current-service assertions, and mark covered answer components and injection success. Static limitation paragraphs remain visible to reviewers, preventing uncited narrative from disappearing from evaluation. A second Gemini verification call does not replace this review.

Release thresholds are evidence-unit Recall@8 ≥90%, citation precision ≥95%, supported factual claims ≥95%, answerable-component coverage ≥90%, all curated unanswerable cases refused, no successful injection and no unsupported critical fact. Reports include category denominators and failures. Missing outputs, absent gold labels and refusing everything cannot produce a passing gate. Curated traces may retain non-sensitive questions/evidence locally in explicit evaluation mode; ordinary API logs never do.

## Results recorded on September 8, 2026

- Backend regression suite: **64 passed**, using real PostgreSQL 16/pgvector, local Tesseract and the pinned quantized ONNX reranker, with Google provider doubles.
- Ten concurrent answer attempts: two admitted, eight rejected with HTTP 429, both slots released. This is a bounded-admission regression, not a hosted load test.
- Development keyword diagnostic: 54 of 56 standalone answerable/partial questions had no strict all-terms block match. Natural-language keyword retrieval now broadens to OR terms only when the strict search has no results; explicit quoted phrases, exclusions and OR syntax retain their semantics. Zero broad-search misses does **not** demonstrate relevant retrieval. Gold evidence is still required to measure recall.
- Existing iOS CI on the step-5 branch built the app but failed a simulator launch/background assertion. The client stage must resolve/retest it; it is not counted as a simulator pass.

### Local synthetic capacity

`evaluations/v2/results/synthetic-capacity.json` contains raw trials. The test uses random normalized 768-dimensional vectors and clearly synthetic repetitive text in a disposable database, with five query vectors per size. These data do not approximate semantic relevance or full production corpus storage. The figures below are local median milliseconds, not hosted p95:

| Synthetic chunks | Exact vector, approval filter | Exact vector, one-report filter | Keyword | Table and exact indexes |
|---:|---:|---:|---:|---:|
| 1,000 | 2.91 ms | 0.34 ms | 13.16 ms | 5.02 MiB |
| 10,000 | 25.66 ms | 1.71 ms | 109.06 ms | 49.33 MiB |
| 50,000 | 196.62 ms | 31.99 ms | 320.22 ms | 242.45 MiB |

Default filtered HNSW returned zero results for all five one-report queries at 10k; iterative scanning restored 94% mean overlap with exact top-30. At 50k the same iterative experiment achieved only 34.7%. Random high-dimensional vectors are deliberately difficult and not an estimate of real report recall. **Keep exact search**; neither benchmark authorizes enabling ANN. HNSW build storage, timings and individual counts are included in the raw file.

Reproduce only in the dedicated disposable database (the script replaces its own synthetic table):

```sh
createdb elderhelp_capacity_test
uv run python evaluations/v2/capacity.py --database-url postgresql://USER:PASS@localhost/elderhelp_capacity_test
```

The serving-only macOS smoke previously peaked at about 152 MiB. Free-host Linux memory under two complete live workflows, warm-service verified-completion p95 ≤30 seconds, cold-start latency, actual free quota use and human-reviewed answer quality are **not measured yet**. Missing quotas or human review delay release; they never trigger paid fallback or weaker verification.

## Agent-assisted evidence review (September 9, 2026)

All 80 development candidates now have separate AI findings: 41 supported, 35 requiring corrections, and four unresolved. These are evidence-review outcomes, not RAG accuracy measurements. The 40 held-out cases remain excluded; human gold and release approval are unchanged. A 12-case manual packet is pending. See [the review report](AGENT_EVIDENCE_REVIEW.md) for methods, limitations, and reproduction commands.
