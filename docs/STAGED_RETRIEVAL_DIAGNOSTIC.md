# Staged retrieval diagnostic — September 9, 2026

Twelve development-only questions ran against the validated, inactive seed corpus:
three direct, three exact-name/number, three layout/OCR, and three unanswerable
candidates. No held-out cases, planning calls, answer generation, or activation
were used. The public API still requires an active corpus; the generation override
is an internal administrator evaluation argument only.

## Observations, not release scores

| Configuration | Questions with any candidate-span match (9 candidates) | Median local retrieval seconds | Maximum seconds |
|---|---:|---:|---:|
| Keyword | 6 | 0.150 | 0.548 |
| Dense | 9 | 0.403 | 5.911 |
| Hybrid | 7 | 0.135 | 2.545 |
| Hybrid plus ONNX reranker | 9 | 1.836 | 3.162 |

All 48 retrieval executions completed without reranker fallback. These are **not
gold recall scores**: candidates remain unreviewed and matching one candidate span
does not establish adequate answer evidence. The three unanswerable questions
also returned search results; retrieval alone cannot establish answerability.
Refusal behavior was not tested by this retrieval-only run.

Query embeddings were shared across configurations. Dense timings include the
initial network embedding request; later configurations use its cache. Execution
order is fixed. These timings are diagnostic observations, not a fair standalone
latency benchmark or hosted/answer-completion p95.

## Review priorities

- `direct-01`: all configurations matched only one of three candidate spans.
  Its candidates include `HOUSING`, a section title, and a general statement that
  the housing commitment increased. A human must identify the actual supporting
  expansion details and decide which spans belong in gold evidence. Counting the
  two headings as missed factual evidence could misrepresent retrieval quality.
- `direct-03`, `layout-02`, and `layout-03`: keyword search did not return any
  nominated candidate spans; dense search did. Inspect lexical wording and actual
  passages before tuning the keyword search.
- `layout-01` and `layout-03`: hybrid fusion lost candidate coverage present in
  dense results; ONNX reranking recovered it. Inspect selected children and parent
  expansion before changing fusion/diversity settings.
- Review the unanswerable cases for topical but insufficient passages before
  live answer tests. Do not equate a nonempty result list with supporting evidence.

Raw span IDs, case hashes, index fingerprint, reranker revision, and code commit
are recorded in `evaluations/v2/results/staged-retrieval-diagnostic.json`.

## Reproduce and review

```sh
PYTHONPATH=backend .venv/bin/python evaluations/v2/staged_diagnostic.py \
  210d8f96-281a-4b3a-959b-675e2262ff1a
PYTHONPATH=backend .venv/bin/python evaluations/v2/review_batches.py
```

The diagnostic validates the dataset hash/configuration, refuses unvalidated or
incompatible generations, and saves query vectors locally for reuse. Preserve the
same database spending ledger and local query cache. Never use the backup's test
database to obtain a fresh spending allowance.

Open the generated development review index under `.local/evaluation-v2/`.
A human should inspect the original PDFs, correct answerability/support and export
feedback with their name. Exported feedback does not automatically change gold
labels. Review all required factual details, not just the nominated headings.

The 12 query embeddings added $0.024 of conservative reservations. The shared
total is now **$0.466**, leaving **$1.334** below the $1.80 cutoff. This is not an
invoice measurement. Paid generation remains disabled pending complete-workflow
spending controls. Quality acceptance and human review are still outstanding.
