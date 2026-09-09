# Local research run — 2026-09-09

The owner authorized **$2 total**, not a daily allowance, for ingestion and
evaluation. The application stops at **$1.80 in conservative reservations**.
This exception applies to the local research experiment; the hosted pilot remains
free by default. No paid generation is enabled yet.

The owner confirmed a paid embedding limit of 3,000 requests/minute. Local
ingestion is deliberately paced at 60/minute with concurrency two and a
300-request daily ceiling. Generation remains disabled. The earlier five/minute
run was gracefully paused and its existing generation resumed, preserving the
embedding cache and the cumulative spending ledger.

## Spending controls

Set `ELDERHELP_PAID_INGESTION_CONFIRMED=true` only with explicit authorization.
Keep `ELDERHELP_FREE_TIER_CONFIRMED=false` for a paid project. Keys belong in the
ignored, private `.env`, never `.env.example` or Git.

Each text embedding reserves $0.002 before SDK dispatch. Requests are restricted
to `gemini-embedding-2` and at most 8192 UTF-8 bytes including prefixes. At the
[reviewed standard text price](https://ai.google.dev/gemini-api/docs/pricing)
of $0.20/million tokens, this exceeds the maximum token-based charge for the
accepted text envelope. The rate review expires December 31, 2026.

Reservations accumulate atomically in PostgreSQL under
`paid:elderhelp-research-total-microusd`. They never reset daily or refund failed,
cancelled, or uncertain dispatches. Preserve this row in every backup, restore,
and future generation implementation. Do not use a fresh database, change the
counter, or rotate keys to replenish the allowance. The ledger records an upper
reservation estimate, not Google's invoice. It cannot control unrelated use of
the Google project or charges outside this application.

Generation fails closed in paid-ingestion mode until complete-workflow monetary
reservations and their tests are implemented using this same total ledger.
The API does not automatically enable paid mode. SDK automatic retries remain
disabled. Daily counters and per-model pacing also apply.

## Review and readiness

Generate ten offline development-only review packets with:

```sh
PYTHONPATH=backend .venv/bin/python evaluations/v2/review_batches.py
PYTHONPATH=backend .venv/bin/python evaluations/v2/readiness.py --check-database
```

Packets keep shared facts/spans together, exclude the 40 held-out cases, and
export human feedback separately from dataset labels. No candidate becomes gold
automatically. There are still zero human-reviewed cases and no live quality
metrics. A staged index must not be activated merely because ingestion completed.

## Completed live ingestion

Generation `210d8f96-281a-4b3a-959b-675e2262ff1a` completed for both seed
reports on September 9, 2026. Structural corpus validation passed at
20:40:13 UTC: 221 chunks, two reports, no errors. The database contains 108
pages, 1,723 source spans, and 221 cached embeddings. The generation is validated
but not active. No answer-generation or semantic-verification calls were made.

The cumulative spend ledger holds **$0.442 in conservative reservations**, leaving
**$1.358** below the $1.80 cutoff. This is not a billed-cost measurement. Graceful
pause/resume reused saved embeddings; the final ledger contains exactly 221
embedding reservations.

A private 1,373,596-byte PostgreSQL backup was restored into a separate disposable
database. Report, page, span, chunk, embedding, and generation-link counts all
matched, as did the $0.442 ledger. Neither the backup nor database files are in
Git. A restore-test copy must never be used for paid requests: it duplicates the
ledger snapshot, not the authorized allowance.

Next gates are human evidence review, live retrieval evaluation, and tested
complete-workflow monetary controls before paid answer generation. These remain
outstanding; structural ingestion success is not an accuracy score.

## Validation

82 backend tests passed against disposable PostgreSQL/pgvector, with real Google
access disabled. Eight Node protocol/review tests passed. Tests cover concurrent
spend admission, persistent exhaustion, unpriced/oversized request rejection,
paid generation blocking, pacing, cancellation accounting, secret redaction,
and review isolation. Live ingestion results will be recorded separately.
