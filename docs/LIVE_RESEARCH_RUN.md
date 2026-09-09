# Local research run — 2026-09-09

The owner authorized **$2 total**, not a daily allowance, for ingestion and
evaluation. The application stops at **$1.80 in conservative reservations**.
This exception applies to the local research experiment; the hosted pilot remains
free by default. No paid generation is enabled yet.

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

## Validation

82 backend tests passed against disposable PostgreSQL/pgvector, with real Google
access disabled. Eight Node protocol/review tests passed. Tests cover concurrent
spend admission, persistent exhaustion, unpriced/oversized request rejection,
paid generation blocking, pacing, cancellation accounting, secret redaction,
and review isolation. Live ingestion results will be recorded separately.
