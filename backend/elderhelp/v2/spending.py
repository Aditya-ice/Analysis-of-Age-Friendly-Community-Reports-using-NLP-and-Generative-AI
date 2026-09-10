"""Non-resetting, conservative spend reservations for the authorized local experiment.

Never refund dispatched/uncertain work. This is a reservation ledger, not an invoice.
The fixed key must also be preserved in backups and future generation controls.
"""

from datetime import UTC, datetime

from elderhelp.v2.quota import reserve

BUDGET_KEY = "paid:elderhelp-research-total-microusd"
EMBEDDING_RESERVATION = 2_000  # $0.002; 8192 text tokens at $0.20/M cost $0.0016384.


async def reserve_embedding_spend(database, settings, text):
    # Bound text by UTF-8 bytes (a conservative token bound), including the prefix.
    # Fail closed on model/pricing changes; reviewed rate card expires after this year.
    if (
        settings.embedding_model != "gemini-embedding-2"
        or len(text.encode("utf-8")) > 8192
        or datetime.now(UTC).date() > datetime(2026, 12, 31).date()
    ):
        raise ValueError("Paid embedding request exceeds the reviewed spending envelope")
    await reserve(
        database,
        [
            (
                BUDGET_KEY,
                EMBEDDING_RESERVATION,
                settings.paid_budget_microusd,
                datetime(9999, 1, 1, tzinfo=UTC),
            )
        ],
    )
