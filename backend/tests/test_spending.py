import asyncio

import pytest
from elderhelp.config import Settings
from elderhelp.v2.google import Gemini, ProviderUnavailable
from elderhelp.v2.models import QuotaCounter
from elderhelp.v2.quota import QuotaExceeded
from elderhelp.v2.spending import BUDGET_KEY, reserve_embedding_spend


async def test_paid_reservations_are_atomic_and_persistent(research_db):
    settings = Settings(_env_file=None, paid_budget_microusd=4000)
    results = await asyncio.gather(
        *(reserve_embedding_spend(research_db, settings, "text") for _ in range(10)),
        return_exceptions=True,
    )
    assert sum(r is None for r in results) == 2
    assert sum(isinstance(r, QuotaExceeded) for r in results) == 8
    async with research_db.sessions() as db:
        row = await db.get(QuotaCounter, BUDGET_KEY)
        assert row.used == 4000 and row.expires_at.year == 9999
    with pytest.raises(QuotaExceeded):
        await reserve_embedding_spend(research_db, settings, "restart cannot reset budget")


async def test_paid_envelope_fails_closed_before_database_access():
    with pytest.raises(ValueError):
        await reserve_embedding_spend(None, Settings(_env_file=None), "a" * 8193)
    with pytest.raises(ValueError):
        await reserve_embedding_spend(
            None, Settings(_env_file=None, embedding_model="unpriced"), "text"
        )
    with pytest.raises(ValueError):
        Settings(_env_file=None, paid_budget_microusd=2_000_000)
    with pytest.raises(ValueError):
        Settings(_env_file=None, free_tier_confirmed=True, paid_ingestion_confirmed=True)


async def test_paid_generation_is_blocked_before_dispatch():
    provider = object.__new__(Gemini)
    provider.settings = Settings(_env_file=None, paid_ingestion_confirmed=True)
    with pytest.raises(ProviderUnavailable):
        await provider.structured(None, "", {})
