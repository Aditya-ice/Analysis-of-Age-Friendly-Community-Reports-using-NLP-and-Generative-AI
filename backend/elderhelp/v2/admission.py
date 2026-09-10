"""Reserve the complete answer workflow before dispatch; refund only unspent calls."""

import asyncio
from datetime import UTC, datetime

from elderhelp.v2.planning import needs_planning
from elderhelp.v2.quota import daily, release, reserve


class BudgetProvider:
    def __init__(self, provider, generation: int, embedding: int):
        self.provider = provider
        self.generation_left, self.embedding_left = generation, embedding

    async def structured(self, *args, **kwargs):
        if self.generation_left <= 0:
            raise RuntimeError("Generation call budget exceeded")
        prepare = getattr(self.provider, "before_dispatch", None)
        if prepare:
            await prepare("generation")
            kwargs["paced"] = True
        # Waiting for a rate slot is not a dispatched call.
        self.generation_left -= 1
        return await self.provider.structured(*args, **{**kwargs, "reserved": True})

    async def embed(self, *args, **kwargs):
        if self.embedding_left <= 0:
            raise RuntimeError("Embedding call budget exceeded")
        prepare = getattr(self.provider, "before_dispatch", None)
        if prepare:
            await prepare("embedding")
            kwargs["paced"] = True
        self.embedding_left -= 1
        return await self.provider.embed(*args, **{**kwargs, "reserved": True})


class Admission:
    def __init__(self, state, pilot, payload):
        self.state, self.pilot, self.payload = state, pilot, payload
        self.acquired, self.reserved = False, False

    async def acquire(self):
        from elderhelp.v2.quota import QuotaExceeded

        try:
            await asyncio.wait_for(self.state.answer_slots.acquire(), timeout=0.05)
            self.acquired = True
        except TimeoutError:
            raise QuotaExceeded(5) from None
        try:
            planned = needs_planning(self.payload)
            gen_count, embed_count = (3, 3) if planned else (2, 1)
            self.provider = BudgetProvider(self.state.provider, gen_count, embed_count)
            settings = self.state.settings
            generation = daily("generation", gen_count, settings.generation_daily_limit)
            embedding = daily("embedding", embed_count, settings.embedding_daily_limit)
            self.gen_key, self.embed_key = generation[0], embedding[0]
            minute = int(datetime.now(UTC).timestamp()) // 60
            expires = datetime.fromtimestamp((minute + 1) * 60, UTC)
            await reserve(
                self.state.database,
                [
                    daily("answers", 1, settings.answers_daily_limit),
                    (f"answers:minute:{minute}", 1, settings.answers_minute_limit, expires),
                    daily(f"session:{self.pilot.session_id}", 1, settings.session_daily_limit),
                    daily(f"invite:{self.pilot.invite_id}", 1, settings.invite_daily_limit),
                    generation,
                    embedding,
                ],
            )
            self.reserved = True
            return self.provider
        except BaseException:
            await self.close()
            raise

    async def close(self):
        try:
            if self.reserved:
                self.reserved = False
                await release(
                    self.state.database,
                    {
                        self.gen_key: self.provider.generation_left,
                        self.embed_key: self.provider.embedding_left,
                    },
                )
        finally:
            if self.acquired:
                self.acquired = False
                self.state.answer_slots.release()
