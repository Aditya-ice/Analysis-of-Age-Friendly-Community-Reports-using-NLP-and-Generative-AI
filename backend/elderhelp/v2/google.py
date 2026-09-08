"""Gemini Developer API only. Enabling live calls requires explicit free-tier confirmation."""

import asyncio
import math

from google import genai
from google.genai import types

from elderhelp.v2.quota import QuotaExceeded, daily, reserve


class ProviderUnavailable(RuntimeError):
    pass


def normalize_vector(vectors) -> list[float]:
    if len(vectors) != 1:
        raise ValueError("Embedding 2 must return exactly one vector for one input")
    vector = list(vectors[0])
    if len(vector) != 768 or not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding must contain 768 finite values")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Embedding must have nonzero finite magnitude")
    return [x / norm for x in vector]


class Gemini:
    def __init__(self, settings, database):
        if not settings.google_api_key or not settings.free_tier_confirmed:
            raise ProviderUnavailable(
                "Configure a billing-disabled Google project and confirm free tier"
            )
        self.settings, self.database = settings, database
        self.client = genai.Client(
            vertexai=False,
            api_key=settings.google_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=int(settings.provider_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self.embedding_slots = asyncio.Semaphore(2)

    async def embed(self, text: str, *, reserved: bool = False) -> list[float]:
        async with self.embedding_slots:
            if not reserved:
                await reserve(
                    self.database, [daily("embedding", 1, self.settings.embedding_daily_limit)]
                )
            try:
                async with asyncio.timeout(self.settings.provider_timeout_seconds):
                    response = await self.client.aio.models.embed_content(
                        model=self.settings.embedding_model,
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=768),
                    )
                return normalize_vector([item.values for item in response.embeddings or []])
            except Exception as exc:
                if getattr(exc, "code", None) == 429:
                    raise QuotaExceeded() from None
                if isinstance(exc, (ValueError, TimeoutError)):
                    raise
                raise ProviderUnavailable("Google embedding unavailable") from None

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()
