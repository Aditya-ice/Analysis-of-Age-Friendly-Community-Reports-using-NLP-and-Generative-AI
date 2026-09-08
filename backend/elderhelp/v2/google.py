"""Gemini Developer API only. Enabling live calls requires explicit free-tier confirmation."""

import asyncio
import json
import logging
import math
import time

from google import genai
from google.genai import types

from elderhelp.observability import request_context
from elderhelp.v2.quota import QuotaExceeded, daily, google_retry_after, pause_google, reserve


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
        if (
            not settings.google_api_key
            or not settings.google_api_key.get_secret_value()
            or not settings.free_tier_confirmed
        ):
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
            retry_after = await google_retry_after(self.database)
            if retry_after:
                raise QuotaExceeded(retry_after)
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
                logging.getLogger("elderhelp.metrics").info(
                    json.dumps(
                        {
                            "event": "embedding_call",
                            "request_id": request_context.get(),
                            "model": self.settings.embedding_model,
                            "inputs": 1,
                            "dimensions": 768,
                            "input_characters": len(text),
                        }
                    )
                )
                return normalize_vector([item.values for item in response.embeddings or []])
            except Exception as exc:
                if getattr(exc, "code", None) == 429:
                    await pause_google(self.database)
                    raise QuotaExceeded() from None
                if isinstance(exc, (ValueError, TimeoutError)):
                    raise
                raise ProviderUnavailable("Google embedding unavailable") from None

    async def structured(self, schema, system: str, payload: dict, *, reserved=False):
        retry_after = await google_retry_after(self.database)
        if retry_after:
            raise QuotaExceeded(retry_after)
        if not reserved:
            await reserve(
                self.database, [daily("generation", 1, self.settings.generation_daily_limit)]
            )
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.settings.provider_timeout_seconds):
                response = await self.client.aio.models.generate_content(
                    model=self.settings.generation_model,
                    contents=json.dumps(payload),
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0,
                        max_output_tokens=3000,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
            result = schema.model_validate_json(response.text or "")
            usage = response.usage_metadata
            logging.getLogger("elderhelp.metrics").info(
                json.dumps(
                    {
                        "event": "model_call",
                        "request_id": request_context.get(),
                        "schema": schema.__name__,
                        "model": self.settings.generation_model,
                        "seconds": round(time.monotonic() - started, 3),
                        "input_tokens": usage.prompt_token_count if usage else None,
                        "output_tokens": usage.candidates_token_count if usage else None,
                    }
                )
            )
            return result
        except Exception as exc:
            if getattr(exc, "code", None) == 429:
                await pause_google(self.database)
                raise QuotaExceeded() from None
            if isinstance(exc, (ValueError, TimeoutError)):
                raise
            raise ProviderUnavailable("Google structured response unavailable") from None

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()
