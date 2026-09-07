from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from elderhelp.config import Settings
from elderhelp.schemas import AnswerRequest
from elderhelp.services.retrieval import RetrievedChunk
from google import genai
from google.cloud import discoveryengine_v1 as discoveryengine
from google.genai import types

logger = logging.getLogger("elderhelp.google")


class GoogleClients:
    def __init__(self, settings: Settings) -> None:
        if not settings.google_cloud_project:
            raise ValueError("ELDERHELP_GOOGLE_CLOUD_PROJECT is required")
        self.settings = settings
        self.genai = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )


class GoogleEmbeddingProvider:
    def __init__(self, clients: GoogleClients) -> None:
        self.clients = clients

    async def embed_query(self, query: str) -> list[float]:
        prepared = f"task: question answering | query: {query}"
        result = await self.clients.genai.aio.models.embed_content(
            model=self.clients.settings.embedding_model,
            contents=prepared,
            config=types.EmbedContentConfig(
                output_dimensionality=self.clients.settings.embedding_dimensions
            ),
        )
        return list(result.embeddings[0].values)

    async def embed_documents(self, documents: list[tuple[str, str]]) -> list[list[float]]:
        contents = [
            types.Content(parts=[types.Part(text=f"title: {title} | text: {content}")])
            for title, content in documents
        ]
        result = await self.clients.genai.aio.models.embed_content(
            model=self.clients.settings.embedding_model,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=self.clients.settings.embedding_dimensions
            ),
        )
        return [list(item.values) for item in result.embeddings]


class GoogleAnswerGenerator:
    def __init__(self, clients: GoogleClients) -> None:
        self.clients = clients

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        stream = await self.clients.genai.aio.models.generate_content_stream(
            model=self.clients.settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=900),
        )
        final_usage = None
        async for chunk in stream:
            final_usage = chunk.usage_metadata or final_usage
            if chunk.text:
                yield chunk.text
        if final_usage:
            logger.info(
                json.dumps(
                    {
                        "event": "model_usage",
                        "model": self.clients.settings.generation_model,
                        "prompt_tokens": final_usage.prompt_token_count,
                        "output_tokens": final_usage.candidates_token_count,
                        "total_tokens": final_usage.total_token_count,
                    }
                )
            )


class GoogleQueryPlanner:
    """Use decomposition only when context or query structure makes it useful."""

    COMPLEX_MARKERS = ("compare", "difference", "versus", " vs ", "both", "across")

    def __init__(self, clients: GoogleClients) -> None:
        self.clients = clients

    async def plan(self, request: AnswerRequest) -> list[str]:
        lowered = f" {request.question.lower()} "
        if not request.history and not any(marker in lowered for marker in self.COMPLEX_MARKERS):
            return [request.question]
        history = "\n".join(f"{turn.role}: {turn.content}" for turn in request.history)
        prompt = f"""Rewrite the latest question as standalone retrieval queries for a report
library.
Return JSON with a `queries` array. Use one query normally and at most three only when comparison
or decomposition is needed. Preserve names, dates, and the user's meaning. Do not answer.

Conversation:
{history}
user: {request.question}
"""
        response = await self.clients.genai.aio.models.generate_content(
            model=self.clients.settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {"queries": {"type": "ARRAY", "items": {"type": "STRING"}}},
                    "required": ["queries"],
                },
            ),
        )
        try:
            queries = json.loads(response.text)["queries"]
        except (TypeError, KeyError, json.JSONDecodeError):
            return [request.question]
        cleaned = [value.strip() for value in queries if isinstance(value, str) and value.strip()]
        return cleaned[:3] or [request.question]


class GoogleSemanticRanker:
    def __init__(self, settings: Settings) -> None:
        if not settings.google_cloud_project:
            raise ValueError("ELDERHELP_GOOGLE_CLOUD_PROJECT is required")
        self.settings = settings
        self.client = discoveryengine.RankServiceAsyncClient()

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], limit: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        ranking_config = self.client.ranking_config_path(
            self.settings.google_cloud_project, "global", self.settings.ranking_config
        )
        response = await self.client.rank(
            request=discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=self.settings.ranking_model,
                top_n=min(limit, len(candidates)),
                query=query,
                records=[
                    discoveryengine.RankingRecord(
                        id=str(candidate.chunk_id),
                        title=candidate.report_title,
                        content=candidate.content,
                    )
                    for candidate in candidates
                ],
            )
        )
        candidates_by_id = {str(candidate.chunk_id): candidate for candidate in candidates}
        return [
            candidates_by_id[record.id]
            for record in response.records
            if record.id in candidates_by_id
        ]


class FallbackRanker:
    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    async def rerank(self, query: str, candidates: list[RetrievedChunk], limit: int):
        try:
            return await self.primary.rerank(query, candidates, limit)
        except Exception:
            return await self.fallback.rerank(query, candidates, limit)
