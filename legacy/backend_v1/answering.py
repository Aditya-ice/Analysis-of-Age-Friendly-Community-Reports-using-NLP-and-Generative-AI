from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4

from elderhelp.schemas import AnswerComplete, AnswerRequest, Citation
from elderhelp.services.retrieval import (
    EmbeddingProvider,
    Ranker,
    RetrievedChunk,
    diversify,
    reciprocal_rank_fusion,
)

CITATION_PATTERN = re.compile(r"\[S(\d+)]")
logger = logging.getLogger("elderhelp.rag")
INSUFFICIENT_ANSWER = (
    "I couldn't find enough evidence in the approved reports to answer that reliably. "
    "Try asking about a specific community, report, or topic."
)


class QueryPlanner(Protocol):
    async def plan(self, request: AnswerRequest) -> list[str]: ...


class AnswerGenerator(Protocol):
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...


class Retriever(Protocol):
    async def retrieve(
        self, query: str, embedding: list[float], filters, *, per_signal: int = 30
    ) -> list[RetrievedChunk]: ...


class DirectQueryPlanner:
    async def plan(self, request: AnswerRequest) -> list[str]:
        return [request.question]


def citation_for(source_number: int, chunk: RetrievedChunk) -> Citation:
    excerpt = " ".join(chunk.content.split())
    if len(excerpt) > 500:
        excerpt = excerpt[:497].rstrip() + "..."
    return Citation(
        id=f"S{source_number}",
        report_id=chunk.report_id,
        report_title=chunk.report_title,
        publisher=chunk.publisher,
        source_url=chunk.source_url,
        publication_date=chunk.publication_date,
        page_number=chunk.page_start,
        excerpt=excerpt,
    )


def validated_citations(answer: str, sources: list[RetrievedChunk]) -> tuple[list[Citation], bool]:
    referenced = [int(value) for value in CITATION_PATTERN.findall(answer)]
    if not referenced:
        return [], False
    valid_numbers = set(range(1, len(sources) + 1))
    if any(number not in valid_numbers for number in referenced):
        return [], False
    ordered_unique = list(dict.fromkeys(referenced))
    return [citation_for(number, sources[number - 1]) for number in ordered_unique], True


def generation_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    evidence = []
    for index, source in enumerate(sources, start=1):
        evidence.append(
            f"[S{index}] {source.report_title}, page {source.page_start}\n"
            f"{source.parent_content.strip()}"
        )
    return f"""You are ElderHelp, an evidence-grounded assistant for age-friendly community reports.
Answer only from the evidence below. Use plain language. Put a citation such as [S1] after every
factual claim. Never cite an identifier not listed below. If the evidence does not answer the
question, say that the approved reports do not contain enough evidence. Do not give medical,
legal, or financial advice.

Question: {question}

Evidence:
{chr(10).join(evidence)}
"""


@dataclass(slots=True)
class AnswerService:
    planner: QueryPlanner
    embedder: EmbeddingProvider
    retriever: Retriever
    ranker: Ranker
    generator: AnswerGenerator

    async def events(
        self, request: AnswerRequest, request_id: UUID | None = None
    ) -> AsyncIterator[str]:
        request_id = request_id or uuid4()
        yield _sse("start", {"request_id": str(request_id)})
        try:
            async for event in self._answer_events(request, request_id):
                yield event
        except Exception:
            yield _sse(
                "error",
                {
                    "request_id": str(request_id),
                    "code": "answer_failed",
                    "message": "The answer service is temporarily unavailable.",
                },
            )

    async def _answer_events(self, request: AnswerRequest, request_id: UUID) -> AsyncIterator[str]:
        queries = (await self.planner.plan(request))[:3] or [request.question]
        rankings: list[list[RetrievedChunk]] = []
        for query in queries:
            embedding = await self.embedder.embed_query(query)
            rankings.append(await self.retriever.retrieve(query, embedding, request.filters))
        fused = reciprocal_rank_fusion(rankings)
        reranked = await self.ranker.rerank(request.question, fused[:20], 20)
        max_per_report = 8 if request.filters.report_ids else 3
        sources = diversify(reranked, limit=8, max_per_report=max_per_report)
        logger.info(
            json.dumps(
                {
                    "event": "rag_retrieval",
                    "request_id": str(request_id),
                    "subqueries": len(queries),
                    "fused_candidates": len(fused),
                    "selected_sources": len(sources),
                }
            )
        )
        if not sources:
            result = AnswerComplete(
                request_id=request_id,
                answer_markdown=INSUFFICIENT_ANSWER,
                status="insufficient_evidence",
                citations=[],
            )
            yield _sse("complete", result.model_dump(mode="json"))
            return

        pieces: list[str] = []
        async for delta in self.generator.stream(generation_prompt(request.question, sources)):
            if not delta:
                continue
            pieces.append(delta)
            yield _sse("delta", {"text": delta})
        raw_answer = "".join(pieces).strip()
        citations, valid = validated_citations(raw_answer, sources)
        final_answer = raw_answer if valid else INSUFFICIENT_ANSWER
        result = AnswerComplete(
            request_id=request_id,
            answer_markdown=final_answer,
            status="grounded" if valid else "insufficient_evidence",
            citations=citations if valid else [],
        )
        yield _sse("complete", result.model_dump(mode="json"))


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
