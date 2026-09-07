from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from typing import Protocol
from uuid import UUID

from elderhelp.schemas import AnswerFilters
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: UUID
    report_id: UUID
    report_title: str
    publisher: str
    source_url: str
    publication_date: date | None
    page_start: int
    page_end: int
    content: str
    parent_content: str
    score: float = 0.0


class EmbeddingProvider(Protocol):
    async def embed_query(self, query: str) -> list[float]: ...


class Ranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], limit: int
    ) -> list[RetrievedChunk]: ...


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], *, k: int = 60
) -> list[RetrievedChunk]:
    """Fuse independent rankings without assuming their scores share a scale."""
    scores: defaultdict[UUID, float] = defaultdict(float)
    chunks: dict[UUID, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] += 1.0 / (k + rank)
    return [
        replace(chunks[chunk_id], score=score)
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def diversify(
    candidates: list[RetrievedChunk], *, limit: int = 8, max_per_report: int = 3
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    report_counts: defaultdict[UUID, int] = defaultdict(int)
    for candidate in candidates:
        if report_counts[candidate.report_id] >= max_per_report:
            continue
        selected.append(candidate)
        report_counts[candidate.report_id] += 1
        if len(selected) == limit:
            break
    return selected


class PostgresHybridRetriever:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _filter_sql(filters: AnswerFilters, params: dict) -> str:
        clauses = ["r.status = 'approved'"]
        if filters.report_ids:
            clauses.append("r.id = ANY(CAST(:report_ids AS uuid[]))")
            params["report_ids"] = [str(value) for value in filters.report_ids]
        if filters.community:
            clauses.append("lower(r.community) = lower(:community)")
            params["community"] = filters.community
        if filters.year_from:
            clauses.append("extract(year from r.publication_date) >= :year_from")
            params["year_from"] = filters.year_from
        if filters.year_to:
            clauses.append("extract(year from r.publication_date) <= :year_to")
            params["year_to"] = filters.year_to
        return " AND ".join(clauses)

    async def retrieve(
        self,
        query: str,
        embedding: list[float],
        filters: AnswerFilters,
        *,
        per_signal: int = 30,
    ) -> list[RetrievedChunk]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        params: dict = {"query": query, "embedding": vector_literal, "limit": per_signal}
        where_sql = self._filter_sql(filters, params)
        fields = """
            c.id AS chunk_id, c.report_id, r.title AS report_title, r.publisher,
            r.source_url, r.publication_date, c.page_start, c.page_end,
            c.content, c.parent_content
        """
        dense_sql = text(
            f"""
            SELECT {fields}, 1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
            FROM chunks c JOIN reports r ON r.id = c.report_id
            WHERE {where_sql}
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        )
        lexical_sql = text(
            f"""
            SELECT {fields},
              ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query)) AS score
            FROM chunks c JOIN reports r ON r.id = c.report_id
            WHERE {where_sql}
              AND c.search_vector @@ websearch_to_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
            """
        )
        dense_rows = (await self.session.execute(dense_sql, params)).mappings().all()
        lexical_rows = (await self.session.execute(lexical_sql, params)).mappings().all()
        rankings = [
            [RetrievedChunk(**dict(row)) for row in dense_rows],
            [RetrievedChunk(**dict(row)) for row in lexical_rows],
        ]
        return reciprocal_rank_fusion(rankings)


class PassthroughRanker:
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], limit: int
    ) -> list[RetrievedChunk]:
        return candidates[:limit]
