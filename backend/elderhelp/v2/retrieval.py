"""Exact filtered hybrid retrieval; database sessions end before any provider work."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from uuid import UUID, uuid5

from sqlalchemy import Text, cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY

from elderhelp.models import Report
from elderhelp.schemas import AnswerFilters
from elderhelp.v2.contracts import EvidenceSpan, QueryPlan
from elderhelp.v2.corpus import fingerprint, index_configuration
from elderhelp.v2.dates import publication_text
from elderhelp.v2.models import (
    ActiveCorpus,
    Embedding,
    Generation,
    GenerationChunk,
    ResearchChunk,
    ResearchPage,
    Revision,
    Section,
    SourceSpan,
)


class CorpusUnavailable(RuntimeError):
    pass


@dataclass
class Candidate:
    id: UUID
    report_id: UUID
    revision_id: UUID
    section_id: UUID
    title: str
    content: str
    span_ids: list[str]
    score: float = 0
    parts: set[int] = field(default_factory=set)
    part_scores: dict[int, float] = field(default_factory=dict)


@dataclass
class EvidenceBlock:
    candidate: Candidate
    spans: list[EvidenceSpan]

    def payload(self):
        first = self.spans[0]
        return {
            "report": {
                "title": first.report_title,
                "publisher": first.publisher,
                "publication_date": first.publication_date,
                "revision": str(first.revision_id),
            },
            "parts": sorted(self.candidate.parts),
            "child_span_ids": self.candidate.span_ids,
            "spans": [{"id": str(s.id), "text": s.text, "page": s.page_number} for s in self.spans],
        }


@dataclass
class RetrievalResult:
    generation: UUID
    blocks: list[EvidenceBlock]
    candidates: int
    degraded: bool = False


def filters_for(filters: AnswerFilters):
    clauses = [Report.status == "approved"]
    if filters.report_ids:
        clauses.append(Report.id.in_(filters.report_ids))
    if filters.community:
        clauses.append(Report.community == filters.community)
    if filters.year_from:
        clauses.append(Report.publication_date >= date(filters.year_from, 1, 1))
    if filters.year_to:
        clauses.append(Report.publication_date <= date(filters.year_to, 12, 31))
    return clauses


async def active_generation(database, settings, *, require_compatible=True) -> UUID:
    async with database.sessions() as db:
        row = await db.scalar(
            select(Generation)
            .join(ActiveCorpus, ActiveCorpus.generation_id == Generation.id)
            .where(ActiveCorpus.id == 1)
        )
        if not row or row.status != "validated":
            raise CorpusUnavailable("No validated active corpus")
        if require_compatible and row.fingerprint != fingerprint(index_configuration(settings)):
            raise CorpusUnavailable("Active index is incompatible with configured embeddings")
        return row.id


def base_candidates(generation, filters):
    return (
        select(ResearchChunk, Revision, Report)
        .join(GenerationChunk, GenerationChunk.chunk_id == ResearchChunk.id)
        .join(Revision, ResearchChunk.revision_id == Revision.id)
        .join(Report, Revision.report_id == Report.id)
        .where(GenerationChunk.generation_id == generation, *filters_for(filters))
    )


async def candidates(database, generation, query, filters, *, vector=None, limit=30):
    statement = base_candidates(generation, filters)

    def lexical(terms):
        title = func.setweight(func.to_tsvector("english", Report.title), literal_column("'A'"))
        document = ResearchChunk.search_vector.op("||")(title)
        return statement.where(document.op("@@")(terms)).order_by(
            func.ts_rank_cd(document, terms).desc(), ResearchChunk.id
        )

    if vector is None:
        selected = lexical(func.websearch_to_tsquery("english", query))
    else:
        selected = statement.join(Embedding, ResearchChunk.embedding_key == Embedding.key).order_by(
            Embedding.vector.cosine_distance(vector), ResearchChunk.id
        )
    async with database.sessions() as db:
        rows = (await db.execute(selected.limit(limit))).all()
        # Natural-language questions often have no all-terms match. Broaden only then;
        # explicit phrase, exclusion and OR syntax keep their user-requested semantics.
        if not rows and vector is None and not re.search(r'"|(?:^|\s)-\w|\bOR\b', query):
            terms = cast(
                func.replace(cast(func.plainto_tsquery("english", query), Text), " & ", " | "),
                TSQUERY,
            )
            rows = (await db.execute(lexical(terms).limit(limit))).all()
    return [
        Candidate(c.id, r.id, revision.id, c.section_id, r.title, c.content, c.span_ids)
        for c, revision, r in rows
    ]


def fuse(rankings: list[tuple[int, list[Candidate]]], k=60) -> list[Candidate]:
    scores = defaultdict(float)
    objects = {}
    part_scores = defaultdict(lambda: defaultdict(float))
    parts = defaultdict(set)
    for part, ranking in rankings:
        for position, item in enumerate(ranking, 1):
            scores[item.id] += 1 / (k + position)
            part_scores[item.id][part] += 1 / (k + position)
            objects[item.id] = item
            parts[item.id].add(part)
    result = []
    for key in sorted(scores, key=lambda i: (-scores[i], str(i))):
        item = objects[key]
        item.score, item.parts = scores[key], parts[key]
        item.part_scores = dict(part_scores[key])
        # Drop near-identical overlapping spans but retain the union of query coverage.
        duplicate = next(
            (
                c
                for c in result
                if c.revision_id == item.revision_id
                and len(set(c.span_ids) & set(item.span_ids))
                / max(1, min(len(c.span_ids), len(item.span_ids)))
                >= 0.8
            ),
            None,
        )
        if duplicate:
            duplicate.parts |= item.parts
        else:
            result.append(item)
    return result


def shortlist(candidates: list[Candidate], parts: int, limit=20):
    chosen = []
    # Protect at least one candidate per retrieval subquery before the global cutoff.
    for part in range(parts):
        match = max(
            (c for c in candidates if part in c.parts and c not in chosen),
            key=lambda c: c.part_scores.get(part, c.score) * c.score,
            default=None,
        )
        if match:
            chosen.append(match)
    chosen.extend(c for c in candidates if c not in chosen)
    return chosen[:limit]


def select_evidence(candidates: list[Candidate], parts: int, limit=8):
    chosen, covered = [], set()
    for part in range(parts):
        candidate = max(
            (c for c in candidates if part in c.parts and c not in chosen),
            key=lambda c: c.part_scores.get(part, c.score) * c.score,
            default=None,
        )
        if candidate:
            chosen.append(candidate)
            covered |= candidate.parts
    while len(chosen) < limit:
        remaining = [c for c in candidates if c not in chosen]
        if not remaining:
            break
        counts = defaultdict(int)
        for c in chosen:
            counts[c.report_id] += 1
        # Soft diversity preference, with no hard per-report exclusion.
        pick = max(remaining, key=lambda c: c.score / (1 + 0.25 * counts[c.report_id]))
        chosen.append(pick)
    return chosen


async def expand(database, generation, candidate: Candidate, *, budget=750) -> EvidenceBlock:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    async with database.sessions() as db:
        config = (await db.get(Generation, generation)).fingerprint
        statement = (
            select(SourceSpan, ResearchPage, Revision, Report, Section)
            .join(ResearchPage, SourceSpan.page_id == ResearchPage.id)
            .join(Revision, ResearchPage.revision_id == Revision.id)
            .join(Report, Revision.report_id == Report.id)
            .join(Section, SourceSpan.section_id == Section.id)
            .where(
                Revision.id == candidate.revision_id,
                SourceSpan.section_id == candidate.section_id,
                SourceSpan.searchable,
                Report.status == "approved",
            )
        )
        children = (
            await db.execute(
                statement.where(SourceSpan.id.in_([UUID(s) for s in candidate.span_ids]))
            )
        ).all()
        if not children:
            raise CorpusUnavailable("Selected child evidence is no longer approved")
        first_page = min(row[1].page_number for row in children)
        first_offset = min(row[0].start for row in children)
        neighbors = (
            await db.execute(
                statement.where(
                    ~SourceSpan.id.in_([UUID(s) for s in candidate.span_ids]),
                    ResearchPage.page_number.between(first_page - 1, first_page + 1),
                )
                .order_by(
                    func.abs(ResearchPage.page_number - first_page),
                    func.abs(SourceSpan.start - first_offset),
                    SourceSpan.id,
                )
                .limit(40)
            )
        ).all()
        rows = sorted([*children, *neighbors], key=lambda row: (row[1].page_number, row[0].start))
    mapped = {}
    order = []
    for span, page, revision, report, section in rows:
        if page.id != uuid5(revision.id, f"page:{config}:{page.page_number}"):
            continue
        if page.text[span.start : span.end] != span.text:
            raise CorpusUnavailable("Invalid persisted source span")
        mapped[str(span.id)] = EvidenceSpan(
            id=span.id,
            revision_id=revision.id,
            report_id=report.id,
            report_title=report.title,
            publisher=report.publisher,
            source_url=revision.metadata_snapshot["source_url"],
            publication_date=publication_text(
                revision.metadata_snapshot.get("publication_date"),
                revision.metadata_snapshot.get("publication_precision", "unknown"),
            ),
            page_number=page.page_number,
            page_label=page.page_label,
            start=span.start,
            end=span.end,
            text=span.text,
            section=section.heading,
            kind=span.kind,
        )
        order.append(str(span.id))
    if any(key not in mapped for key in candidate.span_ids):
        raise CorpusUnavailable("Selected child evidence is no longer approved")
    selected = list(candidate.span_ids)
    tokens = sum(len(enc.encode(mapped[key].text, disallowed_special=())) for key in selected)
    if tokens > budget:
        raise CorpusUnavailable("Selected child exceeds evidence budget")
    positions = [order.index(key) for key in selected]
    neighbors = sorted(
        [i for i in range(len(order)) if order[i] not in selected],
        key=lambda i: min(abs(i - p) for p in positions),
    )
    for index in neighbors:
        key = order[index]
        cost = len(enc.encode(mapped[key].text, disallowed_special=()))
        if tokens + cost > budget:
            continue
        selected.append(key)
        tokens += cost
    # Child is preserved in full even if its matching text is near the end of a large section.
    return EvidenceBlock(candidate, [mapped[key] for key in order if key in selected])


async def retrieve(
    database,
    provider,
    ranker,
    settings,
    plan: QueryPlan,
    filters: AnswerFilters,
    *,
    mode="hybrid",
    reserved=False,
) -> RetrievalResult:
    generation = await active_generation(database, settings, require_compatible=mode != "keyword")
    rankings, degraded = [], False
    for part, query in enumerate(plan.subqueries):
        if mode != "dense":
            keywords = await candidates(database, generation, query, filters)
            rankings.append((part, keywords))
        if mode != "keyword":
            vector = await provider.embed(
                f"task: question answering | query: {query}", reserved=reserved
            )
            dense = await candidates(database, generation, query, filters, vector=vector)
            rankings.append((part, dense))
    merged = fuse(rankings)
    best = shortlist(merged, len(plan.subqueries))
    if best and ranker and mode != "keyword":
        original_scores = {c.id: c.score for c in best}
        try:
            best = await ranker.rerank(plan.standalone_question, best)
        except Exception:
            for c in best:
                c.score = original_scores[c.id]
            degraded = True
    chosen = select_evidence(best, len(plan.subqueries))
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    budget = 6000
    blocks, seen = [], set()
    for candidate in chosen:
        block = await expand(database, generation, candidate)
        # Keep each selected child, deduplicating repeated parent-only context.
        block.spans = [
            s for s in block.spans if s.id not in seen or str(s.id) in candidate.span_ids
        ]

        def cost(block):
            return len(enc.encode(json.dumps(block.payload()), disallowed_special=()))

        while cost(block) > budget:
            removable = [s for s in block.spans if str(s.id) not in candidate.span_ids]
            if not removable:
                break
            block.spans.remove(removable[-1])
        if cost(block) > budget:
            continue
        budget -= cost(block)
        seen.update(s.id for s in block.spans)
        blocks.append(block)
    return RetrievalResult(generation, blocks, len(merged), degraded)


async def still_approved(database, generation, evidence: list[EvidenceBlock]):
    expected = {b.candidate.report_id for b in evidence}
    async with database.sessions() as db:
        active = await db.get(ActiveCorpus, 1)
        current = set(
            (
                await db.scalars(
                    select(Report.id).where(Report.id.in_(expected), Report.status == "approved")
                )
            ).all()
        )
    return bool(active and active.generation_id == generation and current == expected)
