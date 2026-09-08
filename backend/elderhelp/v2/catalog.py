from uuid import UUID

from sqlalchemy import func, select

from elderhelp.models import Report
from elderhelp.schemas import AnswerFilters, ReportSummary
from elderhelp.v2.contracts import ReportDetailV2, ReportListV2
from elderhelp.v2.models import GenerationChunk, ResearchChunk, ResearchPage, Revision
from elderhelp.v2.retrieval import active_generation, filters_for


def visible(generation, filters):
    included = (
        select(Revision.report_id)
        .join(ResearchChunk, ResearchChunk.revision_id == Revision.id)
        .join(GenerationChunk, GenerationChunk.chunk_id == ResearchChunk.id)
        .where(GenerationChunk.generation_id == generation)
    )
    return select(Report).where(Report.id.in_(included), *filters_for(filters))


def metadata(report):
    return ReportSummary.model_validate(
        {name: getattr(report, name) for name in ReportSummary.model_fields}
    )


async def list_reports(database, settings, filters, offset=0, limit=20):
    generation = await active_generation(database, settings, require_compatible=False)
    async with database.sessions() as db:
        statement = visible(generation, filters)
        total = await db.scalar(select(func.count()).select_from(statement.subquery()))
        rows = (
            await db.scalars(
                statement.order_by(Report.title, Report.id).offset(offset).limit(limit)
            )
        ).all()
    return ReportListV2(items=[metadata(r) for r in rows], total=total, offset=offset, limit=limit)


async def get_report(database, settings, report_id: UUID):
    generation = await active_generation(database, settings, require_compatible=False)
    async with database.sessions() as db:
        row = await db.scalar(visible(generation, AnswerFilters(report_ids=[report_id])))
        if not row:
            return None
        revision = await db.scalar(
            select(Revision)
            .join(ResearchChunk, ResearchChunk.revision_id == Revision.id)
            .join(GenerationChunk, GenerationChunk.chunk_id == ResearchChunk.id)
            .where(GenerationChunk.generation_id == generation, Revision.report_id == report_id)
        )
        pages = await db.scalar(
            select(func.count(func.distinct(ResearchPage.page_number))).where(
                ResearchPage.revision_id == revision.id
            )
        )
        return ReportDetailV2(
            **metadata(row).model_dump(),
            revision_id=revision.id,
            acquired_at=revision.acquired_at,
            description=row.description,
            suggested_questions=row.suggested_questions,
            page_count=pages,
        )
