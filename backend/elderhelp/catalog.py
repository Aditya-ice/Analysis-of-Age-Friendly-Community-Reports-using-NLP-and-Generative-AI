from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from elderhelp.models import Page, Report
from elderhelp.schemas import ReportDetail, ReportList, ReportSummary


class CatalogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_reports(
        self, *, community: str | None = None, year: int | None = None
    ) -> ReportList:
        statement = select(Report).where(Report.status == "approved")
        if community:
            statement = statement.where(Report.community.ilike(f"%{community}%"))
        if year:
            statement = statement.where(func.extract("year", Report.publication_date) == year)
        statement = statement.order_by(Report.publication_date.desc().nullslast(), Report.title)
        reports = list((await self.session.scalars(statement)).all())
        items = [
            ReportSummary(
                id=report.id,
                slug=report.slug,
                title=report.title,
                publisher=report.publisher,
                community=report.community,
                publication_date=report.publication_date,
                source_url=report.source_url,
            )
            for report in reports
        ]
        return ReportList(items=items, total=len(items))

    async def get_report(self, report_id: UUID) -> ReportDetail | None:
        statement = (
            select(Report, func.count(Page.id))
            .outerjoin(Page, Page.report_id == Report.id)
            .where(Report.id == report_id, Report.status == "approved")
            .group_by(Report.id)
        )
        row = (await self.session.execute(statement)).one_or_none()
        if not row:
            return None
        report, page_count = row
        return ReportDetail(
            id=report.id,
            slug=report.slug,
            title=report.title,
            publisher=report.publisher,
            community=report.community,
            publication_date=report.publication_date,
            source_url=report.source_url,
            description=report.description,
            suggested_questions=report.suggested_questions,
            page_count=page_count,
        )
