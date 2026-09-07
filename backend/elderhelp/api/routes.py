from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from elderhelp import __version__
from elderhelp.catalog import CatalogRepository
from elderhelp.schemas import AnswerRequest, HealthResponse, ReportDetail, ReportList
from elderhelp.services.answering import AnswerService
from elderhelp.services.retrieval import PostgresHybridRetriever

router = APIRouter()


async def session(request: Request):
    async for value in request.app.state.database.session():
        yield value


Session = Annotated[AsyncSession, Depends(session)]


@router.get("/healthz", response_model=HealthResponse, tags=["operations"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, timestamp=datetime.now(UTC))


@router.get("/readyz", response_model=HealthResponse, tags=["operations"])
async def readiness(request: Request, db: Session) -> HealthResponse:
    checks = {
        "database": False,
        "index": False,
        "google_cloud_project": bool(request.app.state.settings.google_cloud_project),
    }
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
        checks["index"] = bool(
            await db.scalar(text("SELECT EXISTS (SELECT 1 FROM chunks LIMIT 1)"))
        )
    except Exception:
        pass
    if not all(checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )
    return HealthResponse(
        status="ok", version=__version__, checks=checks, timestamp=datetime.now(UTC)
    )


@router.get("/v1/reports", response_model=ReportList, tags=["reports"])
async def reports(
    db: Session,
    community: str | None = Query(default=None, max_length=200),
    year: int | None = Query(default=None, ge=1900, le=2200),
) -> ReportList:
    return await CatalogRepository(db).list_reports(community=community, year=year)


@router.get("/v1/reports/{report_id}", response_model=ReportDetail, tags=["reports"])
async def report(report_id: UUID, db: Session) -> ReportDetail:
    result = await CatalogRepository(db).get_report(report_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return result


@router.post(
    "/v1/answers/stream",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE answer stream"},
        422: {"description": "Invalid request"},
        503: {"description": "RAG provider is not configured"},
    },
    tags=["answers"],
)
async def answer_stream(payload: AnswerRequest, request: Request, db: Session) -> StreamingResponse:
    providers = request.app.state.providers
    if providers is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Cloud provider is not configured",
        )
    request_id = getattr(request.state, "request_id", uuid4())
    service = AnswerService(
        planner=providers.planner,
        embedder=providers.embedder,
        retriever=PostgresHybridRetriever(db),
        ranker=providers.ranker,
        generator=providers.generator,
    )
    return StreamingResponse(
        service.events(payload, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Request-ID": str(request_id),
        },
    )
