import asyncio
import contextlib
import copy
import json
import logging
import time
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from elderhelp import __version__
from elderhelp.schemas import (
    AnswerComplete,
    AnswerFilters,
    AnswerRequest,
    Citation,
    HealthResponse,
    ReportDetail,
    ReportList,
    ReportSummary,
)
from elderhelp.v2 import catalog
from elderhelp.v2.admission import Admission
from elderhelp.v2.answering import INSUFFICIENT, CorpusChanged, answer
from elderhelp.v2.auth import Pilot, authenticate, issue_session
from elderhelp.v2.contracts import (
    Capabilities,
    CitationV2,
    ErrorEvent,
    ProgressEvent,
    QueryPlan,
    ReportDetailV2,
    ReportListV2,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SessionRequest,
    SessionResponse,
)
from elderhelp.v2.google import ProviderUnavailable
from elderhelp.v2.models import QuotaCounter
from elderhelp.v2.quota import QuotaExceeded, daily, google_retry_after, reserve
from elderhelp.v2.retrieval import (
    CorpusUnavailable,
    active_generation,
    candidates,
    expand,
    retrieve,
)

router = APIRouter()
Access = Annotated[Pilot, Depends(authenticate)]


def sse(name, payload):
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


@router.get("/healthz", response_model=HealthResponse, tags=["operations"])
async def health():
    return HealthResponse(status="ok", version=__version__, timestamp=datetime.now(UTC))


@router.get("/readyz", response_model=HealthResponse, tags=["operations"])
async def readiness(request: Request):
    state = request.app.state
    checks = {
        "database_and_index": False,
        "reranker": bool(state.ranker),
        "pilot_access": bool(state.settings.token_secret),
    }
    try:
        await active_generation(state.database, state.settings)
        checks["database_and_index"] = True
    except Exception:
        pass
    result = HealthResponse(
        status="ok" if all(checks.values()) else "not_ready",
        version=__version__,
        timestamp=datetime.now(UTC),
        checks=checks,
    )
    return JSONResponse(
        result.model_dump(mode="json"), status_code=200 if all(checks.values()) else 503
    )


@router.post("/v2/demo/session", response_model=SessionResponse, tags=["pilot"])
async def session(payload: SessionRequest, request: Request):
    return await issue_session(
        request.app.state.database,
        request.app.state.settings,
        payload.invite_code,
        request.client.host if request.client else "unknown",
    )


async def capability(state):
    try:
        generation = await active_generation(
            state.database, state.settings, require_compatible=False
        )
    except CorpusUnavailable:
        return Capabilities(
            generation_available=False, search_available=False, reason="index_not_ready"
        )
    reason = None
    try:
        await active_generation(state.database, state.settings)
    except CorpusUnavailable:
        reason = "index_incompatible"
    if not state.provider:
        reason = "google_not_configured"
    if not state.ranker:
        reason = "reranker_not_ready"
    limits = [
        daily("answers", 1, state.settings.answers_daily_limit),
        daily("generation", 2, state.settings.generation_daily_limit),
        daily("embedding", 1, state.settings.embedding_daily_limit),
    ]
    async with state.database.sessions() as db:
        for key, amount, maximum, _ in limits:
            counter = await db.get(QuotaCounter, key)
            if (counter.used if counter else 0) + amount > maximum:
                reason = "quota_exhausted"
    if await google_retry_after(state.database):
        reason = "quota_exhausted"
    return Capabilities(
        generation_available=reason is None,
        search_available=True,
        corpus_generation=generation,
        reason=reason,
    )


@router.get("/v2/capabilities", response_model=Capabilities, tags=["pilot"])
async def capabilities(request: Request, access: Access):
    return await capability(request.app.state)


@router.get("/v2/reports", response_model=ReportListV2, tags=["reports"])
async def reports(
    request: Request,
    access: Access,
    community: str | None = Query(None, max_length=200),
    year: int | None = Query(None, ge=1900, le=2200),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    return await catalog.list_reports(
        request.app.state.database,
        request.app.state.settings,
        AnswerFilters(community=community, year_from=year, year_to=year),
        offset,
        limit,
    )


@router.get("/v2/reports/{report_id}", response_model=ReportDetailV2, tags=["reports"])
async def report(report_id: UUID, request: Request, access: Access):
    result = await catalog.get_report(
        request.app.state.database, request.app.state.settings, report_id
    )
    if not result:
        raise HTTPException(404, "Report not found")
    return result


@router.get("/v1/reports", response_model=ReportList, tags=["v1 compatibility"])
async def reports_v1(
    request: Request,
    access: Access,
    community: str | None = Query(None, max_length=200),
    year: int | None = Query(None, ge=1900, le=2200),
):
    result = await catalog.list_reports(
        request.app.state.database,
        request.app.state.settings,
        AnswerFilters(community=community, year_from=year, year_to=year),
        0,
        100,
    )
    items = []
    for item in result.items:
        data = item.model_dump()
        if item.publication_precision != "day":
            data["publication_date"] = None
        items.append(ReportSummary.model_validate(data))
    return ReportList(items=items, total=result.total)


@router.get("/v1/reports/{report_id}", response_model=ReportDetail, tags=["v1 compatibility"])
async def report_v1(report_id: UUID, request: Request, access: Access):
    result = await report(report_id, request, access)
    data = result.model_dump()
    if result.publication_precision != "day":
        data["publication_date"] = None
    return ReportDetail.model_validate(data)


@router.post("/v2/search", response_model=SearchResponse, tags=["search"])
async def search(payload: SearchRequest, request: Request, access: Access):
    state = request.app.state
    generation = await active_generation(state.database, state.settings, require_compatible=False)
    mode, degraded = "keyword", False
    selected = []
    minute = int(datetime.now(UTC).timestamp()) // 60
    expires = datetime.fromtimestamp((minute + 1) * 60, UTC)
    await reserve(
        state.database,
        [
            (f"search:{minute}", 1, 60, expires),
            (f"search:{access.session_id}:{minute}", 1, 20, expires),
        ],
    )
    if payload.mode == "hybrid" and state.provider:
        try:
            result = await retrieve(
                state.database,
                state.provider,
                state.ranker,
                state.settings,
                QueryPlan(
                    original_question=payload.question,
                    standalone_question=payload.question,
                    subqueries=[payload.question],
                    required_parts=[payload.question],
                ),
                payload.filters,
            )
            selected, mode, degraded = result.blocks, "hybrid", result.degraded
        except (QuotaExceeded, CorpusUnavailable, ProviderUnavailable, TimeoutError):
            degraded = True
    if mode == "keyword":
        ranked = await candidates(
            state.database, generation, payload.question, payload.filters, limit=payload.limit
        )
        selected = [await expand(state.database, generation, c, budget=400) for c in ranked]
    hits, seen = [], set()
    for block in selected:
        # Return the strongest matching child excerpt, never a complete document or path.
        span = next(s for s in block.spans if str(s.id) in block.candidate.span_ids)
        if span.id in seen:
            continue
        seen.add(span.id)
        hits.append(
            SearchHit(
                score=block.candidate.score,
                citation=CitationV2(
                    id=f"S{len(hits) + 1}",
                    report_id=span.report_id,
                    revision_id=span.revision_id,
                    span_id=span.id,
                    report_title=span.report_title,
                    publisher=span.publisher,
                    source_url=span.source_url,
                    publication_date=span.publication_date,
                    page_number=span.page_number,
                    page_label=span.page_label,
                    excerpt=span.text,
                ),
            )
        )
    return SearchResponse(items=hits[: payload.limit], mode=mode, degraded=degraded)


SSE_RESPONSES = {
    200: {
        "description": "SSE start, progress, verified delta, complete; typed error on failure",
        "content": {"text/event-stream": {}},
        "x-sse-events": {
            "start": "StartEvent",
            "progress": "ProgressEvent",
            "delta": "DeltaEvent",
            "complete": "CompleteV2",
            "error": "ErrorEvent",
        },
    },
    401: {"description": "Invite/session required"},
    413: {"description": "Request body too large"},
    422: {"description": "Invalid input"},
    429: {"description": "Quota/concurrency limit; Retry-After header"},
    503: {"description": "Required dependency unavailable"},
}


V1_SSE_RESPONSES = copy.deepcopy(SSE_RESPONSES)
V1_SSE_RESPONSES[200]["x-sse-events"].pop("progress")
V1_SSE_RESPONSES[200]["x-sse-events"]["complete"] = "AnswerComplete"
V1_SSE_RESPONSES[200]["description"] = "SSE start, verified delta, complete; typed error on failure"


async def stream(payload, request, pilot, v1=False):
    state = request.app.state
    if not state.provider or not state.ranker:
        raise HTTPException(503, "Verified answers are unavailable; use keyword search")
    generation = await active_generation(state.database, state.settings)
    retry_after = await google_retry_after(state.database)
    if retry_after:
        raise QuotaExceeded(retry_after)
    admission = Admission(state, pilot, payload)
    provider = await admission.acquire()
    request_id = request.state.request_id

    async def events():
        started, outcome = time.monotonic(), "incomplete"
        task = None
        queue = asyncio.Queue(maxsize=4)

        async def progress(stage, message):
            await queue.put(ProgressEvent(stage=stage, message=message))

        try:
            yield sse("start", {"request_id": str(request_id)})

            async def bounded():
                async with asyncio.timeout(state.settings.answer_timeout_seconds):
                    return await answer(state, provider, payload, request_id, progress)

            task = asyncio.create_task(bounded())
            while not task.done() or not queue.empty():
                if not queue.empty():
                    message = queue.get_nowait()
                    if not v1:
                        yield sse("progress", message)
                    continue
                pending = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {pending, task}, timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
                if pending in done:
                    message = pending.result()
                    if not v1:
                        yield sse("progress", message)
                else:
                    pending.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pending
                    if not done:
                        yield ": heartbeat\n\n"
            complete = task.result()
            if v1:
                representable = complete.status == "grounded"
                complete = AnswerComplete(
                    request_id=request_id,
                    answer_markdown=complete.answer_markdown if representable else INSUFFICIENT,
                    status="grounded" if representable else "insufficient_evidence",
                    citations=[
                        Citation.model_validate(
                            {
                                **c.model_dump(),
                                "publication_date": c.publication_date
                                if c.publication_date and len(c.publication_date) == 10
                                else None,
                            }
                        )
                        for c in complete.citations
                    ]
                    if representable
                    else [],
                )
            outcome = complete.status
            yield sse("delta", {"text": complete.answer_markdown})
            yield sse("complete", complete)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            code = (
                "quota_exhausted"
                if isinstance(exc, QuotaExceeded)
                else "timeout"
                if isinstance(exc, TimeoutError)
                else "corpus_changed"
                if isinstance(exc, CorpusChanged)
                else "unavailable"
            )
            outcome = code
            yield sse(
                "error",
                ErrorEvent(
                    request_id=request_id,
                    code=code,
                    message=(
                        "A verified answer could not be completed. "
                        "Try keyword search or retry later."
                    ),
                ),
            )
        finally:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await asyncio.shield(admission.close())
            logging.getLogger("elderhelp.metrics").info(
                json.dumps(
                    {
                        "event": "answer_workflow",
                        "request_id": str(request_id),
                        "outcome": outcome,
                        "model": state.settings.generation_model,
                        "index": str(generation),
                        "code_version": __version__,
                        "prompt_version": "claims-v2.1",
                        "completion_seconds": time.monotonic() - started,
                        "generation_remaining": provider.generation_left,
                        "embedding_remaining": provider.embedding_left,
                    }
                )
            )

    class ManagedStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                await asyncio.shield(admission.close())

    return ManagedStream(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
            "X-Request-ID": str(request_id),
        },
    )


@router.post(
    "/v2/answers/stream",
    response_class=StreamingResponse,
    responses=SSE_RESPONSES,
    tags=["answers"],
)
async def answer_stream(payload: AnswerRequest, request: Request, access: Access):
    return await stream(payload, request, access)


@router.post(
    "/v1/answers/stream",
    response_class=StreamingResponse,
    responses=V1_SSE_RESPONSES,
    tags=["v1 compatibility"],
)
async def answer_stream_v1(payload: AnswerRequest, request: Request, access: Access):
    return await stream(payload, request, access, v1=True)
