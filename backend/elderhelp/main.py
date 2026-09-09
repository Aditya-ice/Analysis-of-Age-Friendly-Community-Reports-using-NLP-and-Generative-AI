import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from elderhelp.config import Settings, get_settings
from elderhelp.database import Database
from elderhelp.observability import RequestMetricsMiddleware
from elderhelp.v2.google import Gemini, ProviderUnavailable
from elderhelp.v2.quota import QuotaExceeded
from elderhelp.v2.retrieval import CorpusUnavailable
from elderhelp.v2.routes import router


def create_app(settings: Settings | None = None, *, database=None, provider=None, ranker=None):
    settings = settings or get_settings()
    owned_database = database is None
    database = database or Database(settings)

    @asynccontextmanager
    async def lifespan(app):
        created_provider = False
        if app.state.provider is None and settings.google_api_key and settings.free_tier_confirmed:
            try:
                app.state.provider = Gemini(settings, database)
                created_provider = True
            except (ValueError, ProviderUnavailable):
                app.state.provider = None
        if app.state.ranker is None:
            from elderhelp.v2.reranker import OnnxRanker

            try:
                app.state.ranker = await asyncio.to_thread(
                    OnnxRanker,
                    settings.reranker_directory,
                    Path(__file__).parent / "assets/reranker-manifest.json",
                )
            except (OSError, ValueError, RuntimeError):
                app.state.ranker = None
        try:
            yield
        finally:
            if created_provider:
                await app.state.provider.close()
            if owned_database:
                await database.close()

    app = FastAPI(
        title="ElderHelp API",
        version="2.0.0",
        description="Verified historical research answers from approved reports. "
        "Invite-code pilot; local-only conversation history.",
        lifespan=lifespan,
    )
    app.state.settings, app.state.database = settings, database
    app.state.provider, app.state.ranker = provider, ranker
    app.state.answer_slots = asyncio.Semaphore(2)
    app.add_middleware(RequestMetricsMiddleware, max_request_bytes=settings.max_request_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["Retry-After", "X-Request-ID"],
        )

    @app.exception_handler(QuotaExceeded)
    async def quota_error(request: Request, error: QuotaExceeded):
        return JSONResponse(
            {
                "detail": "Pilot quota is exhausted; keyword search remains available",
                "code": "quota_exhausted",
            },
            429,
            headers={"Retry-After": str(error.retry_after)},
        )

    async def unavailable(request: Request, error):
        return JSONResponse(
            {"detail": "A required service is unavailable. Retry later.", "code": "unavailable"},
            503,
        )

    for category in (CorpusUnavailable, ProviderUnavailable, SQLAlchemyError, TimeoutError):
        app.add_exception_handler(category, unavailable)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error):
        return JSONResponse(
            {
                "detail": [
                    {"loc": e["loc"], "type": e["type"], "msg": "Invalid value"}
                    for e in error.errors()
                ]
            },
            422,
        )

    from elderhelp.web import install

    install(app)
    app.include_router(router)
    return app


app = create_app()
