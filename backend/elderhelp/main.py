from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from elderhelp.api.routes import router
from elderhelp.config import Settings, get_settings
from elderhelp.database import Database
from elderhelp.observability import RequestMetricsMiddleware
from elderhelp.providers.google import (
    FallbackRanker,
    GoogleAnswerGenerator,
    GoogleClients,
    GoogleEmbeddingProvider,
    GoogleQueryPlanner,
    GoogleSemanticRanker,
)
from elderhelp.services.retrieval import PassthroughRanker


@dataclass(slots=True)
class Providers:
    planner: GoogleQueryPlanner
    embedder: GoogleEmbeddingProvider
    generator: GoogleAnswerGenerator
    ranker: object


def build_providers(settings: Settings) -> Providers | None:
    if not settings.google_cloud_project:
        return None
    clients = GoogleClients(settings)
    fallback = PassthroughRanker()
    ranker = (
        FallbackRanker(GoogleSemanticRanker(settings), fallback)
        if settings.reranking_enabled
        else fallback
    )
    return Providers(
        planner=GoogleQueryPlanner(clients),
        embedder=GoogleEmbeddingProvider(clients),
        generator=GoogleAnswerGenerator(clients),
        ranker=ranker,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await database.close()

    app = FastAPI(
        title="ElderHelp API",
        version="1.0.0",
        description="Evidence-grounded answers from approved age-friendly reports.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.providers = build_providers(settings)
    app.add_middleware(RequestMetricsMiddleware)
    app.include_router(router)
    return app


app = create_app()
