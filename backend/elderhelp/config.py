from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ELDERHELP_", env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+asyncpg://elderhelp:elderhelp@localhost:5432/elderhelp"
    google_cloud_project: str | None = Field(default=None)
    google_cloud_location: str = "us-central1"
    generation_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: Literal[768] = 768
    ranking_config: str = "default_ranking_config"
    ranking_model: str = "semantic-ranker-default@latest"
    reranking_enabled: bool = True
    report_manifest: Path = Path("data/reports.yaml")
    seed_root: Path = Path("data/seed/pdfs")
    storage_bucket: str | None = None
    document_ai_processor: str | None = None
    max_question_characters: int = 2_000
    max_history_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
