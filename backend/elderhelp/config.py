from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELDERHELP_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    environment: str = "development"
    google_api_key: SecretStr | None = None
    free_tier_confirmed: bool = False
    paid_ingestion_confirmed: bool = False
    paid_budget_microusd: int = Field(default=1_800_000, ge=0, le=1_800_000)
    token_secret: SecretStr | None = Field(default=None, min_length=32)
    token_ttl_seconds: int = Field(default=7200, ge=300, le=28800)
    answers_daily_limit: int = Field(default=20, ge=0, le=20)
    answers_minute_limit: int = Field(default=6, ge=0, le=6)
    session_daily_limit: int = Field(default=5, ge=0, le=20)
    invite_daily_limit: int = Field(default=20, ge=0, le=20)
    answer_timeout_seconds: float = Field(default=60, ge=1, le=90)
    cors_origins: list[str] = []
    max_request_bytes: int = Field(default=65536, ge=4096, le=65536)
    provider_timeout_seconds: float = Field(default=25, ge=1, le=60)
    embedding_daily_limit: int = Field(default=200, ge=0, le=10000)
    generation_daily_limit: int = Field(default=60, ge=0, le=1000)
    generation_minute_limit: int = Field(default=5, ge=1, le=1000)
    embedding_minute_limit: int = Field(default=5, ge=1, le=1000)
    reranker_directory: Path = Path("models/reranker")
    database_budget_bytes: int = 500 * 1024 * 1024

    database_url: str = "postgresql+asyncpg://elderhelp:elderhelp@localhost:5432/elderhelp"
    database_tls: Literal["disable", "verify-full"] = "disable"
    database_ca_file: Path | None = None
    postgres_bin_directory: Path | None = None

    @model_validator(mode="after")
    def hosted_database_tls(self):
        if self.free_tier_confirmed and self.paid_ingestion_confirmed:
            raise ValueError("Select free tier or paid ingestion, not both")
        if self.environment == "pilot" and self.database_tls != "verify-full":
            raise ValueError("The hosted pilot requires verified database TLS")
        return self

    google_cloud_project: str | None = Field(default=None)
    google_cloud_location: str = "us-central1"
    generation_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: Literal[768] = 768

    @field_validator("embedding_dimensions", mode="before")
    @classmethod
    def parse_embedding_dimensions(cls, value):
        # Environment files contain strings; retain the fixed supported dimension.
        return 768 if value == "768" else value

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
