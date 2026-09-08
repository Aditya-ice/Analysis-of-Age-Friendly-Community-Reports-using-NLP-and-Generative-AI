from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class ReportSummary(BaseModel):
    id: UUID
    slug: str
    title: str
    publisher: str
    community: str
    publication_date: date | None = None
    source_url: HttpUrl


class ReportDetail(ReportSummary):
    description: str | None = None
    suggested_questions: list[str] = Field(default_factory=list)
    page_count: int


class ReportList(BaseModel):
    items: list[ReportSummary]
    total: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8_000)


class AnswerFilters(BaseModel):
    report_ids: list[UUID] = Field(default_factory=list, max_length=20)
    community: str | None = Field(default=None, max_length=200)
    year_from: int | None = Field(default=None, ge=1900, le=2200)
    year_to: int | None = Field(default=None, ge=1900, le=2200)

    @model_validator(mode="after")
    def ordered_years(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("Starting year must not exceed ending year")
        return self


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)
    filters: AnswerFilters = Field(default_factory=AnswerFilters)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must contain text")
        return value


class Citation(BaseModel):
    id: str
    report_id: UUID
    report_title: str
    publisher: str
    source_url: HttpUrl
    publication_date: date | None = None
    page_number: int
    excerpt: str


class AnswerComplete(BaseModel):
    request_id: UUID
    answer_markdown: str
    status: Literal["grounded", "insufficient_evidence"]
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    version: str
    checks: dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime
