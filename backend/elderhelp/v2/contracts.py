from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elderhelp.schemas import AnswerFilters, AnswerRequest, Citation, ReportSummary


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryPlan(StrictModel):
    original_question: str
    standalone_question: str = Field(min_length=1, max_length=2000)
    intent: Literal["direct", "comparison", "followup", "out_of_scope"] = "direct"
    subqueries: list[str] = Field(min_length=1, max_length=3)
    required_parts: list[str] = Field(default_factory=list, max_length=3)
    clarification: str | None = None
    filters: AnswerFilters = Field(default_factory=AnswerFilters)


class EvidenceSpan(StrictModel):
    id: UUID
    revision_id: UUID
    report_id: UUID
    report_title: str
    publisher: str
    source_url: str
    publication_date: str | None = None
    page_number: int = Field(ge=1)
    page_label: str | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text: str = Field(min_length=1)
    section: str = ""
    kind: str = "paragraph"

    @model_validator(mode="after")
    def offsets_match(self):
        if self.end - self.start != len(self.text):
            raise ValueError("Source span offsets must match normalized text")
        return self


class Claim(StrictModel):
    id: str = Field(pattern=r"^C[1-9][0-9]*$")
    text: str = Field(min_length=1, max_length=800)
    span_ids: list[UUID] = Field(min_length=1, max_length=4)
    part: int = Field(default=0, ge=0, le=2)
    entities: list[str] = Field(default_factory=list, max_length=20)
    quotations: list[str] = Field(default_factory=list, max_length=4)


class Draft(StrictModel):
    answerability: Literal["full", "partial", "insufficient", "clarification_required"]
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    missing_parts: list[int] = Field(default_factory=list, max_length=3)


class ClaimVerdict(StrictModel):
    claim_id: str
    supported: bool
    category: Literal["supported", "unsupported", "contradicted", "temporal", "incomplete"]


class Verification(StrictModel):
    verdicts: list[ClaimVerdict]
    answers_question: bool
    covered_parts: list[int] = Field(default_factory=list, max_length=3)


class CitationV2(Citation):
    revision_id: UUID
    span_id: UUID
    page_label: str | None = None


class CompleteV2(BaseModel):
    request_id: UUID
    answer_markdown: str
    status: Literal["grounded", "partial", "insufficient_evidence", "clarification_required"]
    citations: list[CitationV2] = Field(default_factory=list)
    missing_parts: list[str] = Field(default_factory=list)
    corpus_generation: UUID | None = None


class StartEvent(BaseModel):
    request_id: UUID


class ProgressEvent(BaseModel):
    stage: Literal["planning", "retrieving", "verifying"]
    message: str


class DeltaEvent(BaseModel):
    text: str


class ErrorEvent(BaseModel):
    request_id: UUID
    code: Literal["quota_exhausted", "timeout", "unavailable", "corpus_changed"]
    message: str
    search_available: bool = True


class SessionRequest(StrictModel):
    invite_code: str = Field(min_length=12, max_length=256)


class SessionResponse(BaseModel):
    token: str
    expires_at: int


class Capabilities(BaseModel):
    generation_available: bool
    search_available: bool
    reason: str | None = None
    corpus_generation: UUID | None = None
    supported_filters: list[str] = ["report_ids", "community", "year_from", "year_to"]


class ReportListV2(BaseModel):
    items: list[ReportSummary]
    total: int
    offset: int
    limit: int


class SearchRequest(AnswerRequest):
    mode: Literal["keyword", "hybrid"] = "keyword"
    limit: int = Field(default=8, ge=1, le=20)


class SearchHit(BaseModel):
    citation: CitationV2
    score: float


class SearchResponse(BaseModel):
    items: list[SearchHit]
    mode: Literal["keyword", "hybrid"]
    degraded: bool = False


PUBLIC_EVENT_SCHEMAS = [StartEvent, ProgressEvent, DeltaEvent, ErrorEvent, CompleteV2]
