"""Additive v2 tables. Legacy page/chunk data remains untouched and is never activated."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from elderhelp.models import Base


def now():
    return datetime.now(UTC)


class Revision(Base):
    __tablename__ = "report_revisions"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    report_id: Mapped[UUID] = mapped_column(ForeignKey("reports.id"), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    metadata_snapshot: Mapped[dict] = mapped_column(JSONB)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Generation(Base):
    __tablename__ = "index_generations"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    configuration: Mapped[dict] = mapped_column(JSONB)
    manifest_snapshot: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(24), default="staging")
    validation: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ActiveCorpus(Base):
    __tablename__ = "active_corpus"
    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    generation_id: Mapped[UUID | None] = mapped_column(ForeignKey("index_generations.id"))
    previous_id: Mapped[UUID | None] = mapped_column(ForeignKey("index_generations.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ResearchPage(Base):
    __tablename__ = "research_pages"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("report_revisions.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    page_label: Mapped[str | None] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(40))
    quality: Mapped[float] = mapped_column(Float)
    issues: Mapped[list] = mapped_column(JSONB, default=list)


class Section(Base):
    __tablename__ = "sections"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("report_revisions.id"), index=True)
    heading: Mapped[str] = mapped_column(Text)
    parent_id: Mapped[UUID | None] = mapped_column(ForeignKey("sections.id"))
    ordinal: Mapped[int] = mapped_column(Integer)


class SourceSpan(Base):
    __tablename__ = "source_spans"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    page_id: Mapped[UUID] = mapped_column(ForeignKey("research_pages.id"), index=True)
    section_id: Mapped[UUID] = mapped_column(ForeignKey("sections.id"), index=True)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(30))
    bbox: Mapped[list] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    searchable: Mapped[bool] = mapped_column(Boolean, default=True)


class Embedding(Base):
    __tablename__ = "chunk_embeddings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    vector: Mapped[list[float]] = mapped_column(Vector(768))


class ResearchChunk(Base):
    __tablename__ = "research_chunks"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("report_revisions.id"), index=True)
    section_id: Mapped[UUID] = mapped_column(ForeignKey("sections.id"))
    span_ids: Mapped[list] = mapped_column(JSONB)
    content: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer)
    embedding_key: Mapped[str] = mapped_column(ForeignKey("chunk_embeddings.key"))
    search_vector: Mapped[str] = mapped_column(TSVECTOR)


Index("ix_research_chunks_fts", ResearchChunk.search_vector, postgresql_using="gin")


class GenerationChunk(Base):
    __tablename__ = "generation_chunks"
    generation_id: Mapped[UUID] = mapped_column(
        ForeignKey("index_generations.id"), primary_key=True
    )
    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("research_chunks.id"), primary_key=True)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    generation_id: Mapped[UUID] = mapped_column(ForeignKey("index_generations.id"), index=True)
    report_id: Mapped[UUID] = mapped_column(ForeignKey("reports.id"))
    status: Mapped[str] = mapped_column(String(30), default="pending")
    stage: Mapped[str] = mapped_column(String(30), default="download")
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    owner: Mapped[UUID | None] = mapped_column()
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_category: Mapped[str | None] = mapped_column(String(100))


class Invite(Base):
    __tablename__ = "demo_invites"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class QuotaCounter(Base):
    __tablename__ = "quota_counters"
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    used: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
