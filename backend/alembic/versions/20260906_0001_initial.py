"""Create report, page, chunk, and ingestion tables."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "20260906_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(180), nullable=False, unique=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("publisher", sa.String(300), nullable=False),
        sa.Column("community", sa.String(300), nullable=False),
        sa.Column("publication_date", sa.Date()),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("storage_uri", sa.Text()),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("suggested_questions", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_community", "reports", ["community"])
    op.create_index("ix_reports_status", "reports", ["status"])
    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("extraction_method", sa.String(50), nullable=False),
        sa.Column("text_quality", sa.Float(), nullable=False),
        sa.UniqueConstraint("report_id", "page_number", name="uq_pages_report_page"),
    )
    op.create_index("ix_pages_report_id", "pages", ["report_id"])
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section_heading", sa.String(500)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("parent_content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(section_heading, '') || ' ' || content)",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_chunks_report_id", "chunks", ["report_id"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.execute("CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_table(
        "ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("corpus_hash", sa.String(64)),
        sa.Column("embedding_model", sa.String(150), nullable=False),
        sa.Column("generation_model", sa.String(150), nullable=False),
        sa.Column("configuration", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("chunks")
    op.drop_table("pages")
    op.drop_table("reports")
    op.execute("DROP EXTENSION IF EXISTS vector")
