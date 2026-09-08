"""Add immutable research corpus, source spans, generations and pilot counters."""

from alembic import op

revision = "20260908_0002"
down_revision = "20260906_0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE reports DROP CONSTRAINT reports_sha256_key")
    op.execute(
        """
CREATE TABLE chunk_embeddings (
    key VARCHAR(64) NOT NULL, 
    fingerprint VARCHAR(64) NOT NULL, 
    vector VECTOR(768) NOT NULL, 
    PRIMARY KEY (key)
)

"""
    )
    op.execute("CREATE INDEX ix_chunk_embeddings_fingerprint ON chunk_embeddings (fingerprint)")
    op.execute(
        """
CREATE TABLE demo_invites (
    id UUID NOT NULL, 
    code_hash VARCHAR(64) NOT NULL, 
    revoked BOOLEAN NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (code_hash)
)

"""
    )
    op.execute(
        """
CREATE TABLE index_generations (
    id UUID NOT NULL, 
    fingerprint VARCHAR(64) NOT NULL, 
    configuration JSONB NOT NULL, 
    manifest_snapshot JSONB NOT NULL, 
    status VARCHAR(24) NOT NULL, 
    validation JSONB NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
)

"""
    )
    op.execute("CREATE INDEX ix_index_generations_fingerprint ON index_generations (fingerprint)")
    op.execute(
        """
CREATE TABLE quota_counters (
    key VARCHAR(200) NOT NULL, 
    used INTEGER NOT NULL, 
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (key)
)

"""
    )
    op.execute("CREATE INDEX ix_quota_counters_expires_at ON quota_counters (expires_at)")
    op.execute(
        """
CREATE TABLE active_corpus (
    id INTEGER NOT NULL, 
    generation_id UUID, 
    previous_id UUID, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(generation_id) REFERENCES index_generations (id), 
    FOREIGN KEY(previous_id) REFERENCES index_generations (id)
)

"""
    )
    op.execute(
        """
CREATE TABLE ingestion_jobs (
    id UUID NOT NULL, 
    generation_id UUID NOT NULL, 
    report_id UUID NOT NULL, 
    status VARCHAR(30) NOT NULL, 
    stage VARCHAR(30) NOT NULL, 
    lease_until TIMESTAMP WITH TIME ZONE, 
    owner UUID, 
    attempts INTEGER NOT NULL, 
    error_category VARCHAR(100), 
    PRIMARY KEY (id), 
    FOREIGN KEY(generation_id) REFERENCES index_generations (id), 
    FOREIGN KEY(report_id) REFERENCES reports (id)
)

"""
    )
    op.execute("CREATE INDEX ix_ingestion_jobs_generation_id ON ingestion_jobs (generation_id)")
    op.execute(
        """
CREATE TABLE report_revisions (
    id UUID NOT NULL, 
    report_id UUID NOT NULL, 
    sha256 VARCHAR(64) NOT NULL, 
    metadata_snapshot JSONB NOT NULL, 
    acquired_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(report_id) REFERENCES reports (id)
)

"""
    )
    op.execute("CREATE INDEX ix_report_revisions_report_id ON report_revisions (report_id)")
    op.execute("CREATE INDEX ix_report_revisions_sha256 ON report_revisions (sha256)")
    op.execute(
        """
CREATE TABLE research_pages (
    id UUID NOT NULL, 
    revision_id UUID NOT NULL, 
    page_number INTEGER NOT NULL, 
    page_label VARCHAR(40), 
    text TEXT NOT NULL, 
    method VARCHAR(40) NOT NULL, 
    quality FLOAT NOT NULL, 
    issues JSONB NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(revision_id) REFERENCES report_revisions (id)
)

"""
    )
    op.execute("CREATE INDEX ix_research_pages_revision_id ON research_pages (revision_id)")
    op.execute(
        """
CREATE TABLE sections (
    id UUID NOT NULL, 
    revision_id UUID NOT NULL, 
    heading TEXT NOT NULL, 
    parent_id UUID, 
    ordinal INTEGER NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(revision_id) REFERENCES report_revisions (id), 
    FOREIGN KEY(parent_id) REFERENCES sections (id)
)

"""
    )
    op.execute("CREATE INDEX ix_sections_revision_id ON sections (revision_id)")
    op.execute(
        """
CREATE TABLE research_chunks (
    id UUID NOT NULL, 
    revision_id UUID NOT NULL, 
    section_id UUID NOT NULL, 
    span_ids JSONB NOT NULL, 
    content TEXT NOT NULL, 
    tokens INTEGER NOT NULL, 
    embedding_key VARCHAR(64) NOT NULL, 
    search_vector TSVECTOR NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(revision_id) REFERENCES report_revisions (id), 
    FOREIGN KEY(section_id) REFERENCES sections (id), 
    FOREIGN KEY(embedding_key) REFERENCES chunk_embeddings (key)
)

"""
    )
    op.execute("CREATE INDEX ix_research_chunks_fts ON research_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_research_chunks_revision_id ON research_chunks (revision_id)")
    op.execute(
        """
CREATE TABLE source_spans (
    id UUID NOT NULL, 
    page_id UUID NOT NULL, 
    section_id UUID NOT NULL, 
    start INTEGER NOT NULL, 
    "end" INTEGER NOT NULL, 
    text TEXT NOT NULL, 
    kind VARCHAR(30) NOT NULL, 
    bbox JSONB NOT NULL, 
    confidence FLOAT NOT NULL, 
    searchable BOOLEAN NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(page_id) REFERENCES research_pages (id), 
    FOREIGN KEY(section_id) REFERENCES sections (id)
)

"""
    )
    op.execute("CREATE INDEX ix_source_spans_page_id ON source_spans (page_id)")
    op.execute("CREATE INDEX ix_source_spans_section_id ON source_spans (section_id)")
    op.execute(
        """
CREATE TABLE generation_chunks (
    generation_id UUID NOT NULL, 
    chunk_id UUID NOT NULL, 
    PRIMARY KEY (generation_id, chunk_id), 
    FOREIGN KEY(generation_id) REFERENCES index_generations (id), 
    FOREIGN KEY(chunk_id) REFERENCES research_chunks (id)
)

"""
    )


def downgrade():
    op.drop_table("generation_chunks")
    op.drop_table("source_spans")
    op.drop_table("research_chunks")
    op.drop_table("sections")
    op.drop_table("research_pages")
    op.drop_table("report_revisions")
    op.drop_table("ingestion_jobs")
    op.drop_table("active_corpus")
    op.drop_table("quota_counters")
    op.drop_table("index_generations")
    op.drop_table("demo_invites")
    op.drop_table("chunk_embeddings")
    # Duplicate report hashes are legitimate; do not restore the old uniqueness bug.
