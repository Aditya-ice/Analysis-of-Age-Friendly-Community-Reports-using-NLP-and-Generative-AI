"""Resumable administrator indexing; serving never downloads, extracts or ingests."""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from sqlalchemy import func, literal_column, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import ProgrammingError

from elderhelp.manifest import ReportManifest
from elderhelp.v2.corpus import fingerprint, index_configuration, stage
from elderhelp.v2.downloads import acquire
from elderhelp.v2.extraction import extract, prepare
from elderhelp.v2.identities import report_uuid
from elderhelp.v2.models import (
    Embedding,
    Generation,
    GenerationChunk,
    IngestionJob,
    ResearchChunk,
    ResearchPage,
    Revision,
    Section,
    SourceSpan,
)
from elderhelp.v2.quota import QuotaExceeded


async def claim_job(database, generation_id, report_id):
    owner, now = uuid4(), datetime.now(UTC)
    job_id = uuid5(generation_id, str(report_id))
    async with database.sessions() as db, db.begin():
        # This short lock serializes lease acquisition across generations for the same report.
        await db.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": report_id.int % (2**63 - 1)}
        )
        running = await db.scalar(
            select(IngestionJob).where(
                IngestionJob.report_id == report_id, IngestionJob.lease_until > now
            )
        )
        if running:
            return None
        job = await db.get(IngestionJob, job_id)
        if job and job.status == "completed":
            return None
        if not job:
            job = IngestionJob(
                id=job_id, report_id=report_id, generation_id=generation_id, attempts=0
            )
            db.add(job)
        job.status, job.owner = "running", owner
        job.lease_until, job.attempts = now + timedelta(minutes=3), job.attempts + 1
        return job_id, owner


async def update_job(database, job_id, owner, **values):
    async with database.sessions() as db, db.begin():
        row = await db.get(IngestionJob, job_id, with_for_update=True)
        if row.owner != owner or not row.lease_until or row.lease_until < datetime.now(UTC):
            raise ValueError("Ingestion lease expired")
        for name, value in values.items():
            setattr(row, name, value)
        row.lease_until = (
            datetime.now(UTC) + timedelta(minutes=3) if row.status == "running" else None
        )


async def keep_lease(database, job_id, owner):
    while True:
        await asyncio.sleep(30)
        await update_job(database, job_id, owner)


async def persist_extraction(database, revision_id, item, pages, config_hash):
    stored_pages, sections, spans, chunks = prepare(revision_id, pages, config_hash)
    if not chunks:
        raise ValueError("No searchable source spans")
    async with database.sessions() as db, db.begin():
        revision = await db.get(Revision, revision_id)
        if not revision:
            db.add(
                Revision(
                    id=revision_id,
                    report_id=report_uuid(item.slug),
                    sha256=item.expected_sha256,
                    metadata_snapshot=item.model_dump(mode="json"),
                )
            )
            await db.flush()
        for model, rows in ((ResearchPage, stored_pages), (Section, sections), (SourceSpan, spans)):
            for row in rows:
                values = {
                    c.name: getattr(row, c.name)
                    for c in model.__table__.columns
                    if getattr(row, c.name, None) is not None
                }
                await db.execute(insert(model).values(**values).on_conflict_do_nothing())
    return chunks


async def load_prepared(database, revision_id, config_hash):
    """Replay chunk assembly from exact persisted spans after extraction has succeeded."""
    # Stable extraction is inexpensive to reconstruct; do not rerun OCR on resumed work.
    from elderhelp.v2.extraction import Block, ExtractedPage

    async with database.sessions() as db:
        pages = (
            await db.scalars(
                select(ResearchPage)
                .where(ResearchPage.revision_id == revision_id)
                .order_by(ResearchPage.page_number)
            )
        ).all()
        pages = [
            p for p in pages if p.id == uuid5(revision_id, f"page:{config_hash}:{p.page_number}")
        ]
        if not pages:
            return None
        rebuilt = []
        for page in pages:
            spans = (
                await db.scalars(
                    select(SourceSpan)
                    .where(SourceSpan.page_id == page.id)
                    .order_by(SourceSpan.start)
                )
            ).all()
            # A block can have multiple spans. Reconstruct the original blocks using bbox + kind.
            blocks = []
            for s in spans:
                if blocks and blocks[-1].bbox == s.bbox and blocks[-1].kind == s.kind:
                    blocks[-1].text += s.text
                    blocks[-1].end = s.end
                    blocks[-1].searchable = blocks[-1].searchable and s.searchable
                else:
                    blocks.append(
                        Block(s.text, s.bbox, s.kind, s.confidence, s.searchable, s.start, s.end)
                    )
            rebuilt.append(
                ExtractedPage(
                    page.page_number,
                    page.page_label,
                    blocks,
                    page.method,
                    page.quality,
                    page.issues,
                    page.text,
                )
            )
        return prepare(revision_id, rebuilt, config_hash)[3]


async def ingest_report(database, provider, settings, root, generation, item, job_id, owner):
    config_hash = generation.fingerprint
    revision_id = uuid5(report_uuid(item.slug), item.expected_sha256)
    prepared = await load_prepared(database, revision_id, config_hash)
    if prepared is None:
        path = await asyncio.to_thread(acquire, item, root, settings.seed_root)
        await update_job(database, job_id, owner, stage="extraction")
        pages = await asyncio.to_thread(extract, path, item.max_pages)
        # Retain failed extraction provenance for administrator review, but never activate it.
        estimate = (
            sum(len(p.text.encode()) * 12 for p in pages) + sum(len(p.blocks) for p in pages) * 5000
        )
        async with database.sessions() as db:
            size = await db.scalar(text("SELECT pg_database_size(current_database())"))
            if size + estimate >= settings.database_budget_bytes * 0.8:
                raise ValueError("Estimated extraction/index storage exceeds 80% database budget")
        prepared = await persist_extraction(database, revision_id, item, pages, config_hash)
        if any("unreadable" in p.issues or "ocr_failed" in p.issues for p in pages):
            raise ValueError("Unreadable pages require review")
    else:
        async with database.sessions() as db:
            pages = (
                await db.scalars(
                    select(ResearchPage).where(ResearchPage.revision_id == revision_id)
                )
            ).all()
            if any("unreadable" in p.issues or "ocr_failed" in p.issues for p in pages):
                raise ValueError("Unreadable pages require review")
    await update_job(database, job_id, owner, stage="embeddings")
    async with database.sessions() as db:
        revision = await db.get(Revision, revision_id)
        title = revision.metadata_snapshot["title"]

    async def index_chunk(chunk):
        key = fingerprint({"config": config_hash, "content": chunk["content"], "title": title})
        async with database.sessions() as db:
            cached = await db.get(Embedding, key)
        if cached is None:
            vector = await provider.embed(f"title: {title} | text: {chunk['content']}")
            async with database.sessions() as db, db.begin():
                await db.execute(
                    insert(Embedding)
                    .values(key=key, fingerprint=config_hash, vector=vector)
                    .on_conflict_do_nothing()
                )
        async with database.sessions() as db, db.begin():
            # Lease ownership is checked in the same transaction as every durable chunk.
            job = await db.get(IngestionJob, job_id, with_for_update=True)
            if job.owner != owner or job.lease_until < datetime.now(UTC):
                raise ValueError("Ingestion lease expired")
            heading = (await db.get(Section, chunk["section_id"])).heading
            weighted = (
                func.setweight(func.to_tsvector("english", title), literal_column("'A'"))
                .op("||")(
                    func.setweight(func.to_tsvector("english", heading), literal_column("'A'"))
                )
                .op("||")(
                    func.setweight(
                        func.to_tsvector("english", chunk["content"]), literal_column("'B'")
                    )
                )
            )
            await db.execute(
                insert(ResearchChunk)
                .values(**chunk, embedding_key=key, search_vector=weighted)
                .on_conflict_do_nothing()
            )
            await db.execute(
                insert(GenerationChunk)
                .values(generation_id=generation.id, chunk_id=chunk["id"])
                .on_conflict_do_nothing()
            )

    # Bounded groups prevent a quota failure from dispatching the entire corpus at once.
    for start in range(0, len(prepared), 2):
        results = await asyncio.gather(
            *(index_chunk(c) for c in prepared[start : start + 2]), return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
    await update_job(
        database, job_id, owner, status="completed", stage="complete", error_category=None
    )


async def ingest(
    database, provider, settings, root: Path, manifest=None, generation_id: UUID | None = None
):
    async with database.sessions() as db:
        size = await db.scalar(text("SELECT pg_database_size(current_database())"))
        if size >= settings.database_budget_bytes * 0.8:
            raise ValueError("Database exceeds 80% storage budget; archive/prune before ingestion")
    if generation_id:
        async with database.sessions() as db:
            generation = await db.get(Generation, generation_id)
            if not generation or generation.status != "staging":
                raise ValueError("Only staging generations can be resumed")
            if generation.fingerprint != fingerprint(index_configuration(settings)):
                raise ValueError("Resume requires the original index configuration")
            manifest = ReportManifest.model_validate(generation.manifest_snapshot)
    else:
        if not manifest or any(not r.expected_sha256 for r in manifest.reports):
            raise ValueError("A checksum-pinned manifest is required")
        # Approval changes are explicit corpus reconciliation, not an ingestion side effect.
        from elderhelp.models import Report

        async with database.sessions() as db:
            known = set((await db.scalars(select(Report.slug))).all())
        if any(r.slug not in known for r in manifest.reports):
            raise ValueError("Review and apply corpus reconciliation before ingestion")
        generation_id = await stage(database, manifest, index_configuration(settings))
        async with database.sessions() as db:
            generation = await db.get(Generation, generation_id)
    results = {
        "generation": str(generation_id),
        "completed": [],
        "failed": [],
        "skipped": [],
        "storage_warning": size >= settings.database_budget_bytes * 0.7,
    }
    for item in manifest.reports:
        if item.status != "approved":
            continue
        lease = await claim_job(database, generation_id, report_uuid(item.slug))
        if not lease:
            results["skipped"].append(item.slug)
            continue
        job_id, owner = lease
        heartbeat = asyncio.create_task(keep_lease(database, job_id, owner))
        try:
            await ingest_report(database, provider, settings, root, generation, item, job_id, owner)
            results["completed"].append(item.slug)
        except asyncio.CancelledError:
            await asyncio.shield(
                update_job(database, job_id, owner, status="paused", error_category="cancelled")
            )
            raise
        except ProgrammingError:
            await update_job(
                database, job_id, owner, status="quarantined", error_category="database_schema"
            )
            raise
        except Exception as exc:
            category = "quota_exhausted" if isinstance(exc, QuotaExceeded) else type(exc).__name__
            await update_job(database, job_id, owner, status="quarantined", error_category=category)
            results["failed"].append({"slug": item.slug, "category": category})
            if isinstance(exc, QuotaExceeded):
                results["retry_after"] = exc.retry_after
                break
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
    return results
