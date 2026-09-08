from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select, text

from elderhelp.models import Report
from elderhelp.services.chunking import report_uuid
from elderhelp.v2.models import (
    ActiveCorpus,
    Embedding,
    Generation,
    GenerationChunk,
    IngestionJob,
    ResearchChunk,
    ResearchPage,
    Revision,
    SourceSpan,
)


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def index_configuration(settings) -> dict:
    return {
        "version": 2,
        "embedding_model": settings.embedding_model,
        "dimensions": 768,
        "query_prefix": "task: question answering | query: ",
        "document_prefix": "title: {title} | text: {content}",
        "normalization": "l2",
        "extraction": "layout-tesseract-v2",
        "chunker": "paragraph-350-50-v2",
        "tokenizer": "cl100k_base",
        "search": "exact",
    }


async def reconcile(database, manifest, *, apply: bool = False) -> dict:
    """Approval/metadata reconciliation is independent from ingestion and activation."""
    desired = {item.slug: item for item in manifest.reports}
    async with database.sessions() as db:
        rows = {row.slug: row for row in (await db.scalars(select(Report))).all()}
        changes = []
        for slug, item in desired.items():
            existing = rows.get(slug)
            fields = {
                name: getattr(item, name)
                for name in (
                    "title",
                    "publisher",
                    "community",
                    "publication_date",
                    "description",
                    "suggested_questions",
                    "status",
                )
            }
            fields["source_url"] = str(item.source_url)
            if not existing or any(getattr(existing, k) != v for k, v in fields.items()):
                changes.append(
                    {"slug": slug, "action": "update" if existing else "add", "status": item.status}
                )
                if apply:
                    if existing is None:
                        existing = Report(
                            id=report_uuid(slug), slug=slug, sha256=item.expected_sha256 or "0" * 64
                        )
                        db.add(existing)
                    for name, value in fields.items():
                        setattr(existing, name, value)
        for slug, row in rows.items():
            if slug not in desired and row.status != "withdrawn":
                changes.append({"slug": slug, "action": "withdraw", "status": "withdrawn"})
                if apply:
                    row.status = "withdrawn"
        if apply:
            await db.commit()
        return {"applied": apply, "changes": changes}


async def stage(database, manifest, configuration: dict) -> UUID:
    async with database.sessions() as db:
        generation = Generation(
            id=uuid4(),
            configuration=configuration,
            fingerprint=fingerprint(configuration),
            manifest_snapshot=manifest.model_dump(mode="json"),
        )
        db.add(generation)
        await db.commit()
        return generation.id


async def validate_generation(database, generation_id: UUID) -> dict:
    async with database.sessions() as db:
        generation = await db.get(Generation, generation_id)
        if not generation:
            raise ValueError("Unknown generation")
        expected = {
            item["slug"]
            for item in generation.manifest_snapshot["reports"]
            if item.get("status", "approved") == "approved"
        }
        rows = (
            await db.execute(
                select(ResearchChunk, Revision, Report, Embedding)
                .join(GenerationChunk, GenerationChunk.chunk_id == ResearchChunk.id)
                .join(Revision, ResearchChunk.revision_id == Revision.id)
                .join(Report, Revision.report_id == Report.id)
                .join(Embedding, ResearchChunk.embedding_key == Embedding.key)
                .where(GenerationChunk.generation_id == generation_id)
            )
        ).all()
        represented = {row[2].slug for row in rows}
        errors = []
        if not expected or not rows or expected != represented:
            errors.append("manifest_coverage")
        for chunk, revision, _report, embedding in rows:
            if embedding.fingerprint != generation.fingerprint:
                errors.append("embedding_configuration")
            if not chunk.span_ids:
                errors.append("missing_spans")
            for span_id in chunk.span_ids:
                span = await db.get(SourceSpan, UUID(span_id))
                page = await db.get(ResearchPage, span.page_id) if span else None
                if (
                    not span
                    or not page
                    or page.revision_id != revision.id
                    or page.text[span.start : span.end] != span.text
                    or not span.searchable
                ):
                    errors.append("invalid_span")
        pending = await db.scalar(
            select(func.count())
            .select_from(IngestionJob)
            .where(IngestionJob.generation_id == generation_id, IngestionJob.status != "completed")
        )
        if pending:
            errors.append("incomplete_jobs")
        result = {
            "passed": not errors,
            "chunks": len(rows),
            "reports": len(represented),
            "errors": sorted(set(errors)),
            "checked_at": datetime.now(UTC).isoformat(),
        }
        generation.validation = result
        generation.status = "validated" if result["passed"] else "staging"
        await db.commit()
        return result


async def activate(database, generation_id: UUID, configuration: dict) -> None:
    result = await validate_generation(database, generation_id)
    if not result["passed"]:
        raise ValueError(f"Generation validation failed: {result['errors']}")
    async with database.sessions() as db, db.begin():
        # Serialize activations even when there is not yet an active row.
        await db.execute(text("SELECT pg_advisory_xact_lock(813248)"))
        generation = await db.get(Generation, generation_id)
        if generation.fingerprint != fingerprint(configuration):
            raise ValueError("Generation is incompatible with the serving configuration")
        active = await db.get(ActiveCorpus, 1, with_for_update=True)
        if active is None:
            db.add(ActiveCorpus(id=1, generation_id=generation_id))
        elif active.generation_id != generation_id:
            active.previous_id = active.generation_id
            active.generation_id = generation_id
            active.updated_at = datetime.now(UTC)
        # Never restore approval states from a historical snapshot.


async def rollback(database, configuration: dict) -> UUID:
    async with database.sessions() as db:
        active = await db.get(ActiveCorpus, 1)
        previous = active.previous_id if active else None
    if not previous:
        raise ValueError("No previous generation")
    await activate(database, previous, configuration)
    return previous


async def status(database) -> dict:
    async with database.sessions() as db:
        active = await db.get(ActiveCorpus, 1)
        rows = (await db.scalars(select(Generation).order_by(Generation.created_at.desc()))).all()
        size = await db.scalar(text("SELECT pg_database_size(current_database())"))
        return {
            "active": str(active.generation_id) if active and active.generation_id else None,
            "previous": str(active.previous_id) if active and active.previous_id else None,
            "database_bytes": size,
            "generations": [
                {
                    "id": str(r.id),
                    "status": r.status,
                    "fingerprint": r.fingerprint,
                    "validation": r.validation,
                }
                for r in rows
            ],
        }
