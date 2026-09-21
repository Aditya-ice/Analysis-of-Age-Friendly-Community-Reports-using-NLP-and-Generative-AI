"""Explicit, archived pruning of an inactive, validated generation's search data."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

from sqlalchemy import delete, select, text

from elderhelp.v2.models import (
    ActiveCorpus,
    Embedding,
    Generation,
    GenerationChunk,
    IngestionJob,
    ResearchChunk,
)


def save_archive(directory: Path, payload: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    destination = directory / f"{payload['generation_id']}-{digest}.json"
    try:
        with destination.open("xb") as output:
            os.chmod(destination, 0o600)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        if destination.read_bytes() != encoded:
            raise ValueError("Existing generation archive differs") from None
    return str(destination)


async def prune(database, generation_id: UUID, archive_directory: Path, *, apply: bool = False):
    async with database.sessions() as db, db.begin():
        await db.execute(text("SELECT pg_advisory_xact_lock(813248)"))
        # Serialize search-data mutations with ingestion; no provider call is made here.
        if apply:
            await db.execute(
                text(
                    "LOCK TABLE generation_chunks, research_chunks, chunk_embeddings "
                    "IN SHARE ROW EXCLUSIVE MODE"
                )
            )
        active = await db.get(ActiveCorpus, 1)
        if not active or generation_id in (active.generation_id, active.previous_id):
            raise ValueError(
                "An active corpus is required; active and previous generations are kept"
            )
        generation = await db.get(Generation, generation_id, with_for_update=True)
        if not generation or generation.status != "validated":
            raise ValueError("Only an inactive validated generation may be pruned")
        jobs = (
            await db.scalars(
                select(IngestionJob).where(IngestionJob.generation_id == generation_id)
            )
        ).all()
        if any(job.status != "completed" or job.lease_until for job in jobs):
            raise ValueError("Generation still has unfinished ingestion jobs")
        chunks = (
            await db.scalars(
                select(ResearchChunk)
                .join(GenerationChunk)
                .where(GenerationChunk.generation_id == generation_id)
            )
        ).all()
        payload = {
            "generation_id": str(generation_id),
            "configuration": generation.configuration,
            "fingerprint": generation.fingerprint,
            "manifest": generation.manifest_snapshot,
            "validation": generation.validation,
            "chunks": [
                {
                    "id": str(c.id),
                    "revision_id": str(c.revision_id),
                    "span_ids": c.span_ids,
                    "embedding_key": c.embedding_key,
                }
                for c in chunks
            ],
        }
        result = {
            "generation_id": str(generation_id),
            "candidate_chunks": len(chunks),
            "applied": apply,
            "keeps_active_and_previous": True,
        }
        if not apply:
            return result
        result["archive"] = await asyncio.to_thread(save_archive, archive_directory, payload)
        await db.execute(
            delete(GenerationChunk).where(GenerationChunk.generation_id == generation_id)
        )
        if chunks:
            await db.execute(
                delete(ResearchChunk).where(
                    ResearchChunk.id.in_([c.id for c in chunks]),
                    ~ResearchChunk.id.in_(select(GenerationChunk.chunk_id)),
                )
            )
            await db.execute(
                delete(Embedding).where(
                    Embedding.key.in_([c.embedding_key for c in chunks]),
                    ~Embedding.key.in_(select(ResearchChunk.embedding_key)),
                )
            )
        await db.execute(delete(IngestionJob).where(IngestionJob.generation_id == generation_id))
        generation.status = "archived"
        # Immutable revisions, page/span provenance, metadata and approvals are retained.
        return result
