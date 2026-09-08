from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from elderhelp.config import Settings
from elderhelp.database import Database
from elderhelp.manifest import ManifestReport, load_manifest
from elderhelp.models import Chunk, IngestionRun, Page, Report
from elderhelp.providers.document_ai import DocumentAIPageOCR
from elderhelp.providers.google import GoogleClients, GoogleEmbeddingProvider
from elderhelp.services.chunking import chunk_pages, report_uuid
from elderhelp.services.extraction import extract_pdf
from google.cloud import storage
from sqlalchemy import delete, func, select


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def ensure_pdf(item: ManifestReport, repository_root: Path) -> Path:
    path = repository_root / item.local_path
    if path.exists():
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"{path} is not a PDF")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
            temporary = handle.name
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                async with client.stream("GET", str(item.source_url)) as response:
                    response.raise_for_status()
                    async for block in response.aiter_bytes():
                        handle.write(block)
        temporary_path = Path(temporary)
        header = await asyncio.to_thread(lambda: temporary_path.read_bytes()[:5])
        if header != b"%PDF-":
            raise ValueError(f"Publisher response for {item.slug} is not a PDF")
        await asyncio.to_thread(os.replace, temporary_path, path)
        temporary = None
        return path
    finally:
        if temporary:
            temporary_path = Path(temporary)
            if await asyncio.to_thread(temporary_path.exists):
                await asyncio.to_thread(temporary_path.unlink)


async def upload_private_pdf(path: Path, bucket_name: str | None, object_name: str) -> str | None:
    if not bucket_name:
        return None

    def upload() -> None:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        if not blob.exists():
            blob.upload_from_filename(path, content_type="application/pdf")

    await asyncio.to_thread(upload)
    return f"gs://{bucket_name}/{object_name}"


async def ingest(settings: Settings, repository_root: Path) -> dict[str, int]:
    manifest = load_manifest(repository_root / settings.report_manifest)
    if not settings.google_cloud_project:
        raise ValueError("ELDERHELP_GOOGLE_CLOUD_PROJECT is required for embeddings")
    database = Database(settings)
    embedder = GoogleEmbeddingProvider(GoogleClients(settings))
    ocr = (
        DocumentAIPageOCR(settings.document_ai_processor)
        if settings.document_ai_processor
        else None
    )
    counts = {"indexed": 0, "unchanged": 0, "failed": 0}
    corpus_hash = hashlib.sha256()
    async with database.sessions() as session:
        run = IngestionRun(
            embedding_model=settings.embedding_model,
            generation_model=settings.generation_model,
            configuration={"dimensions": settings.embedding_dimensions},
        )
        session.add(run)
        await session.commit()
        try:
            for item in manifest.reports:
                pdf_path = await ensure_pdf(item, repository_root)
                digest = sha256_file(pdf_path)
                corpus_hash.update(f"{item.slug}:{digest}\n".encode())
                report_id = report_uuid(item.slug)
                existing = await session.scalar(select(Report).where(Report.id == report_id))
                existing_chunks = (
                    await session.scalar(
                        select(func.count()).select_from(Chunk).where(Chunk.report_id == report_id)
                    )
                    if existing
                    else 0
                )
                if existing and existing.sha256 == digest and existing_chunks:
                    counts["unchanged"] += 1
                    continue

                pages = await extract_pdf(pdf_path, ocr=ocr)
                prepared = chunk_pages(report_id, pages)
                if not prepared:
                    raise ValueError(f"No searchable content extracted for {item.slug}")
                embeddings: list[list[float]] = []
                for start in range(0, len(prepared), 16):
                    batch = prepared[start : start + 16]
                    embeddings.extend(
                        await embedder.embed_documents(
                            [(item.title, chunk.content) for chunk in batch]
                        )
                    )
                storage_uri = await upload_private_pdf(
                    pdf_path, settings.storage_bucket, f"reports/{digest}.pdf"
                )
                if existing:
                    await session.execute(delete(Chunk).where(Chunk.report_id == report_id))
                    await session.execute(delete(Page).where(Page.report_id == report_id))
                    report = existing
                else:
                    report = Report(id=report_id, slug=item.slug, sha256=digest)
                    session.add(report)
                report.title = item.title
                report.publisher = item.publisher
                report.community = item.community
                report.publication_date = item.publication_date
                report.source_url = str(item.source_url)
                report.description = item.description
                report.storage_uri = storage_uri
                report.sha256 = digest
                report.status = item.status
                report.suggested_questions = item.suggested_questions
                session.add_all(
                    [
                        Page(
                            report_id=report_id,
                            page_number=page.page_number,
                            text=page.text,
                            extraction_method=page.extraction_method,
                            text_quality=page.text_quality,
                        )
                        for page in pages
                    ]
                )
                session.add_all(
                    [
                        Chunk(
                            id=chunk.id,
                            report_id=report_id,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            ordinal=chunk.ordinal,
                            section_heading=chunk.section_heading,
                            content=chunk.content,
                            parent_content=chunk.parent_content,
                            token_count=chunk.token_count,
                            content_hash=chunk.content_hash,
                            embedding=embedding,
                        )
                        for chunk, embedding in zip(prepared, embeddings, strict=True)
                    ]
                )
                await session.commit()
                counts["indexed"] += 1
            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            run.corpus_hash = corpus_hash.hexdigest()
            await session.commit()
        except Exception as error:
            await session.rollback()
            run = await session.get(IngestionRun, run.id)
            run.status = "failed"
            run.completed_at = datetime.now(UTC)
            run.error = type(error).__name__
            await session.commit()
            counts["failed"] += 1
            raise
        finally:
            await database.close()
    return counts
