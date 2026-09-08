from elderhelp.config import Settings
from elderhelp.manifest import ReportManifest
from elderhelp.models import Report
from elderhelp.v2.corpus import (
    fingerprint,
    index_configuration,
    reconcile,
    stage,
    validate_generation,
)
from sqlalchemy import select


def manifest(status="approved", title="Report"):
    return ReportManifest.model_validate(
        {
            "reports": [
                {
                    "slug": "test-report",
                    "title": title,
                    "publisher": "City",
                    "community": "City",
                    "source_url": "https://example.org/report.pdf",
                    "local_path": "data/seed/pdfs/report.pdf",
                    "expected_sha256": "a" * 64,
                    "status": status,
                }
            ]
        }
    )


def test_configuration_changes_invalidate_generation():
    a = index_configuration(Settings())
    b = index_configuration(Settings(embedding_model="different"))
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) == fingerprint(dict(reversed(list(a.items()))))


async def test_reconcile_changes_metadata_and_withdraws_without_reembedding(research_db):
    await reconcile(research_db, manifest(), apply=True)
    await reconcile(research_db, manifest("withdrawn", "New title"), apply=True)
    async with research_db.sessions() as db:
        row = await db.scalar(select(Report))
        assert row.status == "withdrawn" and row.title == "New title"
    await reconcile(research_db, manifest(), apply=True)
    preview = await reconcile(research_db, ReportManifest(reports=[]))
    assert preview["changes"][0]["action"] == "withdraw"
    async with research_db.sessions() as db:
        assert (await db.scalar(select(Report))).status == "approved"
    await reconcile(research_db, ReportManifest(reports=[]), apply=True)
    async with research_db.sessions() as db:
        assert (await db.scalar(select(Report))).status == "withdrawn"


async def test_unpopulated_generation_cannot_validate(research_db):
    generation = await stage(research_db, manifest(), index_configuration(Settings()))
    result = await validate_generation(research_db, generation)
    assert result["passed"] is False
    assert "manifest_coverage" in result["errors"]


async def test_duplicate_content_under_different_report_identity(research_db):
    payload = manifest().model_dump()
    second = {**payload["reports"][0], "slug": "second-report"}
    payload["reports"].append(second)
    await reconcile(research_db, ReportManifest.model_validate(payload), apply=True)
    async with research_db.sessions() as db:
        assert len((await db.scalars(select(Report))).all()) == 2


async def populated_generation(database):
    from uuid import uuid4

    from elderhelp.v2.models import (
        Embedding,
        GenerationChunk,
        ResearchChunk,
        ResearchPage,
        Revision,
        Section,
        SourceSpan,
    )
    from sqlalchemy import func

    config = index_configuration(Settings())
    await reconcile(database, manifest(), apply=True)
    generation_id = await stage(database, manifest(), config)
    async with database.sessions() as db:
        report = await db.scalar(select(Report))
        revision = Revision(
            id=uuid4(),
            report_id=report.id,
            sha256="a" * 64,
            metadata_snapshot=manifest().reports[0].model_dump(mode="json"),
        )
        db.add(revision)
        await db.flush()
        section = Section(id=uuid4(), revision_id=revision.id, heading="Transport", ordinal=0)
        page = ResearchPage(
            id=uuid4(),
            revision_id=revision.id,
            page_number=1,
            text="The report proposed accessible buses in 2017.",
            method="digital",
            quality=1,
        )
        db.add_all([section, page])
        await db.flush()
        span = SourceSpan(
            id=uuid4(),
            page_id=page.id,
            section_id=section.id,
            start=0,
            end=len(page.text),
            text=page.text,
            kind="paragraph",
            bbox=[0, 0, 100, 100],
            confidence=1,
        )
        embedding = Embedding(
            key=uuid4().hex, fingerprint=fingerprint(config), vector=[1.0] + [0.0] * 767
        )
        db.add_all([span, embedding])
        await db.flush()
        chunk = ResearchChunk(
            id=uuid4(),
            revision_id=revision.id,
            section_id=section.id,
            span_ids=[str(span.id)],
            content=span.text,
            tokens=10,
            embedding_key=embedding.key,
            search_vector=func.to_tsvector("english", span.text),
        )
        db.add(chunk)
        await db.flush()
        db.add(GenerationChunk(generation_id=generation_id, chunk_id=chunk.id))
        await db.commit()
    return generation_id


async def test_activation_rollback_preserves_current_withdrawals(research_db):
    from elderhelp.v2.corpus import activate, rollback, status

    config = index_configuration(Settings())
    first = await populated_generation(research_db)
    second = await populated_generation(research_db)
    assert (await status(research_db))["active"] is None
    await activate(research_db, first, config)
    await activate(research_db, second, config)
    await reconcile(research_db, manifest("withdrawn"), apply=True)
    assert await rollback(research_db, config) == first
    result = await status(research_db)
    assert result["active"] == str(first) and result["previous"] == str(second)
    async with research_db.sessions() as db:
        assert (await db.scalar(select(Report))).status == "withdrawn"


async def test_invalid_spans_and_incompatible_configuration_cannot_activate(research_db):
    import pytest
    from elderhelp.v2.corpus import activate, status
    from elderhelp.v2.models import SourceSpan

    generation = await populated_generation(research_db)
    with pytest.raises(ValueError, match="incompatible"):
        await activate(
            research_db, generation, index_configuration(Settings(embedding_model="new"))
        )
    assert (await status(research_db))["active"] is None
    async with research_db.sessions() as db:
        (await db.scalar(select(SourceSpan))).text = "Fabricated evidence"
        await db.commit()
    result = await validate_generation(research_db, generation)
    assert "invalid_span" in result["errors"]
    with pytest.raises(ValueError, match="validation failed"):
        await activate(research_db, generation, index_configuration(Settings()))
