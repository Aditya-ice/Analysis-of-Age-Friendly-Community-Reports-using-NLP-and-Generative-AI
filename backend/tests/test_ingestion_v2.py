import asyncio
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import pymupdf
import pytest
from elderhelp.config import Settings
from elderhelp.manifest import ReportManifest
from elderhelp.v2.corpus import reconcile, validate_generation
from elderhelp.v2.downloads import acquire, safe_destination
from elderhelp.v2.extraction import Block, extract, prepare
from elderhelp.v2.google import Gemini, ProviderUnavailable, normalize_vector
from elderhelp.v2.ingestion import claim_job, ingest
from elderhelp.v2.quota import QuotaExceeded, daily, reserve


def pdf_manifest(tmp_path, text=None):
    path = tmp_path / "pdfs/report.pdf"
    path.parent.mkdir(exist_ok=True)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            text or "The 2017 report proposed accessible buses. " * 100,
            fontsize=10,
        )
        document.save(path)
    return ReportManifest.model_validate(
        {
            "reports": [
                {
                    "slug": "report",
                    "title": "Transport report",
                    "publisher": "City",
                    "community": "NYC",
                    "source_url": "https://example.org/report.pdf",
                    "local_path": "pdfs/report.pdf",
                    "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "allowed_domains": ["example.org"],
                }
            ]
        }
    )


class Embedder:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    async def embed(self, text):
        self.calls += 1
        assert isinstance(text, str) and text.startswith("title:")
        if self.fail_at and self.calls >= self.fail_at:
            raise QuotaExceeded()
        return [1.0] + [0.0] * 767


def test_embedding_validation_and_explicit_free_gate():
    assert sum(v * v for v in normalize_vector([[2.0] * 768])) == pytest.approx(1)
    for value in ([], [[1] * 768, [1] * 768], [[1]], [[0] * 768], [[float("nan")] * 768]):
        with pytest.raises(ValueError):
            normalize_vector(value)
    with pytest.raises(ProviderUnavailable):
        Gemini(Settings(google_api_key="test", free_tier_confirmed=False), None)


def test_local_files_are_bounded_checksummed_and_confined(tmp_path):
    item = pdf_manifest(tmp_path).reports[0]
    assert acquire(item, tmp_path, Path("pdfs")).is_file()
    with pytest.raises(ValueError, match="outside"):
        acquire(
            item.model_copy(update={"local_path": Path("../secret.pdf")}), tmp_path, Path("pdfs")
        )
    with pytest.raises(ValueError, match="Checksum"):
        acquire(item.model_copy(update={"expected_sha256": "a" * 64}), tmp_path, Path("pdfs"))
    with pytest.raises(ValueError, match="byte limit"):
        acquire(item.model_copy(update={"max_bytes": 10}), tmp_path, Path("pdfs"))
    (tmp_path / item.local_path).write_bytes(b"%PDF-corrupt")
    broken = item.model_copy(
        update={"expected_sha256": hashlib.sha256(b"%PDF-corrupt").hexdigest()}
    )
    with pytest.raises(pymupdf.FileDataError):
        acquire(broken, tmp_path, Path("pdfs"))


def test_redirect_hosts_and_nonpublic_addresses_rejected(monkeypatch):
    with pytest.raises(ValueError):
        safe_destination("http://example.org/a", ["example.org"])
    with pytest.raises(ValueError):
        safe_destination("https://evil.org/a", ["example.org"])
    for ip in ("127.0.0.1", "169.254.169.254", "10.0.0.1", "::1", "fd00::1"):
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, ip=ip, **k: [(0, 0, 0, "", (ip, 443))])
        with pytest.raises(ValueError, match="non-public"):
            safe_destination("https://example.org/a", ["example.org"])


def test_exact_spans_deterministic_chunks_and_selective_ocr(tmp_path):
    manifest = pdf_manifest(tmp_path)
    path = tmp_path / manifest.reports[0].local_path
    pages = extract(path, ocr=lambda _: pytest.fail("Digital text must bypass OCR"))
    revision = uuid4()
    p, _, spans, chunks = prepare(revision, pages, "config")
    assert chunks and all(c["tokens"] <= 350 for c in chunks)
    by_page = {v.id: v for v in p}
    assert all(by_page[s.page_id].text[s.start : s.end] == s.text for s in spans)
    assert [c["id"] for c in chunks] == [c["id"] for c in prepare(revision, pages, "config")[3]]
    assert chunks[0]["span_ids"][-1] in chunks[1]["span_ids"]
    blank = tmp_path / "blank.pdf"
    with pymupdf.open() as d:
        p = d.new_page()
        p.draw_line((0, 0), (400, 400))
        d.save(blank)
    recovered = extract(
        blank,
        ocr=lambda _: [
            Block("OCR text read from a scanned page.", [0, 0, 50, 50], confidence=0.95)
        ],
    )
    assert recovered[0].method == "tesseract"
    unreadable = extract(blank, ocr=lambda _: [])
    assert "unreadable" in unreadable[0].issues


async def test_interrupted_ingestion_resumes_cached_embeddings(research_db, tmp_path):
    manifest = pdf_manifest(tmp_path)
    await reconcile(research_db, manifest, apply=True)
    settings = Settings(seed_root=Path("pdfs"))
    broken = Embedder(fail_at=2)
    result = await ingest(research_db, broken, settings, tmp_path, manifest)
    generation = UUID(result["generation"])
    assert result["failed"][0]["category"] == "quota_exhausted"
    assert not (await validate_generation(research_db, generation))["passed"]
    replacement = Embedder()
    resumed = await ingest(research_db, replacement, settings, tmp_path, generation_id=generation)
    assert resumed["completed"] == ["report"]
    assert (await validate_generation(research_db, generation))["passed"]
    # A second generation reuses successful cached vectors without any provider work.
    no_calls = Embedder(fail_at=1)
    rebuilt = await ingest(research_db, no_calls, settings, tmp_path, manifest)
    assert rebuilt["completed"] == ["report"] and no_calls.calls == 0


async def test_leases_and_quota_races(research_db, tmp_path):
    from elderhelp.services.chunking import report_uuid
    from elderhelp.v2.corpus import index_configuration, stage

    manifest = pdf_manifest(tmp_path)
    await reconcile(research_db, manifest, apply=True)
    a = await stage(research_db, manifest, index_configuration(Settings()))
    b = await stage(research_db, manifest, index_configuration(Settings()))
    leases = await asyncio.gather(
        *(claim_job(research_db, g, report_uuid("report")) for g in (a, b))
    )
    assert sum(v is not None for v in leases) == 1
    results = await asyncio.gather(
        *(reserve(research_db, [daily("test", 1, 2)]) for _ in range(10)), return_exceptions=True
    )
    assert sum(r is None for r in results) == 2
    assert sum(isinstance(r, QuotaExceeded) for r in results) == 8


def test_real_tesseract_on_mixed_pdf(tmp_path):
    import shutil

    if not shutil.which("tesseract"):
        pytest.skip("Local OCR requires the ingestion-only Tesseract executable")
    original = pymupdf.open()
    page = original.new_page()
    page.insert_text((50, 100), "Community transport plans support older adults.", fontsize=18)
    pixels = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
    mixed = tmp_path / "mixed.pdf"
    with pymupdf.open() as d:
        d.insert_pdf(original)
        p = d.new_page()
        p.insert_image(p.rect, stream=pixels)
        d.save(mixed)
    original.close()
    results = extract(mixed)
    assert [p.method for p in results] == ["digital", "tesseract"]
    assert "Community transport" in results[1].text
    assert results[1].quality > 0.75 and not results[1].issues


async def test_google_embedding_dispatch_is_single_input_and_quota_cools_down(
    research_db, monkeypatch
):
    from types import SimpleNamespace

    from elderhelp.v2.google import Gemini
    from elderhelp.v2.quota import google_retry_after

    calls = []

    async def embed_content(**kwargs):
        calls.append(kwargs["contents"])
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] * 768)] * 2)

    model = SimpleNamespace(embed_content=embed_content)
    monkeypatch.setattr(
        "elderhelp.v2.google.genai.Client",
        lambda **kw: SimpleNamespace(aio=SimpleNamespace(models=model)),
    )
    provider = Gemini(
        Settings(_env_file=None, google_api_key="fake", free_tier_confirmed=True), research_db
    )
    with pytest.raises(ValueError, match="exactly one"):
        await provider.embed("title: Report | text: One chunk.")
    assert calls == ["title: Report | text: One chunk."]

    class RateLimited(Exception):
        code = 429

    async def limited(**kwargs):
        raise RateLimited()

    model.embed_content = limited
    with pytest.raises(QuotaExceeded):
        await provider.embed("title: Report | text: Another chunk.")
    assert await google_retry_after(research_db) > 0
