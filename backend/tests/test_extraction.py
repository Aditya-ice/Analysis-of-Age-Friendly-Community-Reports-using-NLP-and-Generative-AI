from pathlib import Path

import pymupdf
import pytest
from elderhelp.services.extraction import extract_pdf


class FakeOCR:
    calls = 0

    async def extract(self, pdf_page: bytes) -> str:
        self.calls += 1
        assert pdf_page.startswith(b"%PDF-")
        return "Readable OCR output for an image-only page. " * 20


@pytest.mark.asyncio
async def test_low_text_page_uses_page_level_ocr(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    ocr = FakeOCR()
    pages = await extract_pdf(path, ocr=ocr)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].extraction_method == "document-ai-ocr"
    assert ocr.calls == 1


@pytest.mark.asyncio
async def test_seed_pdf_page_boundaries_are_preserved():
    root = Path(__file__).parents[2]
    pages = await extract_pdf(root / "data/seed/pdfs/AgeFriendlyNYC2017.pdf")
    assert len(pages) == 92
    assert pages[0].page_number == 1
    assert pages[-1].page_number == 92
    assert "Age-friendly NYC" in pages[0].text
