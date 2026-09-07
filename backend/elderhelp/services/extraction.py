from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: str
    text_quality: float


class PageOCR(Protocol):
    async def extract(self, pdf_page: bytes) -> str: ...


def text_quality(text: str) -> float:
    compact = "".join(text.split())
    if not compact:
        return 0.0
    meaningful = sum(character.isalnum() or character in ".,;:!?'-" for character in compact)
    density = min(len(compact) / 500, 1.0)
    return round((meaningful / len(compact)) * density, 4)


def normalize_text(value: str) -> str:
    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


async def extract_pdf(path: Path, ocr: PageOCR | None = None) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            direct = normalize_text(page.get_text("text", sort=True))
            quality = text_quality(direct)
            method = "pymupdf"
            text = direct
            if (len(direct) < 100 or quality < 0.45) and ocr is not None:
                single_page = pymupdf.open()
                single_page.insert_pdf(document, from_page=index, to_page=index)
                try:
                    ocr_text = normalize_text(await ocr.extract(single_page.tobytes()))
                finally:
                    single_page.close()
                if text_quality(ocr_text) > quality:
                    text = ocr_text
                    quality = text_quality(ocr_text)
                    method = "document-ai-ocr"
            pages.append(
                ExtractedPage(
                    page_number=index + 1,
                    text=text,
                    extraction_method=method,
                    text_quality=quality,
                )
            )
    return pages
