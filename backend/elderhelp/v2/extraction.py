"""Local layout extraction. Normalized text and offsets are persisted together."""

import csv
import io
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid5

import pymupdf
import tiktoken


@dataclass
class Block:
    text: str
    bbox: list[float]
    kind: str = "paragraph"
    confidence: float = 1.0
    searchable: bool = True
    start: int = 0
    end: int = 0


@dataclass
class ExtractedPage:
    number: int
    label: str | None
    blocks: list[Block]
    method: str = "digital"
    quality: float = 1.0
    issues: list[str] = field(default_factory=list)
    text: str = ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_quality(text: str) -> float:
    if not text:
        return 0
    good = sum(c.isalnum() or c.isspace() or c in ".,;:!?()-/%&'\"$" for c in text)
    return good / len(text)


def ocr_blocks(page) -> list[Block]:
    """Tesseract TSV preserves word confidence and converts pixel boxes to PDF coordinates."""
    scale = 3
    with tempfile.TemporaryDirectory(prefix="elderhelp-ocr-") as directory:
        image = Path(directory) / "page.png"
        page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(image)
        result = subprocess.run(
            ["tesseract", str(image), "stdout", "-l", "eng", "--psm", "3", "tsv"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    groups = {}
    for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
        if not (row.get("text") or "").strip() or float(row["conf"]) < 0:
            continue
        key = row["block_num"], row["par_num"]
        groups.setdefault(key, []).append(row)
    blocks = []
    for rows in groups.values():
        text = normalize(" ".join(r["text"] for r in rows))
        confidence = statistics.mean(float(r["conf"]) / 100 for r in rows)
        left = min(int(r["left"]) for r in rows) / scale
        top = min(int(r["top"]) for r in rows) / scale
        right = max(int(r["left"]) + int(r["width"]) for r in rows) / scale
        bottom = max(int(r["top"]) + int(r["height"]) for r in rows) / scale
        blocks.append(
            Block(
                text,
                [left, top, right, bottom],
                confidence=confidence,
                searchable=confidence >= 0.75,
            )
        )
    return blocks


def digital_blocks(page) -> tuple[list[Block], bool]:
    raw = page.get_text("dict")["blocks"]
    sizes = [s["size"] for b in raw if "lines" in b for line in b["lines"] for s in line["spans"]]
    body = statistics.median(sizes) if sizes else 10
    tables = page.find_tables().tables
    blocks = []
    for b in raw:
        if "lines" not in b:
            continue
        box = pymupdf.Rect(b["bbox"])
        if any(box.intersects(pymupdf.Rect(t.bbox)) for t in tables):
            continue
        text = normalize(" ".join(s["text"] for line in b["lines"] for s in line["spans"]))
        if not text:
            continue
        size = max(s["size"] for line in b["lines"] for s in line["spans"])
        kind = "heading" if size >= body * 1.18 and len(text) < 160 else "paragraph"
        if re.match(r"^[•●▪\-]|^\d+[.)] ", text):
            kind = "list"
        blocks.append(Block(text, list(box), kind))
    for table in tables:
        rows = table.extract()
        headers = table.header.names
        certain = bool(headers) and all(headers) and all(len(row) == len(headers) for row in rows)
        for index, row in enumerate(rows):
            if index == 0 and not table.header.external:
                continue
            pairs = [
                f"{normalize(str(h or 'Unknown column'))}: {normalize(str(v or ''))}"
                for h, v in zip(headers, row, strict=False)
            ]
            blocks.append(
                Block(
                    " | ".join(pairs),
                    list(table.rows[index].bbox),
                    "table",
                    0.9 if certain else 0.4,
                    certain,
                )
            )
    # Column-major order if there is a clear central gutter; spanning blocks delimit bands.
    mid = page.rect.width / 2
    left = [b for b in blocks if b.bbox[2] < mid + 5]
    right = [b for b in blocks if b.bbox[0] > mid - 5]
    columns = len(left) >= 3 and len(right) >= 3
    if columns:
        spanning = sorted(
            [b for b in blocks if b not in left and b not in right], key=lambda b: b.bbox[1]
        )
        ordered = []
        remaining = left + right
        for separator in [*spanning, None]:
            y = separator.bbox[1] if separator else float("inf")
            band = [b for b in remaining if b.bbox[1] < y]
            ordered.extend(sorted(band, key=lambda b: (b.bbox[0] > mid - 5, b.bbox[1])))
            remaining = [b for b in remaining if b not in band]
            if separator:
                ordered.append(separator)
        blocks = ordered
    else:
        blocks.sort(key=lambda b: (round(b.bbox[1] / 5), b.bbox[0]))
    figure = (
        len(blocks) >= 25
        and len(page.get_drawings()) >= 60
        and sum(len(b.text) < 100 for b in blocks) / len(blocks) > 0.8
        and sum(any(c.isdigit() for c in b.text) for b in blocks) / len(blocks) > 0.4
    )
    if figure:
        for block in blocks:
            block.kind, block.searchable = "figure", False
    suspicious = figure or any(b.text.count("�") > 2 for b in blocks)
    return blocks, suspicious


def extract(path: Path, max_pages: int = 500, ocr=ocr_blocks) -> list[ExtractedPage]:
    pages = []
    with pymupdf.open(path) as document:
        if document.is_encrypted or not 0 < len(document) <= max_pages:
            raise ValueError("Invalid page count or encrypted document")
        for page in document:
            blocks, suspicious = digital_blocks(page)
            body_blocks = [
                b
                for b in blocks
                if b.bbox[3] > page.rect.height * 0.08 and b.bbox[1] < page.rect.height * 0.92
            ]
            text = " ".join(b.text for b in body_blocks)
            result = ExtractedPage(page.number + 1, page.get_label() or None, blocks)
            figure_review = any(b.kind == "figure" for b in blocks)
            if figure_review:
                result.issues.append("figure_review_required")
            if not text:
                pixels = page.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False).samples
                if max(pixels) - min(pixels) <= 2:
                    result.method = "blank"
                    pages.append(result)
                    continue
            if len(text) < 40 or text_quality(text) < 0.88 or suspicious:
                try:
                    result.blocks = ocr(page)
                    result.method = "tesseract"
                except (OSError, subprocess.SubprocessError):
                    result.blocks = []
                    result.issues.append("ocr_failed")
            if figure_review:
                for block in result.blocks:
                    block.kind, block.searchable = "figure", False
            result.quality = (
                statistics.mean(b.confidence for b in result.blocks) if result.blocks else 0
            )
            if not any(b.searchable for b in result.blocks) and not figure_review:
                result.issues.append("unreadable")
            if any(b.kind == "table" and not b.searchable for b in result.blocks):
                result.issues.append("table_review_required")
            pages.append(result)
        # Repeated margin blocks retain provenance but cannot become retrieval evidence.
        margins = Counter()
        for page, result in zip(document, pages, strict=True):
            margins.update(
                {
                    b.text.casefold()
                    for b in result.blocks
                    if b.bbox[1] < page.rect.height * 0.08 or b.bbox[3] > page.rect.height * 0.92
                }
            )
        for page, result in zip(document, pages, strict=True):
            for b in result.blocks:
                if margins[b.text.casefold()] >= max(2, len(pages) // 3):
                    if b.bbox[1] < page.rect.height * 0.08:
                        b.kind, b.searchable = "header", False
                    elif b.bbox[3] > page.rect.height * 0.92:
                        b.kind, b.searchable = "footer", False
            result.text = ""
            for b in result.blocks:
                b.start = len(result.text)
                result.text += b.text
                b.end = len(result.text)
                result.text += "\n"
    return pages


def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def units(text: str, max_tokens: int = 50):
    """Return character boundaries without corrupting multi-byte characters."""
    enc = tokenizer()
    start = 0
    while start < len(text):
        end = min(len(text), start + max_tokens * 4)
        while len(enc.encode(text[start:end], disallowed_special=())) > max_tokens:
            end = start + max(1, (end - start) * 9 // 10)
        if end < len(text):
            boundary = text.rfind(" ", start + (end - start) // 2, end)
            if boundary > start:
                end = boundary + 1
        yield start, end
        start = end


def prepare(revision: UUID, pages: list[ExtractedPage], config_hash: str):
    """Stable sections/spans/child chunks; larger context is assembled during retrieval."""
    from elderhelp.v2.models import ResearchPage, Section, SourceSpan

    enc = tokenizer()
    sections, spans, stored_pages = [], [], []
    section = Section(id=uuid5(revision, "root"), revision_id=revision, heading="Report", ordinal=0)
    sections.append(section)
    for page in pages:
        page_id = uuid5(revision, f"page:{config_hash}:{page.number}")
        stored_pages.append(
            ResearchPage(
                id=page_id,
                revision_id=revision,
                page_number=page.number,
                page_label=page.label,
                text=page.text,
                method=page.method,
                quality=page.quality,
                issues=page.issues,
            )
        )
        for block in page.blocks:
            if block.kind == "heading" and block.searchable:
                section = Section(
                    id=uuid5(page_id, f"section:{block.start}"),
                    revision_id=revision,
                    heading=block.text,
                    parent_id=sections[0].id,
                    ordinal=len(sections),
                )
                sections.append(section)
            for begin, end in units(block.text):
                # Never split a long table row into misleading independent numerical spans.
                searchable = block.searchable and (
                    block.kind != "table" or begin == 0 and end == len(block.text)
                )
                spans.append(
                    SourceSpan(
                        id=uuid5(page_id, f"span:{block.start + begin}:{block.start + end}"),
                        page_id=page_id,
                        section_id=section.id,
                        start=block.start + begin,
                        end=block.start + end,
                        text=block.text[begin:end],
                        kind=block.kind,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        searchable=searchable,
                    )
                )
    children, pending = [], []

    def emit():
        if not pending:
            return
        content = "\n".join(s.text for s in pending)
        children.append(
            {
                "id": uuid5(revision, config_hash + ":" + ":".join(str(s.id) for s in pending)),
                "revision_id": revision,
                "section_id": pending[0].section_id,
                "span_ids": [str(s.id) for s in pending],
                "content": content,
                "tokens": len(enc.encode(content, disallowed_special=())),
            }
        )

    for span in (s for s in spans if s.searchable):
        tokens = len(enc.encode("\n".join(s.text for s in [*pending, span]), disallowed_special=()))
        if pending and (span.section_id != pending[0].section_id or tokens > 350):
            emit()
            same_section = span.section_id == pending[0].section_id
            last = pending[-1]
            pending = (
                [last]
                if same_section and len(enc.encode(last.text, disallowed_special=())) <= 50
                else []
            )
        pending.append(span)
    emit()
    return stored_pages, sections, spans, children
