from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from elderhelp.services.extraction import ExtractedPage

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    id: UUID
    report_id: UUID
    page_start: int
    page_end: int
    ordinal: int
    section_heading: str | None
    content: str
    parent_content: str
    token_count: int
    content_hash: str


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def report_uuid(slug: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"https://elderhelp.app/reports/{slug}")


def _heading(page_text: str) -> str | None:
    for line in page_text.splitlines()[:8]:
        candidate = " ".join(line.split())
        if 3 <= len(candidate) <= 100 and (candidate.isupper() or candidate.istitle()):
            return candidate
    return None


def _split_long_sentence(sentence: str, target: int) -> list[str]:
    tokens = sentence.split()
    return [" ".join(tokens[start : start + target]) for start in range(0, len(tokens), target)]


def chunk_pages(
    report_id: UUID,
    pages: list[ExtractedPage],
    *,
    target_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    ordinal = 0
    for page in pages:
        sentences: list[str] = []
        for sentence in SENTENCE_PATTERN.split(page.text):
            sentence = " ".join(sentence.split())
            if not sentence:
                continue
            sentences.extend(
                _split_long_sentence(sentence, target_tokens)
                if token_count(sentence) > target_tokens
                else [sentence]
            )
        current: list[str] = []
        current_tokens = 0
        page_chunks: list[str] = []
        for sentence in sentences:
            size = token_count(sentence)
            if current and current_tokens + size > target_tokens:
                page_chunks.append(" ".join(current))
                overlap: list[str] = []
                overlap_size = 0
                for existing in reversed(current):
                    if overlap_size + token_count(existing) > overlap_tokens:
                        break
                    overlap.insert(0, existing)
                    overlap_size += token_count(existing)
                current = overlap
                current_tokens = overlap_size
            current.append(sentence)
            current_tokens += size
        if current:
            page_chunks.append(" ".join(current))

        heading = _heading(page.text)
        parent = page.text
        if token_count(parent) > 1_600:
            parent = " ".join(parent.split()[:1_600])
        for content in page_chunks:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            chunk_id = uuid5(
                report_id,
                f"page:{page.page_number}:ordinal:{ordinal}:sha256:{content_hash}",
            )
            chunks.append(
                PreparedChunk(
                    id=chunk_id,
                    report_id=report_id,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    ordinal=ordinal,
                    section_heading=heading,
                    content=content,
                    parent_content=parent,
                    token_count=token_count(content),
                    content_hash=content_hash,
                )
            )
            ordinal += 1
    return chunks
