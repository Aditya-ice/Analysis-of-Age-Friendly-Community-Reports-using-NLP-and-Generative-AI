from elderhelp.services.chunking import chunk_pages, report_uuid
from elderhelp.services.extraction import ExtractedPage, text_quality


def test_chunks_are_page_aware_and_deterministic():
    pages = [
        ExtractedPage(
            page_number=7,
            text=(
                "HOUSING\n\nOlder adults need affordable homes. "
                "Safe housing supports aging in place."
            ),
            extraction_method="pymupdf",
            text_quality=1.0,
        )
    ]
    identifier = report_uuid("sample-report")
    first = chunk_pages(identifier, pages, target_tokens=12, overlap_tokens=3)
    second = chunk_pages(identifier, pages, target_tokens=12, overlap_tokens=3)
    assert first == second
    assert all(chunk.page_start == 7 and chunk.page_end == 7 for chunk in first)
    assert first[0].section_heading == "HOUSING"
    assert len({chunk.id for chunk in first}) == len(first)


def test_text_quality_rejects_empty_text():
    assert text_quality("   \n") == 0
    assert text_quality("This is a readable sentence. " * 30) > 0.8
