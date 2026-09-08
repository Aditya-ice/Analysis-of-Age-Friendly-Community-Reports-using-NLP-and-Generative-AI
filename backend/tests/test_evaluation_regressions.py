from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest
from elderhelp.schemas import AnswerFilters
from elderhelp.v2.catalog import list_reports
from elderhelp.v2.corpus import reconcile
from elderhelp.v2.dates import publication_text
from elderhelp.v2.extraction import Block, extract, prepare
from test_retrieval_v2 import indexed


def test_scanned_body_with_readable_footer_still_uses_ocr(tmp_path):
    path = tmp_path / "roster.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_rect((80, 100, 500, 650), color=(0, 0, 0))
        page.insert_text((50, 810), "Readable report footer is not the scanned commission roster.")
        document.save(path)
    pages = extract(
        path, ocr=lambda _: [Block("Commission member Jane Smith.", [80, 110, 500, 200])]
    )
    assert pages[0].method == "tesseract"
    assert "Jane Smith" in pages[0].text


def test_uncertain_numerical_figure_never_becomes_searchable(tmp_path, monkeypatch):
    path = tmp_path / "figure.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_rect((80, 100, 500, 650))
        document.save(path)
    monkeypatch.setattr(
        "elderhelp.v2.extraction.digital_blocks",
        lambda _: ([Block("2017 65 80 5", [80, 110, 500, 200], "figure", searchable=False)], True),
    )
    pages = extract(path, ocr=lambda _: [Block("65 percent 80 years", [80, 110, 500, 200])])
    assert pages[0].issues == ["figure_review_required"]
    assert not any(b.searchable for b in pages[0].blocks)
    assert not prepare(uuid4(), pages, "test")[3]


def test_seed_commission_roster_ocr_regression():
    import shutil

    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is an administrator-only dependency")
    path = Path("data/seed/pdfs/AgeFriendlyNYC2017.pdf")
    from elderhelp.v2.extraction import ocr_blocks

    with pymupdf.open(path) as document:
        blocks = ocr_blocks(document[80])
    assert any("commission" in b.text.lower() for b in blocks)
    assert any(b.searchable for b in blocks)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("year", "2017"), ("month", "2017-01"), ("day", "2017-01-01"), ("unknown", None)],
)
def test_dates_do_not_invent_precision(precision, expected):
    assert publication_text("2017-01-01", precision) == expected


async def test_catalog_metadata_change_updates_date_precision_without_reindex(
    research_db, tmp_path
):
    settings, generation, manifest = await indexed(research_db, tmp_path)
    manifest.reports[0].publication_precision = "year"
    await reconcile(research_db, manifest, apply=True)
    result = await list_reports(research_db, settings, AnswerFilters(), 0, 20)
    assert result.items[0].publication_date == "2017"
    assert result.items[0].publication_precision == "year"
    from elderhelp.v2.models import ActiveCorpus

    async with research_db.sessions() as db:
        assert (await db.get(ActiveCorpus, 1)).generation_id == generation
