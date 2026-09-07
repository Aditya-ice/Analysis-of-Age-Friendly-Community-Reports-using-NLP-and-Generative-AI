from pathlib import Path

import pytest
from elderhelp.manifest import ManifestReport, ReportManifest, load_manifest
from elderhelp.services.ingestion import sha256_file
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]


def test_seed_manifest_is_valid_and_files_are_pdf():
    manifest = load_manifest(ROOT / "data/reports.yaml")
    assert len(manifest.reports) == 2
    for report in manifest.reports:
        path = ROOT / report.local_path
        assert path.read_bytes()[:5] == b"%PDF-"
        assert len(sha256_file(path)) == 64


def test_manifest_slugs_are_unique():
    manifest = load_manifest(ROOT / "data/reports.yaml")
    slugs = [report.slug for report in manifest.reports]
    assert len(slugs) == len(set(slugs))


def test_duplicate_manifest_slug_is_rejected():
    item = ManifestReport(
        slug="duplicate",
        title="Report",
        publisher="Publisher",
        community="Community",
        source_url="https://example.com/report.pdf",
        local_path=Path("report.pdf"),
    )
    with pytest.raises(ValidationError, match="must be unique"):
        ReportManifest(reports=[item, item])
