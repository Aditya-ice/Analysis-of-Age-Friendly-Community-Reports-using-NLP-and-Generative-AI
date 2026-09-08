from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, model_validator


class ManifestReport(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    publisher: str
    community: str
    publication_date: date | None = None
    source_url: HttpUrl
    local_path: Path
    description: str | None = None
    suggested_questions: list[str] = Field(default_factory=list, max_length=10)
    status: Literal["approved", "withdrawn"] = "approved"
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    redistribution: Literal["unknown", "permitted", "restricted"] = "unknown"
    allowed_domains: list[str] = Field(default_factory=list)
    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024, le=200 * 1024 * 1024)
    max_pages: int = Field(default=500, ge=1, le=2000)


class ReportManifest(BaseModel):
    version: int = 1
    reports: list[ManifestReport]

    @model_validator(mode="after")
    def unique_reports(self):
        slugs = [report.slug for report in self.reports]
        if len(slugs) != len(set(slugs)):
            raise ValueError("Report slugs must be unique")
        return self


def load_manifest(path: Path) -> ReportManifest:
    data = yaml.safe_load(path.read_text())
    return ReportManifest.model_validate(data)
