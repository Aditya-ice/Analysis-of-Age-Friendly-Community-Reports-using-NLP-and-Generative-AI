from datetime import date
from pathlib import Path

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
    status: str = "approved"


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
