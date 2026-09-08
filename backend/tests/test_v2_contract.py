import json
from pathlib import Path
from uuid import uuid4

import pytest
from elderhelp.v2.contracts import CompleteV2, EvidenceSpan, QueryPlan
from pydantic import ValidationError


def test_span_offsets_cannot_lie():
    with pytest.raises(ValidationError):
        EvidenceSpan(
            id=uuid4(),
            revision_id=uuid4(),
            report_id=uuid4(),
            report_title="Report",
            publisher="City",
            source_url="https://example.org",
            page_number=1,
            start=0,
            end=4,
            text="Actual evidence",
        )


def test_plan_limits_decomposition():
    with pytest.raises(ValidationError):
        QueryPlan(original_question="q", standalone_question="q", subqueries=["q"] * 4)


def test_shared_completion_fixture():
    path = Path(__file__).parents[2] / "contracts/fixtures/complete-v2.json"
    result = CompleteV2.model_validate(json.loads(path.read_text()))
    assert result.status == "partial"
    assert result.citations[0].span_id
