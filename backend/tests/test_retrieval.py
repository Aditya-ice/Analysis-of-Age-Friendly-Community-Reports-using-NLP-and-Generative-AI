from datetime import date
from uuid import uuid4

from elderhelp.services.retrieval import RetrievedChunk, diversify, reciprocal_rank_fusion


def chunk(report_id=None):
    return RetrievedChunk(
        chunk_id=uuid4(),
        report_id=report_id or uuid4(),
        report_title="Report",
        publisher="Publisher",
        source_url="https://example.com/report.pdf",
        publication_date=date(2020, 1, 1),
        page_start=1,
        page_end=1,
        content="Evidence",
        parent_content="Evidence in context",
    )


def test_rrf_rewards_results_present_in_both_lists():
    shared, dense_only, keyword_only = chunk(), chunk(), chunk()
    fused = reciprocal_rank_fusion([[dense_only, shared], [keyword_only, shared]])
    assert fused[0].chunk_id == shared.chunk_id
    assert len(fused) == 3


def test_diversity_limits_each_report():
    first_report, second_report = uuid4(), uuid4()
    candidates = [chunk(first_report) for _ in range(4)] + [chunk(second_report) for _ in range(3)]
    selected = diversify(candidates, limit=5, max_per_report=3)
    assert len(selected) == 5
    assert sum(item.report_id == first_report for item in selected) == 3
