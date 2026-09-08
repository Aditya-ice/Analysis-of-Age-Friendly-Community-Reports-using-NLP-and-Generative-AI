from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from elderhelp.config import Settings
from elderhelp.schemas import AnswerFilters, AnswerRequest, ChatTurn
from elderhelp.v2.contracts import QueryPlan
from elderhelp.v2.corpus import activate, index_configuration, reconcile
from elderhelp.v2.ingestion import ingest
from elderhelp.v2.planning import plan
from elderhelp.v2.reranker import OnnxRanker
from elderhelp.v2.retrieval import (
    Candidate,
    active_generation,
    candidates,
    expand,
    fuse,
    retrieve,
    select_evidence,
    still_approved,
)
from test_ingestion_v2 import Embedder, pdf_manifest


class QueryProvider(Embedder):
    def __init__(self):
        super().__init__()
        self.queries = []

    async def embed(self, text, *, reserved=False):
        self.queries.append(text)
        return [1.0] + [0.0] * 767

    async def structured(self, schema, system, payload, **kwargs):
        return QueryPlan(
            original_question=payload["question"],
            standalone_question="What buses did the 2017 report propose?",
            subqueries=["2017 accessible buses"],
            required_parts=["Proposed buses"],
            intent="followup",
        )


async def indexed(database, tmp_path):
    manifest = pdf_manifest(tmp_path)
    manifest.reports[0].publication_date = date(2017, 1, 1)
    await reconcile(database, manifest, apply=True)
    settings = Settings(seed_root=Path("pdfs"))
    result = await ingest(database, Embedder(), settings, tmp_path, manifest)
    generation = UUID(result["generation"])
    await activate(database, generation, index_configuration(settings))
    return settings, generation, manifest


async def test_real_postgres_dense_keyword_filters_and_withdrawal(research_db, tmp_path):
    settings, generation, manifest = await indexed(research_db, tmp_path)
    assert await active_generation(research_db, settings) == generation
    keyword = await candidates(research_db, generation, "accessible buses", AnswerFilters())
    assert keyword
    dense = await candidates(
        research_db, generation, "bus", AnswerFilters(), vector=[1.0] + [0.0] * 767
    )
    assert dense and dense[0].revision_id == keyword[0].revision_id
    for filters in (
        AnswerFilters(community="Wrong"),
        AnswerFilters(year_from=2018),
        AnswerFilters(year_to=2016),
        AnswerFilters(report_ids=[uuid4()]),
    ):
        assert not await candidates(
            research_db, generation, "bus", filters, vector=[1.0] + [0.0] * 767
        )
    manifest.reports[0].status = "withdrawn"
    await reconcile(research_db, manifest, apply=True)
    assert not await candidates(research_db, generation, "accessible buses", AnswerFilters())


async def test_child_survives_parent_expansion_and_resolved_question_reaches_ranker(
    research_db, tmp_path
):
    settings, generation, manifest = await indexed(research_db, tmp_path)
    provider = QueryProvider()
    request = AnswerRequest(
        question="What about their buses?",
        history=[ChatTurn(role="user", content="Discuss the 2017 report.")],
    )
    query_plan = await plan(request, provider)

    class Ranker:
        async def rerank(self, query, items):
            assert query == query_plan.standalone_question
            return items

    result = await retrieve(research_db, provider, Ranker(), settings, query_plan, AnswerFilters())
    assert provider.queries == ["task: question answering | query: 2017 accessible buses"]
    block = result.blocks[-1]
    parent = await expand(research_db, generation, block.candidate, budget=400)
    assert set(block.candidate.span_ids) <= {str(s.id) for s in parent.spans}
    assert await still_approved(research_db, generation, result.blocks)
    manifest.reports[0].status = "withdrawn"
    await reconcile(research_db, manifest, apply=True)
    assert not await still_approved(research_db, generation, result.blocks)


async def test_reranker_operational_failure_keeps_rrf(research_db, tmp_path):
    settings, _, _ = await indexed(research_db, tmp_path)

    class Broken:
        async def rerank(self, *_):
            raise RuntimeError("local model failure")

    query_plan = QueryPlan(
        original_question="buses",
        standalone_question="buses",
        subqueries=["buses"],
        required_parts=["buses"],
    )
    result = await retrieve(
        research_db, QueryProvider(), Broken(), settings, query_plan, AnswerFilters()
    )
    assert result.degraded and result.blocks


async def test_simple_question_bypasses_planner():
    result = await plan(AnswerRequest(question="What transport commitments were reported?"), None)
    assert result.standalone_question == result.original_question


def test_rrf_and_soft_diversity_preserve_comparison_parts():
    report = uuid4()
    items = [
        Candidate(uuid4(), report, uuid4(), uuid4(), "Report", "evidence", [str(uuid4())])
        for _ in range(12)
    ]
    fused = fuse([(0, items[:10]), (1, [items[-1]])])
    result = select_evidence(fused, 2)
    assert items[-1] in result and len(result) == 8
    assert {0, 1} == set.union(*(c.parts for c in result))


def test_real_onnx_model_and_long_candidate_windows():
    path = Path("models/reranker")
    if not path.exists():
        pytest.skip("Run tools/download_reranker.py for the optional real ONNX smoke test")
    ranker = OnnxRanker(path, Path("contracts/reranker-manifest.json"))
    unrelated = ranker.score("What transport was proposed?", "Gardens grow flowers.")
    relevant = ranker.score(
        "What transport was proposed?", "Accessible bus transport was proposed."
    )
    assert relevant > unrelated
    long = "Garden flowers grow in parks. " * 150 + " Accessible bus transport was proposed."
    assert ranker.score("What transport was proposed?", long) > unrelated
