import json
from datetime import date
from uuid import uuid4

import pytest
from elderhelp.schemas import AnswerRequest
from elderhelp.services.answering import INSUFFICIENT_ANSWER, AnswerService, validated_citations
from elderhelp.services.retrieval import PassthroughRanker, RetrievedChunk


def source():
    return RetrievedChunk(
        chunk_id=uuid4(),
        report_id=uuid4(),
        report_title="Housing Plan",
        publisher="City",
        source_url="https://example.com/plan.pdf",
        publication_date=date(2024, 1, 1),
        page_start=12,
        page_end=12,
        content="The city will preserve affordable homes.",
        parent_content="Housing. The city will preserve affordable homes.",
    )


class Planner:
    async def plan(self, request):
        return [request.question]


class Embedder:
    async def embed_query(self, query):
        return [0.1, 0.2]


class Retriever:
    def __init__(self, results):
        self.results = results

    async def retrieve(self, *args, **kwargs):
        return self.results


class Generator:
    def __init__(self, pieces):
        self.pieces = pieces

    async def stream(self, prompt):
        for piece in self.pieces:
            yield piece


class FailingRetriever:
    async def retrieve(self, *args, **kwargs):
        raise RuntimeError("provider details must not leak")


def data(event):
    return json.loads(event.split("data: ", 1)[1])


@pytest.mark.asyncio
async def test_streams_grounded_answer_and_citation():
    service = AnswerService(
        Planner(),
        Embedder(),
        Retriever([source()]),
        PassthroughRanker(),
        Generator(["Affordable ", "homes [S1]."]),
    )
    events = [
        event async for event in service.events(AnswerRequest(question="What about housing?"))
    ]
    assert [event.splitlines()[0] for event in events] == [
        "event: start",
        "event: delta",
        "event: delta",
        "event: complete",
    ]
    complete = data(events[-1])
    assert complete["status"] == "grounded"
    assert complete["citations"][0]["page_number"] == 12


@pytest.mark.asyncio
async def test_invalid_or_missing_citation_becomes_insufficient():
    service = AnswerService(
        Planner(),
        Embedder(),
        Retriever([source()]),
        PassthroughRanker(),
        Generator(["Unsupported answer."]),
    )
    events = [event async for event in service.events(AnswerRequest(question="Question"))]
    assert data(events[-1])["answer_markdown"] == INSUFFICIENT_ANSWER
    assert data(events[-1])["status"] == "insufficient_evidence"


def test_unknown_citation_is_rejected():
    citations, valid = validated_citations("Claim [S2].", [source()])
    assert not valid and citations == []


@pytest.mark.asyncio
async def test_provider_failure_becomes_safe_sse_error():
    service = AnswerService(
        Planner(), Embedder(), FailingRetriever(), PassthroughRanker(), Generator([])
    )
    events = [event async for event in service.events(AnswerRequest(question="Question"))]
    assert events[-1].startswith("event: error")
    assert "provider details" not in events[-1]
