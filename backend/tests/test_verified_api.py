import asyncio
import json
from contextlib import aclosing
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from elderhelp.config import Settings
from elderhelp.main import create_app
from elderhelp.v2.answering import deterministic, render
from elderhelp.v2.auth import create_invite
from elderhelp.v2.contracts import Claim, ClaimVerdict, Draft, EvidenceSpan, QueryPlan, Verification
from elderhelp.v2.corpus import reconcile
from elderhelp.v2.models import Invite, QuotaCounter
from sqlalchemy import select
from test_retrieval_v2 import QueryProvider, indexed


class Provider(QueryProvider):
    def __init__(self, *, claim_text=None, supported=True, fail=False, gate=None):
        super().__init__()
        self.claim_text = claim_text or "The 2017 report proposed accessible buses."
        self.supported, self.fail, self.gate = supported, fail, gate
        self.entered = asyncio.Event()
        self.cancelled = False
        self.calls = []

    async def structured(self, schema, system, payload, **kwargs):
        self.calls.append(schema.__name__)
        if schema == Draft:
            span = next(s for b in payload["evidence"] for s in b["spans"] if "2017" in s["text"])
            return Draft(
                answerability="full",
                claims=[Claim(id="C1", text=self.claim_text, span_ids=[UUID(span["id"])])],
            )
        if schema == Verification:
            self.entered.set()
            try:
                if self.gate:
                    await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            if self.fail:
                raise RuntimeError("sensitive provider details")
            return Verification(
                verdicts=[
                    ClaimVerdict(
                        claim_id="C1",
                        supported=self.supported,
                        category="supported" if self.supported else "unsupported",
                    )
                ],
                answers_question=self.supported,
                covered_parts=[0] if self.supported else [],
            )
        return await super().structured(schema, system, payload, **kwargs)


class Ranker:
    async def rerank(self, query, candidates):
        return candidates


async def api(database, tmp_path, provider=None, **settings_options):
    _, _, manifest = await indexed(database, tmp_path)
    settings = Settings(
        _env_file=None, token_secret="x" * 32, seed_root=Path("pdfs"), **settings_options
    )
    app = create_app(settings, database=database, provider=provider or Provider(), ranker=Ranker())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    invite = await create_invite(database, settings)
    response = await client.post("/v2/demo/session", json={"invite_code": invite["invite_code"]})
    assert response.status_code == 200
    client.headers["Authorization"] = "Bearer " + response.json()["token"]
    return app, client, invite, manifest


def parse_events(text):
    events = []
    for frame in text.split("\n\n"):
        if frame.startswith("event: "):
            event, data = frame.split("\ndata: ", 1)
            events.append((event.removeprefix("event: "), json.loads(data)))
    return events


async def test_verified_sse_complete_exactly_matches_display(research_db, tmp_path):
    app, client, _, _ = await api(research_db, tmp_path)
    async with aclosing(client):
        response = await client.post(
            "/v2/answers/stream", json={"question": "What buses were proposed?"}
        )
    assert response.status_code == 200
    events = parse_events(response.text)
    assert [e[0] for e in events] == [
        "start",
        "progress",
        "progress",
        "progress",
        "delta",
        "complete",
    ]
    assert events[-1][1]["status"] == "grounded"
    assert events[-2][1]["text"] == events[-1][1]["answer_markdown"]
    citation = events[-1][1]["citations"][0]
    assert citation["span_id"] and citation["revision_id"] and "2017" in citation["excerpt"]
    assert app.state.answer_slots._value == 2


@pytest.mark.parametrize(
    "text,supported",
    [
        ("The report proposed 5000 free flights.", True),
        ("The report proposed accessible buses.", False),
        ("The report currently provides accessible buses.", True),
    ],
)
async def test_fabricated_or_temporally_invalid_claims_are_not_grounded(
    research_db, tmp_path, text, supported
):
    _, client, _, _ = await api(
        research_db, tmp_path, Provider(claim_text=text, supported=supported)
    )
    async with aclosing(client):
        response = await client.post(
            "/v2/answers/stream", json={"question": "What buses were proposed?"}
        )
    completion = parse_events(response.text)[-1][1]
    assert completion["status"] == "insufficient_evidence"
    assert text not in response.text and completion["citations"] == []


async def test_verification_outage_emits_no_answer_text(research_db, tmp_path):
    _, client, _, _ = await api(research_db, tmp_path, Provider(fail=True))
    async with aclosing(client):
        response = await client.post(
            "/v2/answers/stream", json={"question": "What buses were proposed?"}
        )
    events = parse_events(response.text)
    assert events[-1][0] == "error" and all(e[0] not in ("delta", "complete") for e in events)
    assert "sensitive provider details" not in response.text


async def raw_stream(app, token, sent, disconnected):
    body = json.dumps({"question": "What buses were proposed?"}).encode()
    initial = True

    async def receive():
        nonlocal initial
        if initial:
            initial = False
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v2/answers/stream",
        "raw_path": b"/v2/answers/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", token.encode()), (b"content-type", b"application/json")],
        "server": ("test", 80),
        "client": ("127.0.0.1", 123),
    }
    await app(scope, receive, send)


@pytest.mark.parametrize("cancel", [False, True])
async def test_no_draft_before_verification_and_disconnect_releases_slots(
    research_db, tmp_path, cancel
):
    provider = Provider(gate=asyncio.Event())
    app, client, _, _ = await api(research_db, tmp_path, provider)
    sent, disconnected = [], asyncio.Event()
    task = asyncio.create_task(raw_stream(app, client.headers["Authorization"], sent, disconnected))
    await asyncio.wait_for(provider.entered.wait(), 5)
    emitted = b"".join(m.get("body", b"") for m in sent)
    assert b"event: start" in emitted and b"event: delta" not in emitted
    assert provider.claim_text.encode() not in emitted
    if cancel:
        disconnected.set()
    else:
        provider.gate.set()
    await asyncio.wait_for(task, 5)
    assert app.state.answer_slots._value == 2
    assert provider.cancelled == cancel
    await client.aclose()


async def test_withdrawal_during_verification_discards_answer(research_db, tmp_path):
    provider = Provider(gate=asyncio.Event())
    _, client, _, manifest = await api(research_db, tmp_path, provider)
    task = asyncio.create_task(
        client.post("/v2/answers/stream", json={"question": "What buses were proposed?"})
    )
    await asyncio.wait_for(provider.entered.wait(), 5)
    manifest.reports[0].status = "withdrawn"
    await reconcile(research_db, manifest, apply=True)
    provider.gate.set()
    response = await task
    assert parse_events(response.text)[-1][1]["code"] == "corpus_changed"
    assert "event: delta" not in response.text
    await client.aclose()


async def test_auth_revocation_validation_quota_and_keyword_fallback(research_db, tmp_path):
    _, client, invite, _ = await api(research_db, tmp_path, generation_daily_limit=1)
    async with aclosing(client):
        capabilities = await client.get("/v2/capabilities")
        assert capabilities.json()["generation_available"] is False
        rejected = await client.post(
            "/v2/answers/stream", json={"question": "What buses were proposed?"}
        )
        assert rejected.status_code == 429 and rejected.headers["Retry-After"]
        search = await client.post("/v2/search", json={"question": "accessible buses"})
        assert search.status_code == 200 and search.json()["items"]
        assert "local_path" not in search.text and "storage_uri" not in search.text
        invalid = await client.post("/v2/answers/stream", json={"question": " "})
        assert invalid.status_code == 422
        huge = await client.post("/v2/answers/stream", content=b"x" * 70000)
        assert huge.status_code == 413
        async with research_db.sessions() as db:
            (await db.get(Invite, UUID(invite["id"]))).revoked = True
            await db.commit()
        assert (await client.get("/v2/reports")).status_code == 401
    async with research_db.sessions() as db:
        # Failed multi-counter reservation rolled back even the answer count.
        assert not (
            await db.scalars(select(QuotaCounter).where(QuotaCounter.key.like("answers:%")))
        ).all()


def test_partial_comparison_names_missing_component():
    text = "The report proposed accessible buses."
    span = EvidenceSpan(
        id=uuid4(),
        report_id=uuid4(),
        revision_id=uuid4(),
        report_title="Transport report",
        publisher="City",
        source_url="https://example.org/report",
        page_number=1,
        start=0,
        end=len(text),
        text=text,
    )
    draft = Draft(
        answerability="partial", claims=[Claim(id="C1", text=text, span_ids=[span.id], part=0)]
    )
    review = Verification(
        verdicts=[ClaimVerdict(claim_id="C1", supported=True, category="supported")],
        answers_question=True,
        covered_parts=[0],
    )
    query = QueryPlan(
        original_question="Compare",
        standalone_question="Compare transport",
        subqueries=["NYC transport", "Other transport"],
        required_parts=["NYC", "Other city"],
    )
    result = render(draft, review, {span.id: span}, query, uuid4(), uuid4())
    assert result.status == "partial" and result.missing_parts == ["Other city"]
    assert not deterministic(Claim(id="C2", text=text, span_ids=[uuid4()]), {span.id: span})


async def test_v1_converts_partial_result_before_any_delta(research_db, tmp_path, monkeypatch):
    from elderhelp.v2.contracts import CompleteV2

    async def partial(state, provider, payload, request_id, progress):
        return CompleteV2(
            request_id=request_id, answer_markdown="A partial research result.", status="partial"
        )

    monkeypatch.setattr("elderhelp.v2.routes.answer", partial)
    _, client, _, _ = await api(research_db, tmp_path)
    async with aclosing(client):
        response = await client.post(
            "/v1/answers/stream", json={"question": "What buses were proposed?"}
        )
    events = parse_events(response.text)
    assert [e[0] for e in events] == ["start", "delta", "complete"]
    assert events[-1][1]["status"] == "insufficient_evidence"
    assert "partial research result" not in response.text


async def test_timeout_is_typed_and_model_budget_is_released(research_db, tmp_path):
    provider = Provider(gate=asyncio.Event())
    app, client, _, _ = await api(research_db, tmp_path, provider, answer_timeout_seconds=1)
    async with aclosing(client):
        response = await client.post(
            "/v2/answers/stream", json={"question": "What buses were proposed?"}
        )
    assert parse_events(response.text)[-1][1]["code"] == "timeout"
    assert "event: delta" not in response.text and app.state.answer_slots._value == 2


async def test_burst_rejects_excess_concurrency_before_model_calls(research_db, tmp_path):
    provider = Provider(gate=asyncio.Event())
    app, client, _, _ = await api(research_db, tmp_path, provider)
    requests = [
        asyncio.create_task(
            client.post("/v2/answers/stream", json={"question": "What buses were proposed?"})
        )
        for _ in range(10)
    ]
    await asyncio.wait_for(provider.entered.wait(), 5)
    await asyncio.sleep(0.2)
    provider.gate.set()
    responses = await asyncio.gather(*requests)
    assert sum(r.status_code == 200 for r in responses) == 2
    assert sum(r.status_code == 429 for r in responses) == 8
    assert app.state.answer_slots._value == 2
    await client.aclose()


def test_uncited_claim_and_missing_verdict_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Draft.model_validate(
            {
                "answerability": "full",
                "claims": [{"id": "C1", "text": "Uncited claim", "span_ids": []}],
            }
        )
    claim = Claim(id="C1", text="The report proposed buses.", span_ids=[uuid4()])
    query = QueryPlan(
        original_question="q", standalone_question="q", subqueries=["q"], required_parts=["q"]
    )
    with pytest.raises(ValueError, match="verdict"):
        render(
            Draft(answerability="full", claims=[claim]),
            Verification(verdicts=[], answers_question=True),
            {},
            query,
            uuid4(),
            uuid4(),
        )
