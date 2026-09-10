import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from elderhelp.config import Settings
from elderhelp.v2.admission import BudgetProvider
from elderhelp.v2.google import Gemini
from elderhelp.v2.models import QuotaCounter
from elderhelp.v2.quota import daily, pace

EVALUATIONS = Path(__file__).resolve().parents[2] / "evaluations/v2"
sys.path.insert(0, str(EVALUATIONS))
from readiness import local_checks, preflight  # noqa: E402
from review_batches import batches, generate  # noqa: E402


def test_dotenv_dimension_and_secret_redaction(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ELDERHELP_EMBEDDING_DIMENSIONS=768\n")
    assert Settings(_env_file=env).embedding_dimensions == 768
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_dimensions="1536")
    with pytest.raises(ValueError) as error:
        Settings(
            _env_file=None,
            environment="pilot",
            database_tls="disable",
            google_api_key="sensitive-test-value",
        )
    assert "sensitive-test-value" not in str(error.value)


def test_review_batches_preserve_groups_and_exclude_heldout(tmp_path):
    source = EVALUATIONS / "cases.jsonl"
    before = source.read_bytes()
    cases = [json.loads(line) for line in before.splitlines()]
    grouped = batches(cases)
    owners = {}
    for i, group in enumerate(grouped):
        for case in group:
            assert case["split"] == "development"
            for key in [case["fact_group"]] + [s["span_id"] for s in case["candidate_support"]]:
                assert owners.setdefault(key, i) == i
    assert sum(map(len, grouped)) == 80
    result = generate(tmp_path)
    index = Path(result["index"])
    manifest = json.loads((index.parent / "batches.json").read_text())
    exported = {c["id"] for b in manifest for c in b["cases"]}
    assert exported == {c["id"] for c in cases if c["split"] == "development"}
    assert result["heldout_cases_exported"] == 0
    assert source.read_bytes() == before
    assert "human_verified" not in (index.parent / manifest[0]["file"]).read_text()


def test_shared_spans_keep_different_fact_groups_in_same_batch():
    cases = [
        dict(
            id=str(i),
            split="development",
            fact_group=str(i),
            candidate_support=[{"span_id": "shared" if i in (0, 2) else "other"}],
        )
        for i in range(3)
    ]
    result = batches(cases, size=1)
    assert {c["id"] for c in result[0]} == {"0", "2"}


async def test_readiness_does_not_expose_key_or_require_model_calls():
    settings = Settings(
        _env_file=None,
        google_api_key="sensitive-test-value",
        free_tier_confirmed=False,
        token_secret=None,
    )
    result = await preflight(settings)
    assert "sensitive-test-value" not in json.dumps(result)
    assert result["checks"]["google_key_present"]
    assert result["model_calls_made"] == 0
    assert not result["ingestion_ready"] and not result["evaluation_ready"]
    assert not result["release_quality_certified"]
    checks, _ = local_checks(settings)
    assert checks["seed_assets_valid"] and checks["dataset_integrity_valid"]


async def test_pacer_serializes_concurrent_dispatches(research_db):
    async def dispatch():
        await pace(research_db, "embedding", 600)
        return asyncio.get_running_loop().time()

    times = sorted(await asyncio.gather(dispatch(), dispatch(), dispatch()))
    assert min(b - a for a, b in zip(times, times[1:], strict=False)) >= 0.10


async def test_cancellation_before_dispatch_preserves_workflow_budget():
    entered = asyncio.Event()

    class Provider:
        async def before_dispatch(self, kind):
            entered.set()
            await asyncio.Event().wait()

    provider = BudgetProvider(Provider(), 2, 1)
    task = asyncio.create_task(provider.structured(None, "", {}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.generation_left == 2


async def test_cancelled_direct_embedding_refunds_unspent_daily_reservation(research_db):
    entered = asyncio.Event()
    provider = object.__new__(Gemini)
    provider.database = research_db
    provider.settings = Settings(_env_file=None, embedding_daily_limit=20)

    async def wait_for_slot(kind):
        entered.set()
        await asyncio.Event().wait()

    provider.before_dispatch = wait_for_slot
    task = asyncio.create_task(provider.prepare("embedding", reserved=False, paced=False))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with research_db.sessions() as db:
        counter = await db.get(QuotaCounter, daily("embedding", 1, 20)[0])
        assert counter.used == 0


def test_google_daily_quota_resets_at_pacific_midnight(monkeypatch):
    from elderhelp.v2 import quota

    class Clock(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return datetime(2026, 9, 9, 1, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(quota, "datetime", Clock)
    generation = quota.daily("generation", 1, 20)
    assert generation[0] == "generation:2026-09-08"
    assert generation[3].astimezone(UTC) == datetime(2026, 9, 9, 7, tzinfo=UTC)
    assert quota.daily("answers", 1, 20)[0] == "answers:2026-09-09"
