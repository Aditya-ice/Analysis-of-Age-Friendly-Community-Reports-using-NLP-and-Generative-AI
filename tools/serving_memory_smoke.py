"""Serving-image Linux RSS experiment; synthetic inputs, no network/model API calls.

Run inside the final Docker image with --network none --memory 512m. This is
not corpus quality, real workflow latency, or a hosted-service capacity claim.
"""
import asyncio
import importlib.util
import json
import platform
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from elderhelp.config import Settings
from elderhelp.main import create_app
from elderhelp.v2.reranker import OnnxRanker
import elderhelp


async def main():
    assert platform.system() == "Linux", "Measure the Linux serving image"
    for forbidden in ("pymupdf", "torch", "pytesseract"):
        assert importlib.util.find_spec(forbidden) is None, forbidden
    ranker = OnnxRanker(
        Path("/app/models/reranker"),
        Path(elderhelp.__file__).parent / "assets/reranker-manifest.json",
    )
    app = create_app(Settings(google_api_key=None, free_tier_confirmed=False), ranker=ranker)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/")).status_code == 200
            async def workload(index):
                candidates = [SimpleNamespace(
                    id=f"{index}-{i}", title="Synthetic research report", score=0,
                    content=("The historical community report described accessible housing and transport. " * 100),
                ) for i in range(20)]
                result = await ranker.rerank("What did the report say about housing?", candidates)
                assert len(result) == 20
            start = time.monotonic()
            await asyncio.gather(workload(1), workload(2))
            elapsed = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    allowance = 512 * 1024 * 1024
    result = {
        "experiment": "linux-serving-two-synthetic-reranking-workloads",
        "peak_rss_bytes": rss, "allowance_bytes": allowance,
        "limit_fraction": 0.8, "passed": rss < allowance * 0.8,
        "elapsed_seconds": elapsed, "google_calls": 0,
        "quality_or_hosted_latency_result": False,
    }
    print(json.dumps(result))
    assert result["passed"], "Serving RSS exceeded 80% of the free instance allowance"


asyncio.run(main())
