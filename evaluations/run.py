"""Run the versioned evaluation set against a configured ElderHelp API."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def complete_event(client: httpx.AsyncClient, case: dict) -> dict:
    payload = {
        "question": case["question"],
        "history": case.get("history", []),
        "filters": {},
    }
    event_name = None
    async with client.stream("POST", "/v1/answers/stream", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif event_name == "complete" and line.startswith("data: "):
                return json.loads(line.removeprefix("data: "))
    raise RuntimeError(f"No complete event for {case['id']}")


async def run(base_url: str, path: Path) -> int:
    cases = load_cases(path)
    passed = 0
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        for case in cases:
            result = await complete_event(client, case)
            cited_titles = " ".join(item["report_title"].lower() for item in result["citations"])
            report_hit = all(
                slug.replace("-", " ").split()[0] in cited_titles
                or (slug == "age-friendly-nyc-2017" and "nyc" in cited_titles)
                or (slug == "engaging-community-2018" and "engaging" in cited_titles)
                for slug in case["expected_report_slugs"]
            )
            refusal_hit = case["answerable"] or result["status"] == "insufficient_evidence"
            terms_hit = all(
                term.lower() in result["answer_markdown"].lower() for term in case["required_terms"]
            )
            ok = report_hit and refusal_hit and terms_hit
            passed += int(ok)
            print(json.dumps({"id": case["id"], "passed": ok, "status": result["status"]}))
    print(json.dumps({"passed": passed, "total": len(cases), "rate": passed / len(cases)}))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evaluations/cases.jsonl"))
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(run(arguments.base_url, arguments.cases)))
