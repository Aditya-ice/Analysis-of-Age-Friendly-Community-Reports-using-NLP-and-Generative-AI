"""Curated-only live evaluation. No paid fallback, background runs, or arbitrary prompts."""

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from elderhelp.config import Settings
from elderhelp.database import Database
from elderhelp.schemas import AnswerRequest
from elderhelp.v2.admission import Admission
from elderhelp.v2.answering import answer
from elderhelp.v2.auth import Pilot
from elderhelp.v2.corpus import fingerprint, index_configuration
from elderhelp.v2.google import Gemini
from elderhelp.v2.quota import QuotaExceeded
from elderhelp.v2.reranker import OnnxRanker
from integrity import audit, digest, reviewed
from metrics import measure
from usage import capture

ROOT = Path(__file__).resolve().parents[2]


def cases():
    return [
        json.loads(line)
        for line in Path(__file__).with_name("cases.jsonl").read_text().splitlines()
    ]


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def claims_from_markdown(complete):
    claims = []
    for paragraph in complete["answer_markdown"].split("\n\n"):
        citations = re.findall(r"(?<!\\)\[(S\d+)\]", paragraph)
        if paragraph.strip():
            claims.append({"id": f"D{len(claims) + 1}", "text": paragraph, "citations": citations})
    return claims


async def live(output, split, limit, kind="answers"):
    settings = Settings()
    dataset = json.loads(Path(__file__).with_name("dataset.json").read_text())
    if fingerprint(index_configuration(settings)) != dataset["index_fingerprint"]:
        raise ValueError("Index configuration differs from the reviewed evaluation dataset")
    selected = [c for c in cases() if c["split"] == split and reviewed(c["review"])]
    if errors := audit(cases()):
        raise ValueError(errors)
    if not selected:
        return {"new_completed": 0, "blocked": "human_label_review_pending"}
    database = Database(settings)
    provider = None
    completed = 0
    try:
        provider = Gemini(settings, database)
        state = SimpleNamespace(
            database=database,
            settings=settings,
            provider=provider,
            ranker=OnnxRanker(
                settings.reranker_directory, ROOT / "contracts/reranker-manifest.json"
            ),
            answer_slots=asyncio.Semaphore(2),
        )
        # Persist evaluation identity so restarting cannot reset session limits.
        identity = output / "identity.json"
        if not identity.exists():
            write_atomic(identity, {"invite": str(uuid4()), "session": str(uuid4())})
        ids = json.loads(identity.read_text())
        pilot = Pilot(UUID(ids["invite"]), UUID(ids["session"]))
        for case in selected:
            path = output / f"{case['id']}.json"
            if path.exists():
                if json.loads(path.read_text()).get("case_sha256") != digest(case):
                    raise ValueError(
                        "Existing trace is stale; choose a new versioned output directory"
                    )
                continue
            if case["review"]["status"] != "human_verified":
                print(json.dumps({"case": case["id"], "blocked": "human_label_review_pending"}))
                continue
            payload = AnswerRequest(
                question=case["question"], history=case["history"], filters=case["filters"]
            )
            admission = Admission(state, pilot, payload)
            trace = {
                "case_id": case["id"],
                "case_sha256": digest(case),
                "dataset_sha256": dataset["dataset_sha256"],
                "model": settings.generation_model,
                "index_fingerprint": dataset["index_fingerprint"],
                "question": case["question"],
                "evaluation_mode": "curated_non_sensitive",
            }

            async def progress(stage, message):
                pass

            try:
                bounded = await admission.acquire()
                started = time.monotonic()
                request_id = uuid4()
                with capture(request_id, trace):
                    async with asyncio.timeout(settings.answer_timeout_seconds):
                        if kind == "answers":
                            result = await answer(
                                state, bounded, payload, request_id, progress, trace=trace
                            )
                            trace["complete"] = result.model_dump(mode="json")
                            trace["displayed_claims"] = claims_from_markdown(trace["complete"])
                        else:
                            from ablation import run

                            trace["ablation"] = await run(state, bounded, payload, case, output)
                trace["engine_completion_seconds"] = time.monotonic() - started
                write_atomic(path, trace)
                completed += 1
            except QuotaExceeded as exc:
                print(json.dumps({"paused": "free_quota", "retry_after": exc.retry_after}))
                break
            except Exception as exc:
                # Failed attempts are recorded separately and remain eligible for explicit resume.
                write_atomic(
                    output / "errors" / path.name,
                    {"case_id": case["id"], "error_category": type(exc).__name__},
                )
                break
            finally:
                await admission.close()
            if completed >= limit:
                break
    finally:
        if provider:
            await provider.close()
        await database.close()
    return {"new_completed": completed, "output": str(output)}


def report(traces_dir, annotations_dir, split):
    traces = {
        p.stem: json.loads(p.read_text()) for p in traces_dir.glob("*.json") if p.stem != "identity"
    }
    annotations = {p.stem: json.loads(p.read_text()) for p in annotations_dir.glob("*.json")}
    selected = [c for c in cases() if c["split"] == split]
    result = measure(selected, traces, annotations)
    errors = audit(cases())
    if errors:
        result["blockers"].extend(errors)
        result["release_gate"] = "blocked"
    result["dataset_sha256"] = hashlib.sha256(
        Path(__file__).with_name("cases.jsonl").read_bytes()
    ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["live", "report"])
    parser.add_argument("--kind", choices=["answers", "ablation"], default="answers")
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/evaluation-v2")
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / ".local/evaluation-v2/annotations"
    )
    parser.add_argument("--limit", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    result = (
        asyncio.run(
            live(
                args.output / "ablation" if args.kind == "ablation" else args.output,
                args.split,
                args.limit,
                args.kind,
            )
        )
        if args.command == "live"
        else report(args.output, args.annotations, args.split)
    )
    print(json.dumps(result, indent=2))
    if result.get("release_gate") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
