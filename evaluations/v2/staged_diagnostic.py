"""Development retrieval diagnostics only: no gold scores, answers, or activation."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path
from uuid import UUID

from ablation import CachedQueries, save
from elderhelp.config import Settings
from elderhelp.database import Database
from elderhelp.schemas import AnswerFilters
from elderhelp.v2.contracts import QueryPlan
from elderhelp.v2.corpus import fingerprint, index_configuration
from elderhelp.v2.google import Gemini
from elderhelp.v2.reranker import OnnxRanker
from elderhelp.v2.retrieval import retrieve
from integrity import audit, digest

ROOT = Path(__file__).resolve().parents[2]


def selected_cases(cases):
    selected = []
    for category in ("direct", "exact", "layout", "unanswerable"):
        selected.extend(
            [
                c
                for c in cases
                if c["split"] == "development" and c["category"] == category and not c["history"]
            ][:3]
        )
    return selected


async def run(generation, output):
    settings = Settings()
    cases = [json.loads(x) for x in (ROOT / "evaluations/v2/cases.jsonl").read_text().splitlines()]
    metadata = json.loads((ROOT / "evaluations/v2/dataset.json").read_text())
    if (
        hashlib.sha256((ROOT / "evaluations/v2/cases.jsonl").read_bytes()).hexdigest()
        != metadata["dataset_sha256"]
    ):
        raise ValueError("Dataset content hash mismatch")
    if audit(cases) or fingerprint(index_configuration(settings)) != metadata["index_fingerprint"]:
        raise ValueError("Dataset or configuration audit failed")
    database = Database(settings)
    provider = Gemini(settings, database)
    try:
        ranker = OnnxRanker(settings.reranker_directory, ROOT / "contracts/reranker-manifest.json")
        embedder = CachedQueries(
            provider, output.parent / ".query-cache", metadata["index_fingerprint"]
        )
        result = {
            "scope": "Development diagnostic; candidate support is NOT human-reviewed gold",
            "quality_gate_passed": False,
            "generation": str(generation),
            "code_commit": (
                await asyncio.to_thread(
                    subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
                )
            ).strip(),
            "dataset_hash": digest(cases),
            "index_fingerprint": metadata["index_fingerprint"],
            "reranker_revision": ranker.version,
            "results": [],
        }
        for case in selected_cases(cases):
            query = QueryPlan(
                original_question=case["question"],
                standalone_question=case["question"],
                subqueries=[case["question"]],
            )
            row = {
                "case_id": case["id"],
                "category": case["category"],
                "case_hash": digest(case),
                "configurations": {},
            }
            for name, mode, reranker in [
                ("keyword", "keyword", None),
                ("dense", "dense", None),
                ("hybrid", "hybrid", None),
                ("hybrid_reranker", "hybrid", ranker),
            ]:
                start = time.perf_counter()
                found = await retrieve(
                    database,
                    embedder,
                    reranker,
                    settings,
                    query,
                    AnswerFilters.model_validate(case["filters"]),
                    mode=mode,
                    evaluation_generation=generation,
                )
                spans = sorted({str(s.id) for block in found.blocks for s in block.spans})
                candidate_ids = {
                    s["span_id"] for s in case["candidate_support"] if s.get("searchable", True)
                }
                row["configurations"][name] = {
                    "seconds": round(time.perf_counter() - start, 4),
                    "candidates": found.candidates,
                    "blocks": len(found.blocks),
                    "degraded": found.degraded,
                    "retrieved_span_ids": spans,
                    "candidate_span_matches_not_gold": len(candidate_ids.intersection(spans)),
                    "candidate_spans_not_gold": len(candidate_ids),
                }
            result["results"].append(row)
            save(output, result)
            print(
                json.dumps(
                    {"completed_case": case["id"], "cases_completed": len(result["results"])}
                ),
                flush=True,
            )
    finally:
        await provider.close()
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=UUID)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/evaluation-v2/staged-diagnostic.json")
    )
    args = parser.parse_args()
    asyncio.run(run(args.generation, args.output))
