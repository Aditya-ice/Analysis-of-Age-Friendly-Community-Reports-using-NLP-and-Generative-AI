"""Same reviewed cases and resolved plans across four retrieval configurations."""

import json
import time

from elderhelp.v2.google import normalize_vector
from elderhelp.v2.planning import plan
from elderhelp.v2.retrieval import retrieve
from integrity import digest
from metrics import recalled


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class CachedQueries:
    def __init__(self, provider, directory, configuration):
        self.provider, self.directory, self.configuration = provider, directory, configuration

    async def embed(self, text, **kwargs):
        path = self.directory / f"{digest([self.configuration, text])}.json"
        if path.exists():
            return normalize_vector([json.loads(path.read_text())])
        vector = await self.provider.embed(text, **kwargs)
        save(path, vector)
        return vector


async def run(state, provider, payload, case, directory):
    from elderhelp.v2.contracts import QueryPlan

    cache = directory / ".resume" / digest(case)
    plan_file = cache / "plan.json"
    query_plan = (
        QueryPlan.model_validate_json(plan_file.read_text())
        if plan_file.exists()
        else await plan(payload, provider)
    )
    save(plan_file, query_plan.model_dump(mode="json"))
    if query_plan.clarification:
        return {
            "query_plan": query_plan.model_dump(mode="json"),
            "clarification": True,
            "configurations": {},
            "blocked": "query_requires_clarification",
        }
    embedder = CachedQueries(provider, directory / ".query-cache", state.settings.embedding_model)
    result = {"query_plan": query_plan.model_dump(mode="json"), "configurations": {}}
    for name, mode, ranker in [
        ("keyword", "keyword", None),
        ("dense", "dense", None),
        ("hybrid", "hybrid", None),
        ("hybrid_reranker", "hybrid", state.ranker),
    ]:
        path = cache / f"{name}.json"
        if path.exists():
            result["configurations"][name] = json.loads(path.read_text())
            continue
        start = time.perf_counter()
        found = await retrieve(
            state.database,
            embedder,
            ranker,
            state.settings,
            query_plan,
            payload.filters,
            mode=mode,
            reserved=True,
        )
        spans = {str(s.id) for block in found.blocks for s in block.spans}
        record = {
            "generation": str(found.generation),
            "seconds": time.perf_counter() - start,
            "candidates": found.candidates,
            "degraded": found.degraded,
            "evidence": [b.payload() for b in found.blocks],
            "evidence_units": len(case["gold_evidence_units"]),
            "recalled_units": sum(recalled(unit, spans) for unit in case["gold_evidence_units"]),
        }
        save(path, record)
        result["configurations"][name] = record
    return result
