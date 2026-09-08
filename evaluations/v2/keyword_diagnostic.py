"""Development-only search-yield diagnostic. Candidate matches are NOT gold recall."""

import argparse
import asyncio
import json
from pathlib import Path

import asyncpg


async def run(url, output):
    db = await asyncpg.connect(url)
    try:
        if not (await db.fetchval("SELECT current_database()")).endswith("_test"):
            raise ValueError("Use a disposable _test database")
        await db.execute("CREATE TEMP TABLE search_diagnostic (body tsvector)")
        paths = await asyncio.to_thread(lambda: list(Path(".local/extraction").glob("*.json")))
        for path in paths:
            pages = json.loads(path.read_text())
            for page in pages:
                for block in page["blocks"]:
                    if block["searchable"]:
                        await db.execute(
                            "INSERT INTO search_diagnostic VALUES(to_tsvector('english', $1))",
                            block["text"],
                        )
        case_text = await asyncio.to_thread(Path("evaluations/v2/cases.jsonl").read_text)
        cases = [json.loads(line) for line in case_text.splitlines()]
        selected = [
            c
            for c in cases
            if c["split"] == "development"
            and c["expected_answerability"] in ("full", "partial")
            and not c["history"]
        ]
        rows = []
        for case in selected:
            counts = {}
            for mode, terms in [
                ("strict", "websearch_to_tsquery('english', $1)"),
                ("broad", "replace(plainto_tsquery('english', $1)::text, ' & ', ' | ')::tsquery"),
            ]:
                counts[mode] = await db.fetchval(
                    f"SELECT count(*) FROM search_diagnostic WHERE body @@ {terms}",
                    case["question"],
                )
            rows.append(dict(case_id=case["id"], **counts))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                dict(
                    scope="Development-only searchable-block yield; "
                    "not recall or answer correctness",
                    cases=len(rows),
                    zero_strict=sum(r["strict"] == 0 for r in rows),
                    zero_broad=sum(r["broad"] == 0 for r in rows),
                    results=rows,
                ),
                indent=2,
            )
            + "\n"
        )
        print(output)
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluations/v2/results/keyword-yield.json")
    )
    args = parser.parse_args()
    asyncio.run(run(args.database_url, args.output))
