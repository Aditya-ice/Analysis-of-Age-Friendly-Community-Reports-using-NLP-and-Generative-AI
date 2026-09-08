"""Synthetic PostgreSQL capacity experiment. Never a corpus-quality or hosted-load result."""

import argparse
import asyncio
import json
import resource
import time
from pathlib import Path

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector


async def run(url, output):
    db = await asyncpg.connect(url)
    try:
        if await db.fetchval("SELECT current_database()") != "elderhelp_capacity_test":
            raise ValueError("Only the disposable elderhelp_capacity_test database is permitted")
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(db)
        await db.execute("DROP TABLE IF EXISTS synthetic_capacity")
        await db.execute("""CREATE TABLE synthetic_capacity (
            id integer PRIMARY KEY, report_id integer NOT NULL,
            approved boolean NOT NULL, content text NOT NULL,
            vector vector(768) NOT NULL, keywords tsvector GENERATED ALWAYS AS
            (to_tsvector('english', content)) STORED)""")
        await db.execute("CREATE INDEX synthetic_keyword ON synthetic_capacity USING gin(keywords)")
        rng = np.random.default_rng(20260908)
        queries = rng.normal(size=(5, 768)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        results = {
            "data": "SYNTHETIC random unit vectors; no Google calls or real quality labels",
            "platform": "local PostgreSQL; no hosted uptime or latency claim",
            "sizes": [],
        }
        count = 0
        for target in (1000, 10000, 50000):
            while count < target:
                batch = min(250, target - count)
                vectors = rng.normal(size=(batch, 768)).astype(np.float32)
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                await db.copy_records_to_table(
                    "synthetic_capacity",
                    columns=["id", "report_id", "approved", "content", "vector"],
                    records=[
                        (
                            i,
                            i % 100,
                            i % 10 != 0,
                            (
                                f"SYNTHETIC report {i % 100} accessible transport "
                                "housing civic participation " * 30
                            ),
                            v,
                        )
                        for i, v in zip(range(count, count + batch), vectors, strict=True)
                    ],
                )
                count += batch
            await db.execute("ANALYZE synthetic_capacity")
            record = {"chunks": target, "exact": {}, "keyword_ms": []}
            exact_ids = {}
            for filtered in (False, True):
                clause = "approved AND report_id=3" if filtered else "approved"
                key = "report_filter" if filtered else "approval_filter"
                sql = (
                    f"SELECT id FROM synthetic_capacity WHERE {clause} "
                    "ORDER BY vector <=> $1 LIMIT 30"
                )
                timings = []
                for i, vector in enumerate(queries):
                    start = time.perf_counter()
                    rows = await db.fetch(sql, vector)
                    timings.append(1000 * (time.perf_counter() - start))
                    exact_ids[key, i] = {r["id"] for r in rows}
                record["exact"][key] = {"milliseconds": timings}
            for _ in range(5):
                start = time.perf_counter()
                await db.fetch("""SELECT id FROM synthetic_capacity WHERE approved
                    AND keywords @@ websearch_to_tsquery('english', 'accessible transport')
                    ORDER BY ts_rank_cd(keywords,
                        websearch_to_tsquery('english', 'accessible transport')) DESC
                    LIMIT 30""")
                record["keyword_ms"].append(1000 * (time.perf_counter() - start))
            if target >= 10000:
                start = time.perf_counter()
                await db.execute(
                    "CREATE INDEX synthetic_hnsw ON synthetic_capacity "
                    "USING hnsw(vector vector_cosine_ops)"
                )
                record["hnsw_build_seconds"] = time.perf_counter() - start
                record["hnsw"] = {}
                for filtered in (False, True):
                    clause = "approved AND report_id=3" if filtered else "approved"
                    key = "report_filter" if filtered else "approval_filter"
                    sql = (
                        f"SELECT id FROM synthetic_capacity WHERE {clause} "
                        "ORDER BY vector <=> $1 LIMIT 30"
                    )
                    for scan in ("off", "strict_order"):
                        timings, recalls, returned = [], [], []
                        async with db.transaction():
                            # Force ANN for this experiment; record default and iterative scans.
                            await db.execute("SET LOCAL enable_seqscan=off")
                            await db.execute(f"SET LOCAL hnsw.iterative_scan='{scan}'")
                            for i, vector in enumerate(queries):
                                start = time.perf_counter()
                                rows = await db.fetch(sql, vector)
                                timings.append(1000 * (time.perf_counter() - start))
                                ids = {r["id"] for r in rows}
                                returned.append(len(ids))
                                recalls.append(
                                    len(ids & exact_ids[key, i]) / len(exact_ids[key, i])
                                )
                        record["hnsw"][f"{key}_{scan}"] = {
                            "milliseconds": timings,
                            "recall_vs_exact_at_30": recalls,
                            "returned": returned,
                        }
                record["bytes_with_hnsw"] = await db.fetchval(
                    "SELECT pg_total_relation_size('synthetic_capacity')"
                )
                await db.execute("DROP INDEX synthetic_hnsw")
            record["bytes_exact"] = await db.fetchval(
                "SELECT pg_total_relation_size('synthetic_capacity')"
            )
            results["sizes"].append(record)
            results["driver_peak_rss_native_units"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            results["decision"] = (
                "Keep exact retrieval; ANN is an unapproved future configuration change"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps({"completed_synthetic_chunks": target, "bytes": record["bytes_exact"]}),
                flush=True,
            )
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluations/v2/results/synthetic-capacity.json")
    )
    args = parser.parse_args()
    asyncio.run(run(args.database_url, args.output))
