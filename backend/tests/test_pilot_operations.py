import ssl
from pathlib import Path

import pytest
from elderhelp.config import Settings
from elderhelp.database import connection_options
from elderhelp.v2.backup import connection_environment
from sqlalchemy import text

ROLE_SQL = Path("infra/free-pilot/roles.sql").read_text()


def test_hosted_connections_verify_certificates_and_backup_tls():
    with pytest.raises(ValueError, match="verified database TLS"):
        Settings(environment="pilot", database_tls="disable")
    settings = Settings(environment="pilot", database_tls="verify-full")
    context = connection_options(settings)["ssl"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    env = connection_environment(
        "postgresql+asyncpg://user:p%25ss@db.example.org:5432/pilot",
        tls="verify-full",
        ca_file=Path("/etc/secrets/ca.crt"),
    )
    assert env["PGSSLMODE"] == "verify-full"
    assert env["PGSSLROOTCERT"] == "/etc/secrets/ca.crt"
    assert env["PGPASSWORD"] == "p%ss"
    with pytest.raises(FileNotFoundError):
        connection_options(Settings(database_tls="verify-full", database_ca_file="/missing-ca"))


async def test_database_roles_block_corpus_mutation_and_public_access(research_db):
    async with research_db.engine.connect() as connection:
        transaction = await connection.begin()
        try:
            raw = await connection.get_raw_connection()
            await raw.driver_connection.execute("""
                DO $test$ BEGIN
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='anon') THEN
                        CREATE ROLE anon NOLOGIN;
                    END IF;
                END $test$;
                GRANT SELECT ON reports TO anon;
            """)
            # asyncpg execute supports the administrator script's multiple statements.
            await raw.driver_connection.execute(ROLE_SQL)
            await raw.driver_connection.execute(ROLE_SQL)  # Reapplying policy is safe.
            assert not await connection.scalar(
                text("SELECT has_table_privilege('anon','reports','SELECT')")
            )
            assert await connection.scalar(
                text("SELECT has_table_privilege('elderhelp_serving','reports','SELECT')")
            )
            assert not await connection.scalar(
                text("SELECT has_table_privilege('elderhelp_serving','reports','UPDATE')")
            )
            assert await connection.scalar(
                text("SELECT has_table_privilege('elderhelp_ingestion','reports','UPDATE')")
            )
            await connection.execute(text("SET LOCAL ROLE elderhelp_serving"))
            assert await connection.scalar(text("SELECT count(*) FROM reports")) == 0
            await connection.execute(
                text(
                    "INSERT INTO quota_counters (key, used, expires_at) "
                    "VALUES ('role-test', 1, now() + interval '1 hour')"
                )
            )
            await connection.execute(text("RESET ROLE"))
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM quota_counters WHERE key='role-test'")
                )
                == 1
            )
        finally:
            await transaction.rollback()


async def test_pruning_archives_only_inactive_search_data(research_db, tmp_path):
    import asyncio
    import json

    from elderhelp.v2.corpus import activate, index_configuration
    from elderhelp.v2.maintenance import prune
    from elderhelp.v2.models import Generation
    from test_corpus_v2 import populated_generation

    generations = []
    for _ in range(3):
        generation = await populated_generation(research_db)
        await activate(research_db, generation, index_configuration(Settings()))
        generations.append(generation)
    for protected in generations[1:]:
        with pytest.raises(ValueError, match="kept"):
            await prune(research_db, protected, tmp_path, apply=True)
    preview = await prune(research_db, generations[0], tmp_path)
    assert preview["candidate_chunks"] == 1 and not preview["applied"]
    result = await prune(research_db, generations[0], tmp_path, apply=True)
    archive = json.loads(await asyncio.to_thread(Path(result["archive"]).read_text))
    assert archive["generation_id"] == str(generations[0]) and len(archive["chunks"]) == 1
    async with research_db.sessions() as db:
        assert (await db.get(Generation, generations[0])).status == "archived"
        assert await db.scalar(text("SELECT count(*) FROM research_chunks")) == 2
        assert await db.scalar(text("SELECT count(*) FROM chunk_embeddings")) == 2
        assert await db.scalar(text("SELECT count(*) FROM source_spans")) == 3


async def test_activation_cannot_publish_generation_pruned_after_validation(
    research_db, tmp_path, monkeypatch
):
    from elderhelp.v2 import corpus
    from elderhelp.v2.maintenance import prune
    from test_corpus_v2 import populated_generation

    config = corpus.index_configuration(Settings())
    ids = []
    for _ in range(3):
        item = await populated_generation(research_db)
        await corpus.activate(research_db, item, config)
        ids.append(item)
    original = corpus.validate_generation

    async def validate_then_prune(database, generation):
        result = await original(database, generation)
        await prune(database, generation, tmp_path, apply=True)
        return result

    monkeypatch.setattr(corpus, "validate_generation", validate_then_prune)
    with pytest.raises(ValueError, match="changed after validation"):
        await corpus.activate(research_db, ids[0], config)
    assert (await corpus.status(research_db))["active"] == str(ids[2])


async def test_cancelled_reranker_stops_remaining_candidates_before_releasing_lock():
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    from elderhelp.v2.reranker import OnnxRanker

    ranker = object.__new__(OnnxRanker)
    ranker.lock = asyncio.Lock()
    entered = threading.Event()
    calls = []
    running = 0

    def score(question, content, *, stop=None):
        nonlocal running
        running += 1
        assert running == 1
        calls.append(question)
        entered.set()
        time.sleep(0.03)  # One uninterruptible native batch.
        running -= 1
        return 1.0

    ranker.score = score

    def candidate(i):
        return SimpleNamespace(id=i, title="Report", content="Evidence", score=0)

    first = asyncio.create_task(ranker.rerank("cancel", [candidate(i) for i in range(20)]))
    assert await asyncio.to_thread(entered.wait, 1)
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()  # A disconnect can arrive after the workflow deadline cancels it.
    second = asyncio.create_task(ranker.rerank("next", [candidate(21)]))
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 2)
    assert calls == ["cancel", "next"]


async def test_supabase_extension_schema_is_visible_to_serving_role(research_db):
    async with research_db.engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(text("CREATE SCHEMA IF NOT EXISTS extensions"))
            await connection.execute(text("ALTER EXTENSION vector SET SCHEMA extensions"))
            raw = await connection.get_raw_connection()
            await raw.driver_connection.execute(ROLE_SQL)
            await connection.execute(text("SET LOCAL ROLE elderhelp_serving"))
            distance = await connection.scalar(text("SELECT '[1,0]'::vector <=> '[1,0]'::vector"))
            assert distance == 0
        finally:
            await transaction.rollback()
