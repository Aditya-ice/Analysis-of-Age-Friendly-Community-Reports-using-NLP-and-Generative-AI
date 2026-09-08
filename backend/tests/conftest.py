import os

import pytest_asyncio
from elderhelp.config import Settings
from elderhelp.database import Database
from sqlalchemy import text


@pytest_asyncio.fixture
async def research_db():
    """Use only a dedicated disposable database, after migrations have run."""
    import pytest

    url = os.environ.get("ELDERHELP_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set ELDERHELP_TEST_DATABASE_URL to a disposable migrated PostgreSQL database")
    database = Database(Settings(database_url=url))
    async with database.sessions() as db:
        name = await db.scalar(text("SELECT current_database()"))
        if not name.endswith("_test"):
            raise RuntimeError("Integration tests require a database name ending in _test")
        await db.execute(
            text(
                "TRUNCATE reports, index_generations, demo_invites, "
                "quota_counters, chunk_embeddings CASCADE"
            )
        )
        await db.commit()
    yield database
    await database.close()
