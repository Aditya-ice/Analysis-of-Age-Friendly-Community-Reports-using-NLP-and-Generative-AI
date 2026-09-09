import ssl
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from elderhelp.config import Settings


def connection_options(settings: Settings) -> dict:
    if settings.database_tls == "verify-full":
        return {"ssl": ssl.create_default_context(cafile=settings.database_ca_file)}
    return {}


class Database:
    def __init__(self, settings: Settings) -> None:
        options = (
            {}
            if settings.database_url.startswith("sqlite")
            else {"pool_size": 2, "max_overflow": 0, "pool_timeout": 5}
        )
        self.engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            hide_parameters=True,
            **options,
            connect_args=connection_options(settings),
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session

    async def close(self) -> None:
        await self.engine.dispose()
