"""Content-free PostgreSQL counters; admission survives process restarts."""

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from elderhelp.v2.models import QuotaCounter


class QuotaExceeded(RuntimeError):
    def __init__(self, retry_after: int = 60):
        super().__init__("Free pilot quota exhausted")
        self.retry_after = retry_after


async def reserve(database, limits: list[tuple[str, int, int, datetime]]) -> None:
    """All reservations succeed atomically or none do, including across API processes."""
    async with database.sessions() as db, db.begin():
        for key, amount, maximum, expires in sorted(limits):
            if amount > maximum:
                raise QuotaExceeded(max(1, int((expires - datetime.now(UTC)).total_seconds())))
            statement = insert(QuotaCounter).values(key=key, used=amount, expires_at=expires)
            result = await db.scalar(
                statement.on_conflict_do_update(
                    index_elements=[QuotaCounter.key],
                    set_={"used": QuotaCounter.used + amount},
                    where=QuotaCounter.used + amount <= maximum,
                ).returning(QuotaCounter.used)
            )
            if result is None:
                raise QuotaExceeded(max(1, int((expires - datetime.now(UTC)).total_seconds())))
        await db.execute(
            delete(QuotaCounter).where(
                QuotaCounter.expires_at < datetime.now(UTC) - timedelta(days=2)
            )
        )


async def release(database, amounts: dict[str, int]):
    async with database.sessions() as db, db.begin():
        for key, amount in amounts.items():
            if amount > 0:
                await db.execute(
                    update(QuotaCounter)
                    .where(QuotaCounter.key == key, QuotaCounter.used >= amount)
                    .values(used=QuotaCounter.used - amount)
                )


def daily(name: str, amount: int, limit: int):
    zone = ZoneInfo("America/Los_Angeles") if name in ("generation", "embedding") else UTC
    now = datetime.now(zone)
    end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return f"{name}:{now:%Y-%m-%d}", amount, limit, end


async def pace(database, name: str, limit: int):
    """Space dispatches using a shared database clock, with no transaction while waiting."""
    if limit < 1:
        raise QuotaExceeded()
    interval = timedelta(seconds=60 / limit + 0.05)
    while True:
        async with database.sessions() as db, db.begin():
            now = func.clock_timestamp()
            statement = insert(QuotaCounter).values(
                key=f"google:{name}:pace", used=1, expires_at=now + interval
            )
            granted = await db.scalar(
                statement.on_conflict_do_update(
                    index_elements=[QuotaCounter.key],
                    set_={"expires_at": now + interval, "used": 1},
                    where=QuotaCounter.expires_at <= now,
                ).returning(QuotaCounter.key)
            )
            if granted:
                return
            remaining = await db.scalar(
                select(
                    func.extract("epoch", QuotaCounter.expires_at - func.clock_timestamp())
                ).where(QuotaCounter.key == f"google:{name}:pace")
            )
        await asyncio.sleep(min(60, max(0.05, float(remaining))))


async def pause_google(database, seconds=60):
    expires = datetime.now(UTC) + timedelta(seconds=seconds)
    async with database.sessions() as db, db.begin():
        statement = insert(QuotaCounter).values(key="google:cooldown", used=1, expires_at=expires)
        await db.execute(
            statement.on_conflict_do_update(
                index_elements=[QuotaCounter.key], set_={"expires_at": expires, "used": 1}
            )
        )


async def google_retry_after(database):
    async with database.sessions() as db:
        row = await db.get(QuotaCounter, "google:cooldown")
        if row:
            return max(0, int((row.expires_at - datetime.now(UTC)).total_seconds()))
    return 0
