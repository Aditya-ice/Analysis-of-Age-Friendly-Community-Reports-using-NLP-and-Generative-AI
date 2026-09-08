"""Content-free PostgreSQL counters; admission survives process restarts."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, update
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
    now = datetime.now(UTC)
    end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return f"{name}:{now:%Y-%m-%d}", amount, limit, end


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
