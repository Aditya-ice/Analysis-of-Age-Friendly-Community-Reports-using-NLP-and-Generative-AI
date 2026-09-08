import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from elderhelp.v2.models import Invite
from elderhelp.v2.quota import reserve

bearer = HTTPBearer(auto_error=False)


def secret(settings):
    if not settings.token_secret or len(settings.token_secret.get_secret_value()) < 32:
        raise HTTPException(503, "Pilot access is not configured")
    return settings.token_secret.get_secret_value().encode()


def digest(settings, purpose, value):
    return hmac.new(secret(settings), f"{purpose}\0{value}".encode(), hashlib.sha256).hexdigest()


def b64(value: bytes):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def decode(value: str):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass
class Pilot:
    invite_id: UUID
    session_id: UUID


async def issue_session(database, settings, code: str, client_address: str):
    now = datetime.now(UTC)
    window = int(now.timestamp()) // 60
    expires = datetime.fromtimestamp((window + 1) * 60, UTC)
    await reserve(
        database,
        [
            (f"login:{window}", 1, 30, expires),
            (f"login:{digest(settings, 'address', client_address)}:{window}", 1, 5, expires),
        ],
    )
    code_hash = digest(settings, "invite", code)
    async with database.sessions() as db:
        invite = await db.scalar(
            select(Invite).where(Invite.code_hash == code_hash, ~Invite.revoked)
        )
    if not invite:
        raise HTTPException(401, "Invite code is invalid or revoked")
    expiry = int(time.time()) + settings.token_ttl_seconds
    payload = b64(
        json.dumps(
            {"invite": str(invite.id), "session": str(uuid4()), "exp": expiry, "v": 1},
            separators=(",", ":"),
        ).encode()
    )
    signature = b64(hmac.new(secret(settings), payload.encode(), hashlib.sha256).digest())
    return {"token": payload + "." + signature, "expires_at": expiry}


async def authenticate(
    request: Request, header: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> Pilot:
    settings = request.app.state.settings
    if not header or header.scheme.lower() != "bearer" or len(header.credentials) > 1024:
        raise HTTPException(401, "Enter an invite code to use the pilot")
    try:
        payload, signature = header.credentials.split(".")
        expected = hmac.new(secret(settings), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(decode(signature), expected):
            raise ValueError("Signature")
        claims = json.loads(decode(payload))
        if claims["v"] != 1 or claims["exp"] <= int(time.time()):
            raise ValueError("Expired")
        pilot = Pilot(UUID(claims["invite"]), UUID(claims["session"]))
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise HTTPException(
            401, "Pilot session expired or invalid; enter your invite code again"
        ) from None
    async with request.app.state.database.sessions() as db:
        invite = await db.get(Invite, pilot.invite_id)
        if not invite or invite.revoked:
            raise HTTPException(401, "This invite has been revoked")
    return pilot


async def create_invite(database, settings):
    code = secrets.token_urlsafe(24)
    row = Invite(id=uuid4(), code_hash=digest(settings, "invite", code))
    async with database.sessions() as db, db.begin():
        db.add(row)
    return {"id": str(row.id), "invite_code": code}
