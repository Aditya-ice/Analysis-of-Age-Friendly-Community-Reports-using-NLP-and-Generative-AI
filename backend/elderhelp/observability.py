"""ASGI metrics and bounded requests, without retaining request/response text."""

import json
import logging
import time
from contextvars import ContextVar
from uuid import UUID, uuid4

from starlette.responses import JSONResponse

logger = logging.getLogger("elderhelp.http")
request_context = ContextVar("request_id", default=None)


class RequestMetricsMiddleware:
    def __init__(self, app, max_request_bytes=65536):
        self.app, self.maximum = app, max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started, status = time.monotonic(), 500
        headers = dict(scope.get("headers", []))
        try:
            request_id = UUID(headers.get(b"x-request-id", b"").decode("ascii"))
        except (ValueError, UnicodeError):
            request_id = uuid4()
        scope.setdefault("state", {})["request_id"] = request_id
        if (
            headers.get(b"content-length", b"0").isdigit()
            and int(headers.get(b"content-length", b"0")) > self.maximum
        ):
            await JSONResponse({"detail": "Request exceeds the 64 KB pilot limit"}, 413)(
                scope, receive, send
            )
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.maximum:
                await JSONResponse({"detail": "Request exceeds the 64 KB pilot limit"}, 413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", str(request_id).encode()),
                ]
            await send(message)

        context_token = request_context.set(str(request_id))
        try:
            await self.app(scope, bounded_receive, measured_send)
        finally:
            request_context.reset(context_token)
            # Route templates, not user-controlled URL paths/query strings.
            route = scope.get("route")
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": str(request_id),
                        "method": scope["method"],
                        "route": getattr(route, "path", "unmatched"),
                        "status_code": status,
                        "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    }
                )
            )
