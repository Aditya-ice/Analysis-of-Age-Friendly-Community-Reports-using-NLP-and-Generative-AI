import json
import logging
import time
from uuid import UUID, uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("elderhelp.http")


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.monotonic()
        try:
            request_id = UUID(request.headers.get("X-Request-ID", ""))
        except ValueError:
            request_id = uuid4()
        request.state.request_id = request_id
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = str(request_id)
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": str(request_id),
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    }
                )
            )
