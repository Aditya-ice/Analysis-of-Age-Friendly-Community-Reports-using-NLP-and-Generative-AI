from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def install(app):
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "index.html", headers=HEADERS)

    @app.get("/sw.js", include_in_schema=False)
    async def worker():
        return FileResponse(
            ROOT / "sw.js",
            media_type="text/javascript",
            headers={**HEADERS, "Cache-Control": "no-cache"},
        )

    @app.get("/assets/{name}", include_in_schema=False)
    async def asset(name: str):
        if name not in {"style.css", "app.mjs", "core.mjs", "storage.mjs"}:
            raise HTTPException(404)
        return FileResponse(
            ROOT / name,
            headers=HEADERS,
            media_type="text/css" if name.endswith(".css") else "text/javascript",
        )
