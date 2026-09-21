import httpx
from elderhelp.config import Settings
from elderhelp.main import create_app


async def test_browser_shell_works_without_database_and_does_not_expose_files():
    app = create_app(Settings(_env_file=None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        result = await client.get("/")
        assert result.status_code == 200 and "Use non-sensitive research questions" in result.text
        assert "frame-ancestors 'none'" in result.headers["Content-Security-Policy"]
        assert (await client.get("/assets/app.mjs")).status_code == 200
        assert (await client.get("/assets/__init__.py")).status_code == 404
        assert (await client.get("/data/seed/pdfs/AgeFriendlyNYC2017.pdf")).status_code == 404
        worker = await client.get("/sw.js")
        assert worker.status_code == 200 and "Cache only the application shell" in worker.text
    await app.state.database.close()
