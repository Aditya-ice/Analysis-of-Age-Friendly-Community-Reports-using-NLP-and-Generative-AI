import asyncio
import json
from pathlib import Path
from uuid import UUID

import typer

from elderhelp.config import get_settings
from elderhelp.database import Database
from elderhelp.manifest import load_manifest
from elderhelp.v2.commands import app as corpus_app

app = typer.Typer(no_args_is_help=True)
manifest_app = typer.Typer(no_args_is_help=True)
ingest_app = typer.Typer(invoke_without_command=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(manifest_app, name="manifest")
app.add_typer(ingest_app, name="ingest")


@manifest_app.command("validate")
def manifest_validate():
    """Validate structure, approval policy and pinned source checksums without network calls."""
    manifest = load_manifest(get_settings().report_manifest)
    for item in manifest.reports:
        if not item.expected_sha256 or not item.allowed_domains:
            raise typer.BadParameter(f"{item.slug}: checksum and approved domains required")
        if item.source_url.scheme != "https" or item.source_url.host not in item.allowed_domains:
            raise typer.BadParameter(f"{item.slug}: source hostname must be explicitly approved")
    typer.echo(json.dumps({"valid": True, "reports": len(manifest.reports)}))


def run_ingestion(root: Path, generation: UUID | None = None):
    from elderhelp.v2.google import Gemini
    from elderhelp.v2.ingestion import ingest

    root = root.resolve()

    async def execute():
        settings = get_settings()
        database = Database(settings)
        provider = None
        try:
            provider = Gemini(settings, database)
            return await ingest(
                database,
                provider,
                settings,
                root,
                load_manifest(root / settings.report_manifest) if not generation else None,
                generation_id=generation,
            )
        finally:
            if provider:
                await provider.close()
            await database.close()

    typer.echo(json.dumps(asyncio.run(execute()), indent=2))


@ingest_app.callback()
def ingest_start(ctx: typer.Context, repository_root: Path = Path(".")):
    """Stage the configured corpus; activation is a separate administrator action."""
    if ctx.invoked_subcommand is None:
        run_ingestion(repository_root)


@ingest_app.command()
def resume(generation: UUID, repository_root: Path = Path(".")):
    run_ingestion(repository_root, generation)


@app.command()
def backup(destination: Path):
    from elderhelp.v2.backup import backup as create_backup

    typer.echo(json.dumps(create_backup(get_settings().database_url, destination)))


@app.command()
def restore_check(archive: Path):
    """Restore into the empty *_restore_test database configured in ELDERHELP_DATABASE_URL."""
    from elderhelp.v2.backup import restore_check as restore

    typer.echo(json.dumps(restore(get_settings().database_url, archive)))


if __name__ == "__main__":
    app()
