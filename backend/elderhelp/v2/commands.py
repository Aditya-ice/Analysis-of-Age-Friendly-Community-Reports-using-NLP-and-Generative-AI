"""Administrator-only operations: use ingestion database credentials, never serving credentials."""

import asyncio
import json
from uuid import UUID

import typer

from elderhelp.config import get_settings
from elderhelp.database import Database
from elderhelp.manifest import load_manifest
from elderhelp.v2 import corpus

app = typer.Typer(no_args_is_help=True)


def run(operation):
    async def execute():
        database = Database(get_settings())
        try:
            return await operation(database)
        finally:
            await database.close()

    try:
        typer.echo(json.dumps(asyncio.run(execute()), default=str, indent=2))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def reconcile(dry_run: bool = True, apply: bool = False):
    """Preview changes; --apply explicitly applies approval/metadata changes and removals."""
    if not dry_run and not apply:
        raise typer.BadParameter("Use --apply to apply the reviewed reconciliation")
    manifest = load_manifest(get_settings().report_manifest)
    run(lambda db: corpus.reconcile(db, manifest, apply=apply))


@app.command()
def validate(generation: UUID):
    run(lambda db: corpus.validate_generation(db, generation))


@app.command()
def activate(generation: UUID):
    run(lambda db: corpus.activate(db, generation, corpus.index_configuration(get_settings())))


@app.command()
def rollback():
    run(lambda db: corpus.rollback(db, corpus.index_configuration(get_settings())))


@app.command()
def status():
    run(corpus.status)
