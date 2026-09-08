import asyncio
import json
from pathlib import Path

import typer

from elderhelp.config import get_settings
from elderhelp.services.ingestion import ingest
from elderhelp.v2.commands import app as corpus_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


@app.command()
def ingest_reports(repository_root: Path = Path(".")) -> None:
    """Index approved reports from the configured manifest."""
    result = asyncio.run(ingest(get_settings(), repository_root.resolve()))
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
