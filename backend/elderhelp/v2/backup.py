"""Local, private PostgreSQL backups. Passwords are supplied through subprocess environment."""

import os
import subprocess
import tempfile
from pathlib import Path

from sqlalchemy.engine import make_url


def connection_environment(url: str, *, tls: str = "disable", ca_file: Path | None = None):
    parsed = make_url(url)
    return {
        **os.environ,
        "PGHOST": parsed.host or "localhost",
        "PGPORT": str(parsed.port or 5432),
        "PGUSER": parsed.username or "",
        "PGPASSWORD": parsed.password or "",
        "PGDATABASE": parsed.database or "",
        "PGSSLMODE": "verify-full"
        if tls == "verify-full"
        else parsed.query.get("sslmode", "prefer"),
        **({"PGSSLROOTCERT": str(ca_file)} if ca_file else {}),
    }


def backup(url: str, destination: Path, *, tls: str = "disable", ca_file: Path | None = None):
    if destination.exists():
        raise ValueError("Backup destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".dump") as file:
        os.chmod(file.name, 0o600)
        subprocess.run(
            ["pg_dump", "--format=custom", "--no-owner", "--no-privileges", "--file", file.name],
            env=connection_environment(url, tls=tls, ca_file=ca_file),
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["pg_restore", "--list", file.name], check=True, capture_output=True, timeout=30
        )
        # Exclusive destination creation avoids replacing any prior backup.
        with destination.open("xb") as output:
            os.chmod(destination, 0o600)
            file.seek(0)
            import shutil

            shutil.copyfileobj(file, output)
    return {"backup": str(destination), "bytes": destination.stat().st_size}


def restore_check(url: str, archive: Path, *, tls: str = "disable", ca_file: Path | None = None):
    name = make_url(url).database or ""
    if not name.endswith("_restore_test"):
        raise ValueError("Restore drill requires an empty database ending in _restore_test")
    subprocess.run(
        [
            "pg_restore",
            "--no-owner",
            "--no-privileges",
            "--exit-on-error",
            "--single-transaction",
            "--dbname",
            name,
            str(archive),
        ],
        env=connection_environment(url, tls=tls, ca_file=ca_file),
        check=True,
        capture_output=True,
        timeout=300,
    )
    return {"restored": True, "database": name}
