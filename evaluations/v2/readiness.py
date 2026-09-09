"""Read-only ingestion/evaluation preflight. Never creates a provider or makes model calls."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from elderhelp.config import Settings
from elderhelp.database import Database
from elderhelp.manifest import load_manifest
from elderhelp.v2.corpus import fingerprint, index_configuration
from integrity import audit, reviewed
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]


def local_checks(settings, root=ROOT):
    checks, blockers = {}, []
    try:
        manifest = load_manifest(root / settings.report_manifest)
        assets = []
        for report in manifest.reports:
            path = (root / report.local_path).resolve()
            path.relative_to((root / settings.seed_root).resolve())
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            assets.append(
                {
                    "report_slug": report.slug,
                    "checksum_matches": actual == report.expected_sha256,
                    "approved": report.status == "approved",
                }
            )
        checks["seed_reports"] = assets
        checks["seed_assets_valid"] = bool(assets) and all(
            a["checksum_matches"] and a["approved"] for a in assets
        )
    except (OSError, ValueError):
        checks["seed_assets_valid"] = False
    if not checks["seed_assets_valid"]:
        blockers.append("seed_assets_invalid")
    checks["extraction_dependency_present"] = importlib.util.find_spec("pymupdf") is not None
    checks["tesseract_present"] = shutil.which("tesseract") is not None
    for field in ("extraction_dependency_present", "tesseract_present"):
        if not checks[field]:
            blockers.append(field.replace("_present", "_missing"))
    source = root / "evaluations/v2/cases.jsonl"
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    metadata = json.loads((source.parent / "dataset.json").read_text())
    checks["dataset_integrity_valid"] = (
        not audit(cases)
        and hashlib.sha256(source.read_bytes()).hexdigest() == metadata["dataset_sha256"]
    )
    checks["index_configuration_matches_dataset"] = (
        fingerprint(index_configuration(settings)) == metadata["index_fingerprint"]
    )
    checks["human_reviewed_development_cases"] = sum(
        c["split"] == "development" and reviewed(c["review"]) for c in cases
    )
    checks["google_key_present"] = bool(
        settings.google_api_key and settings.google_api_key.get_secret_value()
    )
    checks["free_tier_confirmed"] = settings.free_tier_confirmed
    checks["ingestion_authorized"] = (
        settings.free_tier_confirmed or settings.paid_ingestion_confirmed
    )
    checks["token_secret_present"] = bool(settings.token_secret)
    for field in (
        "google_key_present",
        "free_tier_confirmed",
        "token_secret_present",
        "dataset_integrity_valid",
        "index_configuration_matches_dataset",
    ):
        if not checks[field]:
            blockers.append(field + "_required")
    if not checks["human_reviewed_development_cases"]:
        blockers.append("development_evidence_review_required")
    return checks, blockers


async def database_checks(settings):
    database = None
    try:
        database = Database(settings)
        async with asyncio.timeout(8):
            async with database.sessions() as db:
                await db.execute(text("SELECT 1"))
                version = await db.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                )
                active = await db.scalar(
                    text(
                        "SELECT g.fingerprint FROM active_corpus a "
                        "JOIN index_generations g ON g.id=a.generation_id WHERE a.id=1"
                    )
                )
        return {
            "database_available": True,
            "pgvector_present": bool(version),
            "compatible_active_corpus": active == fingerprint(index_configuration(settings)),
        }
    except Exception as error:
        # Never include exception text: connection errors may contain host/user credentials.
        return {
            "database_available": False,
            "pgvector_present": False,
            "compatible_active_corpus": False,
            "error_category": type(error).__name__,
        }
    finally:
        if database is not None:
            await database.close()


async def preflight(settings, check_database=False):
    checks, blockers = local_checks(settings)
    if check_database:
        checks.update(await database_checks(settings))
        if not checks["database_available"] or not checks["pgvector_present"]:
            blockers.append("database_setup_required")
        if not checks["compatible_active_corpus"]:
            blockers.append("compatible_corpus_ingestion_and_activation_required")
    else:
        checks["database_checked"] = False
        blockers.append("database_readiness_not_checked")
    ingest_fields = (
        "seed_assets_valid",
        "extraction_dependency_present",
        "tesseract_present",
        "google_key_present",
        "ingestion_authorized",
        "database_available",
        "pgvector_present",
    )
    return {
        "model_calls_made": 0,
        "writes_to_database": 0,
        "checks": checks,
        "ingestion_ready": all(checks.get(k, False) for k in ingest_fields),
        "evaluation_ready": not blockers,
        "release_quality_certified": False,
        "blockers": blockers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-database",
        action="store_true",
        help="Run read-only checks against the configured database; no ingestion",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(preflight(Settings(), args.check_database))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    raise SystemExit(0 if result["evaluation_ready"] else 2)


if __name__ == "__main__":
    main()
