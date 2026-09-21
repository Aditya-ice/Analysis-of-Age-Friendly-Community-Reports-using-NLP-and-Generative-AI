# ElderHelp

ElderHelp is being rebuilt as a research assistant for approved age-friendly community reports. The v2 backend retrieves exact passages, creates structured claims, and verifies them before displaying any answer text. It distinguishes historical report commitments from current services and falls back to keyword search when generation is unavailable.

**Implementation is ongoing.** Backend engineering checks are recorded in [the checkpoint ledger](docs/IMPLEMENTATION_STATUS.md). Fixture-based iOS and Android simulator checks pass. Live Google evaluation, human-reviewed quality gates, assistive-technology checks and free hosted deployment remain outstanding. The project does not claim production uptime or measured corpus accuracy yet.

## Local setup

Use Python 3.12, `uv`, PostgreSQL 16 with pgvector, and local Tesseract for ingestion only.

```sh
cp .env.example .env
uv sync --frozen --group dev --extra ingestion
docker compose up -d postgres
uv run alembic upgrade head
uv run python tools/download_reranker.py
uv run elderhelp manifest validate
uv run elderhelp corpus reconcile --dry-run
uv run elderhelp corpus reconcile --apply
```

Set `ELDERHELP_TOKEN_SECRET` privately to a random string of at least 32 characters. For live model work, configure a **billing-disabled Google Gemini Developer API project**, verify its actual free quota, set `ELDERHELP_GOOGLE_API_KEY`, and explicitly set `ELDERHELP_FREE_TIER_CONFIRMED=true`. Lower the example admission limits if the account allows less. No model/provider changes or paid fallback happen automatically.

```sh
uv run elderhelp ingest
# Save the returned generation UUID. Resume after interruptions/quota exhaustion:
uv run elderhelp ingest resume GENERATION_UUID
uv run elderhelp corpus validate GENERATION_UUID
uv run elderhelp corpus activate GENERATION_UUID
uv run elderhelp corpus invite-create
uv run uvicorn elderhelp.main:app --reload --no-access-log
```

The invite-creation command returns a revocable code once; it is not embedded in client builds. Exchange it at `POST /v2/demo/session`. Send the returned short-lived token as `Authorization: Bearer TOKEN` for report, search and answer endpoints. `/healthz` requires no database or model call; `/readyz` checks the database, compatible active corpus and local components without spending Google quota.

API reference: `/docs`; committed contract: [openapi.json](openapi.json). `/v1` uses the same corrected engine and maps richer results to a safe insufficient-evidence response. The browser at `/` and updated native clients accept invite codes at runtime. See [client setup and verification](docs/CLIENTS_V2.md).

## Verification

Use an isolated, migrated PostgreSQL database ending in `_test`. The integration fixture clears that test database.

```sh
ELDERHELP_DATABASE_URL=postgresql+asyncpg://USER:PASS@localhost:5432/elderhelp_test uv run alembic upgrade head
ELDERHELP_TEST_DATABASE_URL=postgresql+asyncpg://USER:PASS@localhost:5432/elderhelp_test uv run pytest
uv run ruff check backend tools
uv run python -m elderhelp.openapi
```

Tests use provider doubles; the optional real ONNX smoke test runs after downloading local artifacts. CI also exercises PostgreSQL/pgvector, Tesseract and the serving image. A second Gemini call is a fallible safeguard: human-reviewed evaluations must pass before deployment.

## Documentation and preserved work

- [Free Render/Supabase setup, verified TLS, roles, backup and release gates](infra/free-pilot/README.md)
- [Corpus reconciliation, activation and rollback](docs/CORPUS_OPERATIONS.md)
- [Safe ingestion, extraction and private backup/restore](docs/INGESTION_V2.md)
- [Hybrid retrieval and CPU ONNX reranking](docs/RETRIEVAL_V2.md)
- [Verified answers, API access and failure behavior](docs/VERIFIED_API_V2.md)
- [Evaluation protocol, human review and measured limitations](docs/EVALUATION_V2.md)
- [Swift prototype](ios/README.md) and [Kotlin prototype](android/README.md)
- [Original research prototype](legacy/README.md); historical v1 serving code under `legacy/backend_v1/`

Questions, answers and chat history are not stored by the server. Citations contain exact excerpts, dates, attribution and publisher URLs; the API does not publish PDFs or filesystem paths. Full-report redistribution and repository licensing remain unresolved. Google receives research questions and evidence when live models are enabled; submit only non-sensitive research questions under its unpaid-service terms.

The selected pilot target is Render Free plus Supabase Free. Google Cloud configuration in `infra/` is preserved as a future option and is excluded from this setup. No environment is deployed automatically, and existing history, reports and native app work are preserved.
