# ElderHelp

ElderHelp is an evidence-grounded question-answering system for approved age-friendly
community reports. The new service preserves report and page provenance, combines semantic
and keyword retrieval, reranks evidence, and streams answers with validated citations.

The repository currently contains the Python backend and a native iOS simulator prototype. The
original research prototype is preserved in [`legacy/`](legacy/README.md).

## Repository layout

- `backend/elderhelp/` — FastAPI application, ingestion pipeline, and modern RAG services.
- `backend/tests/` — offline unit and API-contract tests.
- `data/reports.yaml` — curated report manifest; PDFs under `data/seed/` are private seed inputs.
- `ios/ElderHelp/` — SwiftUI client with Ask, Reports, and local History features.
- `infra/terraform/` — Google Cloud Run, Cloud SQL, private Storage, IAM, and WIF resources.
- `evaluations/` — versioned retrieval and grounded-answer evaluation cases.
- `legacy/` — preserved Flask, LangChain, OCR, NLP, and extracted-text prototype.

The Android client follows after the API and iOS interaction contract stabilize.

## Local backend

Requirements: Python 3.12 and `uv`. PostgreSQL 16 with the `vector` extension is required for
the catalog and retrieval endpoints.

```sh
cp .env.example .env
uv sync --frozen --group dev
docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn elderhelp.main:app --reload
```

The process health endpoint works without provider credentials. Retrieval and readiness require
Application Default Credentials and `ELDERHELP_GOOGLE_CLOUD_PROJECT`.

```sh
uv run pytest
uv run ruff check backend
uv run python -m elderhelp.openapi
uv run elderhelp ingest-reports .
```

Ingestion sends report text to the configured Vertex AI embedding model. It invokes Document AI
only when a processor is configured and direct page extraction has poor quality. It uploads PDFs
only when a private storage bucket is configured.

## Privacy and report access

The API does not persist questions, answers, or conversation history. Mobile history remains on
the device. Full PDFs are not returned by the public API; citations contain a short excerpt,
publisher attribution, page number, and publisher URL.

This repository does not yet state redistribution rights for the seed reports. Confirm those
rights before distributing their full contents or enabling public downloads.

## Android prototype

The native Kotlin/Compose app is in [`android/`](android/README.md), with Ask, Reports, local History, and source citation sheets. Build and emulator instructions are in its README. Both mobile clients use the same versioned backend API; Android DTOs are generated from `openapi.json`.
