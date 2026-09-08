# Verified answers and pilot API

The serving path now uses Gemini Developer API through `google-genai`. Historical Vertex/Ranking API and unchecked streaming implementations live in `legacy/backend_v1/`; they are not mounted or installed in the serving wheel. There is no automatic paid fallback. The serving image excludes PyMuPDF, Tesseract, PyTorch, Google Cloud service libraries, PDFs and legacy source directories.

## Answer workflow

Admission reserves the complete maximum workflow in PostgreSQL before headers are sent: two generation calls plus one embedding for a direct question, or up to three generation calls and three embeddings when planning is needed. Two answer workflows can run concurrently, with a global 20/day and 6/minute ceiling and additional invite/session limits. Unused model-call reservations are refunded; dispatched calls remain counted when outcomes are uncertain. A provider 429 creates a persistent temporary cooldown and leaves keyword search available. Lower configuration limits always apply.

A structured draft contains atomic claims, source-span IDs and missing components. Verification checks schema/coverage, known spans and exact offsets, quantities, units, entities and historical wording. A separate structured Gemini review judges support and coverage. The server renders Markdown only from accepted claims; unsupported claims are dropped. A central unsupported question receives insufficient evidence. A partially supported comparison identifies missing components. No repair/regeneration loop exists.

Immediately before returning a result, the backend rechecks current report approvals and the active generation. A changed corpus discards the answer. The verifier is fallible and requires calibration with human-reviewed cases; passing a schema or second model call is not a claim of truth.

SSE emits `start`, progress and heartbeat comments while working. Only after verification completes does it emit one prompt `delta` containing the full approved text, then `complete` with exactly that text and citations. It does not simulate typing. Operational verification failures emit typed `error` events with no answer text. Disconnects cancel asynchronous provider work and release workflow capacity. An already-running native ONNX call finishes under its lock before the next inference begins.

## Endpoints and access

- `POST /v2/demo/session`: exchange an administrator invite for a signed expiring pilot token.
- `GET /v2/capabilities`: generation/search availability, active index and temporary limitations.
- `GET /v2/reports`, `GET /v2/reports/{id}`: paginated approved catalog and revision metadata.
- `POST /v2/search`: keyword search, optionally hybrid; no generation/verification call.
- `POST /v2/answers/stream`: verified SSE answers.
- `/v1/reports`, `/v1/reports/{id}`, `/v1/answers/stream`: compatibility adapters to the same engine.
- `/healthz`, `/readyz`: public process/local dependency checks; no live model probes.

All corpus and answer endpoints require `Authorization: Bearer TOKEN`. Invite hashes and revocation persist in PostgreSQL; no personal accounts exist. Configure a secret of at least 32 characters. `uv run elderhelp corpus invite-create` returns the code once, and `uv run elderhelp corpus invite-revoke INVITE_UUID` invalidates its sessions. Login attempts are throttled using keyed address hashes, without storing raw addresses.

The API returns 401 for access failures, 413 for bodies over 64 KB, 422 for invalid fields, 429 with `Retry-After` before admission, and 503 for missing dependencies. After streaming begins, errors use the SSE schema. Questions retain the 2,000-character limit, six prior local turns and report/community/year filters. The server does not infer context from stored conversations.

`openapi.json` commits all HTTP schemas and SSE payload schemas with an `x-sse-events` mapping. Shared fixtures under `contracts/fixtures/` support client migration. V1 does not have a partial/clarification status, so those results become its insufficient-evidence response before any delta.

## Observability and operating limits

Request logs use request IDs and route templates, never question text, raw URL queries, answer bodies, evidence bodies or provider error details. Workflow records include code/model/prompt/index versions, outcome, timings, candidate/evidence counts and quota usage. Model logs include token counts when returned by Google. Curated evaluation traces are an explicit separate workflow.

Database pooling is two connections with no overflow and short acquisition timeouts; no database transaction is retained during model calls. The serving command uses one process and a bounded HTTP concurrency limit. Persisted counters/corpus survive restarts; the filesystem is not durable storage. Token secrets, keys and database credentials belong only in server secrets.

The warm-service objective is p95 verified completion within 30 seconds. It has **not** been measured against live Gemini. Cold starts and live-model performance require separate acceptance records; verification must not be weakened to meet a latency target.
