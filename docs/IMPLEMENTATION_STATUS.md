# Reliable RAG implementation checkpoints

Baseline: `8ae26b8`. Original code/history, seed reports and mobile apps are preserved.

| Step | State | Verification |
|---|---|---|
| 1. Audit and v2 contract | Implemented | Contract/schema tests; source audit and reproduced grounding failures |
| 2. Corpus lifecycle | Implemented | PostgreSQL 16/pgvector migrations; 27 tests, including metadata-only changes, duplicate hashes, exact-span validation, atomic activation, incompatible configuration rejection and withdrawal-safe rollback |
| 3. Extraction and ingestion | Implemented | 34 backend tests; digital/mixed/Tesseract extraction, safe downloads, resumable embeddings, leases and quota races; both seed reports extracted locally; backup restored into a disposable database. Live Google ingestion awaits free-tier configuration. |
| 4. Retrieval | Implemented | 40 backend tests; real PostgreSQL dense/keyword filters, child preservation, resolved follow-ups, withdrawal and RRF fallback; pinned quantized ONNX short/long-candidate smoke tests pass. Corpus quality metrics remain unmeasured. |
| 5. Verified answers/API | Implemented | 51 backend tests; no draft before verification, fabricated claims, withdrawal, cancellation, timeout, v1 mapping, access/quota controls and ten-request burst. Serving-only Python 3.12 smoke passed without OCR/paid-cloud libraries (159,547,392-byte peak RSS on macOS; not a Linux capacity result). Live quality and latency gates remain pending. |
| 6. Evaluation | Harness implemented; quality gate blocked | 64 backend tests; 120 candidate cases in an 80/40 grouped split; resumable four-way ablations and human-review scoring. Local synthetic 1k/10k/50k PostgreSQL experiments recorded. Zero human-gold cases or live Google outputs yet; quality and hosted latency remain unmeasured. |
| 7. Browser/mobile clients | Implemented; fixture platform checks pass | Shared v2 fixtures, browser protocol/static tests and manual browser flow pass. At `80f3073`, full iOS simulator (including Keychain and citations) and Android build/lint/emulator checks pass in PR12. Live staging and assistive-technology checks remain pending. |
| 8. Free deployment/handover | Operations implemented; deployment blocked | 69 backend tests pass locally, including verified TLS, database roles, archived pruning and pruning/activation races. Render/Supabase setup and restore/runbook prepared. Linux serving-image capacity CI pending; human-reviewed quality, live free-account evaluation and hosted acceptance still required. |

Each completed step is committed and pushed separately. No quality scores, live-model results or deployment are claimed until actually measured. Human gold-label review, free-tier account configuration and live quota are external gates; paid services are prohibited.
